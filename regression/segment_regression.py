import joblib
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


# ==========================
# 1) CONFIGURATION
# ==========================
CSV_PATH = "../scraper/final_results_parsed.csv"
CLASSIFIER_PATH = "price_classifier.pkl"
OUTPUT_PATH = "price_moe_soft_with_classifier.pkl"

TARGET = "sold_price"

FEATURES = [
    "sold_price",
    "seller_type",
    "mileage",
    "primary_damage",
    "secondary_damage",
    "transmission",
    "fuel_type",
    "drive_type",
    "color",
    "year",
    "make",
    "model",
    "engine_volume",
    "cylinders",
]

NUM_COLS = [
    "year",
    "mileage",
    "engine_volume",
    "cylinders",
]

CAT_COLS = [
    "make",
    "model",
    "fuel_type",
    "transmission",
    "drive_type",
    "primary_damage",
    "secondary_damage",
]

BINS = [100, 2000, 5000, 10000, 20000, 50000, 80000, 100001]
MIN_SAMPLES_PER_BIN = 150
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ==========================
# 2) HELPER FUNCTIONS
# ==========================
def evaluate_regression(y_true, y_pred, title="MODEL"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n{title}")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def calc_regression_metrics(y_true, y_pred):
    return {
        "count": int(len(y_true)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
    }


def print_metrics_table(title: str, df_metrics: pd.DataFrame):
    print(f"\n{'=' * 90}")
    print(title)
    print('=' * 90)
    print(df_metrics.to_string(float_format=lambda x: f"{x:.4f}"))


def make_price_classes(y: pd.Series, bins) -> pd.Series:
    return pd.cut(y, bins=bins, right=False, include_lowest=True).astype(str)


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.dropna(subset=[TARGET])

    if "currency" in df.columns:
        df = df[df["currency"] != "CAD"].copy()

    df = df[(df["sold_price"] >= 500) & (df["sold_price"] <= 100000)].copy()

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CAT_COLS:
        df[col] = df[col].astype("string")

    for col in CAT_COLS:
        df[col] = df[col].fillna("UNKNOWN")

    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                NUM_COLS,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                CAT_COLS,
            ),
        ],
        remainder="drop",
    )


def build_regressor() -> Pipeline:
    return Pipeline([
        ("prep", build_preprocessor()),
        ("reg", XGBRegressor(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ))
    ])


def evaluate_by_true_segment(y_true, y_pred, true_segments, title):
    eval_df = pd.DataFrame({
        "true_price": np.asarray(y_true),
        "pred_price": np.asarray(y_pred),
        "true_segment": np.asarray(true_segments),
    })

    rows = []
    for seg in sorted(eval_df["true_segment"].unique()):
        g = eval_df[eval_df["true_segment"] == seg]
        rows.append({
            "segment": seg,
            **calc_regression_metrics(g["true_price"], g["pred_price"])
        })

    result = pd.DataFrame(rows).set_index("segment")
    print_metrics_table(title, result)
    return result


def get_classifier_classes(price_classifier, label_encoder):
    """
    Returns textual class labels in the same order
    as classifier.predict_proba() outputs probabilities.
    """
    if not hasattr(price_classifier, "classes_"):
        raise ValueError("Classifier does not have a classes_ attribute, cannot map predict_proba outputs to segments.")

    classes_raw = price_classifier.classes_

    if label_encoder is not None:
        return label_encoder.inverse_transform(classes_raw)
    return classes_raw


def soft_moe_predict(
    X: pd.DataFrame,
    classifier,
    label_encoder,
    expert_models: dict,
    global_regressor,
    class_temperature: float = 1.0,
    use_top_k: int | None = None,
):
    """
    Soft Mixture of Experts:
    - get class probabilities from the classifier
    - take expert model predictions for each class
    - final result = sum(weight_i * pred_i)

    class_temperature:
        <1.0 makes the probability distribution sharper
        >1.0 makes it smoother

    use_top_k:
        if specified, only the top-k most likely experts are used
    """
    proba = classifier.predict_proba(X)
    class_labels = np.asarray(get_classifier_classes(classifier, label_encoder), dtype=object)

    if proba.shape[1] != len(class_labels):
        raise ValueError("The number of predict_proba columns does not match the number of class labels.")

    # temperature scaling
    if class_temperature != 1.0:
        proba = np.power(proba, 1.0 / class_temperature)
        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        proba = proba / row_sums

    final_preds = []
    top1_classes = []
    routing_entropy = []

    for i in range(len(X)):
        row = X.iloc[[i]]
        row_proba = proba[i].copy()

        # top-k experts only
        if use_top_k is not None and use_top_k < len(row_proba):
            top_idx = np.argsort(row_proba)[-use_top_k:]
            mask = np.zeros_like(row_proba, dtype=bool)
            mask[top_idx] = True
            row_proba = np.where(mask, row_proba, 0.0)
            s = row_proba.sum()
            if s > 0:
                row_proba = row_proba / s

        pred_sum = 0.0

        for cls_label, weight in zip(class_labels, row_proba):
            if weight <= 0:
                continue

            if cls_label in expert_models:
                pred = expert_models[cls_label].predict(row)[0]
            else:
                pred = global_regressor.predict(row)[0]

            pred_sum += weight * pred

        final_preds.append(pred_sum)
        top1_classes.append(class_labels[np.argmax(proba[i])])

        # routing entropy to understand "confidence"
        eps = 1e-12
        entropy = -np.sum(proba[i] * np.log(proba[i] + eps))
        routing_entropy.append(entropy)

    return np.array(final_preds), np.array(top1_classes), np.array(routing_entropy), proba, class_labels


# ==========================
# 3) DATA LOADING
# ==========================
df = pd.read_csv(CSV_PATH)
df = prepare_dataframe(df)

X = df[FEATURES].copy()
y = df[TARGET].copy()
y_classes = make_price_classes(y, BINS)

#!!!!! stratify needs to be reviewed!!
X_train, X_test, y_train, y_test, y_train_cls, y_test_cls = train_test_split(
    X,
    y,
    y_classes,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_classes,
)

print("Train size:", len(X_train))
print("Test size :", len(X_test))

print("\nTrain class counts:")
print(y_train_cls.value_counts().sort_index())

print("\nTest class counts:")
print(y_test_cls.value_counts().sort_index())


# ==========================
# 4) LOAD PRICE CLASSIFIER
# ==========================
classifier_bundle = joblib.load(CLASSIFIER_PATH)

if hasattr(classifier_bundle, "predict"):
    price_classifier = classifier_bundle
    label_encoder = None
else:
    if "classifier" in classifier_bundle:
        price_classifier = classifier_bundle["classifier"]
    elif "model" in classifier_bundle:
        price_classifier = classifier_bundle["model"]
    else:
        raise ValueError("Could not find a classifier object inside price_classifier.pkl")

    label_encoder = classifier_bundle.get("label_encoder")

print("\nClassifier loaded from:", CLASSIFIER_PATH)


# ==========================
# 5) CLASSIFIER QUALITY CHECK
# ==========================
pred_test_cls_encoded = price_classifier.predict(X_test)

if label_encoder is not None:
    pred_test_cls = label_encoder.inverse_transform(pred_test_cls_encoded)
else:
    pred_test_cls = pred_test_cls_encoded

clf_acc = accuracy_score(y_test_cls, pred_test_cls)

print("\nCLASSIFIER QUALITY")
print("Accuracy:", round(clf_acc, 4))
print(classification_report(y_test_cls, pred_test_cls, zero_division=0))

print("\nPREDICTED CLASS COUNTS")
print(pd.Series(pred_test_cls).value_counts().sort_index())


# ==========================
# 6) TRAIN REGRESSORS FOR EACH INTERVAL
# ==========================
expert_models = {}
expert_counts = {}
expert_interval_metrics = []

for class_label in sorted(y_train_cls.unique()):
    train_mask = y_train_cls == class_label
    test_mask = y_test_cls == class_label

    n_train = int(train_mask.sum())
    n_test = int(test_mask.sum())

    if n_train < MIN_SAMPLES_PER_BIN:
        print(f"Skipping {class_label}: only {n_train} training rows")
        continue

    X_seg_train = X_train.loc[train_mask].copy()
    y_seg_train = y_train.loc[train_mask].copy()

    X_seg_test = X_test.loc[test_mask].copy()
    y_seg_test = y_test.loc[test_mask].copy()

    expert = build_regressor()
    expert.fit(X_seg_train, y_seg_train)

    expert_models[class_label] = expert
    expert_counts[class_label] = n_train

    print(f"Regressor trained for interval {class_label}: train={n_train}, test={n_test}")

    if n_test > 0:
        pred_seg_test = expert.predict(X_seg_test)
        metrics = calc_regression_metrics(y_seg_test, pred_seg_test)
        metrics["segment"] = class_label
        metrics["train_count"] = n_train
        metrics["test_count"] = n_test
        expert_interval_metrics.append(metrics)

expert_interval_metrics_df = pd.DataFrame(expert_interval_metrics)
if not expert_interval_metrics_df.empty:
    expert_interval_metrics_df = expert_interval_metrics_df.set_index("segment")
    print_metrics_table(
        "EXPERT REGRESSORS: PERFORMANCE ON THEIR OWN TRUE INTERVALS",
        expert_interval_metrics_df[["train_count", "test_count", "MAE", "RMSE", "R2"]],
    )
else:
    print("\nNo expert regressors available for evaluation.")


# ==========================
# 7) GLOBAL REGRESSOR
# ==========================
global_regressor = build_regressor()
global_regressor.fit(X_train, y_train)

global_preds = global_regressor.predict(X_test)
global_overall_metrics = evaluate_regression(
    y_test, global_preds, title="GLOBAL REGRESSOR: OVERALL"
)

global_by_segment_df = evaluate_by_true_segment(
    y_true=y_test,
    y_pred=global_preds,
    true_segments=y_test_cls,
    title="GLOBAL REGRESSOR: METRICS BY TRUE SEGMENT"
)


# ==========================
# 8) HARD ROUTING BASELINE
# ==========================
hard_preds = []
hard_used_model = []

for i in range(len(X_test)):
    row = X_test.iloc[[i]]
    pred_class = pred_test_cls[i]

    if pred_class in expert_models:
        pred_price = expert_models[pred_class].predict(row)[0]
        hard_used_model.append(pred_class)
    else:
        pred_price = global_regressor.predict(row)[0]
        hard_used_model.append("GLOBAL")

    hard_preds.append(pred_price)

hard_preds = np.array(hard_preds)

hard_overall_metrics = evaluate_regression(
    y_test, hard_preds, title="HARD ROUTING: CLASSIFIER -> ONE EXPERT"
)

hard_by_segment_df = evaluate_by_true_segment(
    y_true=y_test,
    y_pred=hard_preds,
    true_segments=y_test_cls,
    title="HARD ROUTING: METRICS BY TRUE SEGMENT"
)


# ==========================
# 9) SOFT MIXTURE OF EXPERTS
# ==========================
# possible values to try:
# class_temperature = 0.7 / 0.8 / 1.0 / 1.2
# use_top_k = 2 or 3
SOFT_TEMPERATURE = 0.8
SOFT_TOP_K = 2

soft_preds, soft_top1_classes, soft_entropy, soft_proba, soft_class_labels = soft_moe_predict(
    X=X_test,
    classifier=price_classifier,
    label_encoder=label_encoder,
    expert_models=expert_models,
    global_regressor=global_regressor,
    class_temperature=SOFT_TEMPERATURE,
    use_top_k=SOFT_TOP_K,
)

soft_overall_metrics = evaluate_regression(
    y_test, soft_preds, title=f"SOFT MOE: OVERALL (temperature={SOFT_TEMPERATURE}, top_k={SOFT_TOP_K})"
)

soft_by_segment_df = evaluate_by_true_segment(
    y_true=y_test,
    y_pred=soft_preds,
    true_segments=y_test_cls,
    title="SOFT MOE: METRICS BY TRUE SEGMENT"
)


# ==========================
# 10) COMPARISON TABLES
# ==========================
summary_overall_df = pd.DataFrame([
    {"model": "global_regressor", **global_overall_metrics},
    {"model": "hard_routing", **hard_overall_metrics},
    {"model": "soft_moe", **soft_overall_metrics},
]).set_index("model")

print_metrics_table("OVERALL MODEL COMPARISON", summary_overall_df)

per_segment_comparison_df = (
    global_by_segment_df.add_prefix("global_")
    .join(hard_by_segment_df.add_prefix("hard_"), how="outer")
    .join(soft_by_segment_df.add_prefix("soft_"), how="outer")
)

print_metrics_table(
    "COMPARISON BY SEGMENT: GLOBAL VS HARD ROUTING VS SOFT MOE",
    per_segment_comparison_df
)


# ==========================
# 11) DETAILED EVALUATION TABLE
# ==========================
eval_df = pd.DataFrame({
    "true_price": y_test.values,
    "true_segment": y_test_cls.values,
    "hard_pred_segment": pred_test_cls,
    "soft_top1_segment": soft_top1_classes,
    "global_pred": global_preds,
    "hard_pred": hard_preds,
    "soft_pred": soft_preds,
    "routing_entropy": soft_entropy,
}, index=y_test.index)

eval_df["global_abs_error"] = np.abs(eval_df["true_price"] - eval_df["global_pred"])
eval_df["hard_abs_error"] = np.abs(eval_df["true_price"] - eval_df["hard_pred"])
eval_df["soft_abs_error"] = np.abs(eval_df["true_price"] - eval_df["soft_pred"])

print(f"\n{'=' * 90}")
print("FINAL COMPARISON EXAMPLE (BEGINNING)")
print('=' * 90)
print(eval_df.head(20).to_string(float_format=lambda x: f"{x:.4f}"))


# save class probabilities for analysis
soft_proba_df = pd.DataFrame(
    soft_proba,
    columns=[f"proba_{c}" for c in soft_class_labels],
    index=X_test.index
)

print(f"\n{'=' * 90}")
print("SOFT CLASS PROBABILITIES EXAMPLE (BEGINNING)")
print('=' * 90)
print(soft_proba_df.head(10).to_string(float_format=lambda x: f"{x:.4f}"))


# ==========================
# 12) SAVE EVERYTHING
# ==========================
bundle = {
    "saved_at": datetime.now(),
    "features": FEATURES,
    "num_cols": NUM_COLS,
    "cat_cols": CAT_COLS,
    "target": TARGET,
    "bins": BINS,

    "classifier": price_classifier,
    "label_encoder": label_encoder,

    "global_regressor": global_regressor,
    "expert_models": expert_models,
    "expert_counts": expert_counts,

    "classifier_accuracy": clf_acc,
    "expert_interval_metrics": (
        expert_interval_metrics_df.to_dict(orient="index")
        if not expert_interval_metrics_df.empty else {}
    ),

    "global_overall_metrics": global_overall_metrics,
    "global_by_segment_metrics": global_by_segment_df.to_dict(orient="index"),

    "hard_overall_metrics": hard_overall_metrics,
    "hard_by_segment_metrics": hard_by_segment_df.to_dict(orient="index"),

    "soft_overall_metrics": soft_overall_metrics,
    "soft_by_segment_metrics": soft_by_segment_df.to_dict(orient="index"),

    "overall_comparison": summary_overall_df.to_dict(orient="index"),
    "per_segment_comparison": per_segment_comparison_df.to_dict(orient="index"),

    "soft_moe_config": {
        "temperature": SOFT_TEMPERATURE,
        "top_k": SOFT_TOP_K,
    },
}

joblib.dump(bundle, OUTPUT_PATH)
print(f"\nBundle saved to: {OUTPUT_PATH}")


# ==========================
# 13) PREDICTION FUNCTION
# ==========================
def predict_price_with_soft_moe(bundle_path: str, raw_df: pd.DataFrame) -> np.ndarray:
    bundle = joblib.load(bundle_path)

    classifier = bundle["classifier"]
    label_encoder = bundle.get("label_encoder")
    expert_models = bundle["expert_models"]
    global_regressor = bundle["global_regressor"]
    features = bundle["features"]

    soft_cfg = bundle.get("soft_moe_config", {})
    temperature = soft_cfg.get("temperature", 1.0)
    top_k = soft_cfg.get("top_k", None)

    raw_df = raw_df.copy()

    missing = [c for c in features if c not in raw_df.columns]
    if missing:
        raise ValueError(f"raw_df is missing columns: {missing}")

    for col in NUM_COLS:
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

    for col in CAT_COLS:
        raw_df[col] = raw_df[col].astype("string").fillna("UNKNOWN")

    raw_df["damage_secondary"] = raw_df["damage_secondary"].replace({"UNKNOWN": "NONE"})
    mask_none = raw_df["damage_secondary"] == "NONE"
    raw_df.loc[mask_none, "damage_secondary_severity"] = "NONE"
    raw_df["damage_secondary_severity"] = raw_df["damage_secondary_severity"].fillna("UNKNOWN")

    X_new = raw_df[features].copy()

    preds, _, _, _, _ = soft_moe_predict(
        X=X_new,
        classifier=classifier,
        label_encoder=label_encoder,
        expert_models=expert_models,
        global_regressor=global_regressor,
        class_temperature=temperature,
        use_top_k=top_k,
    )

    return np.array(preds)


# ==========================
# 14) OPTIONAL: HARD PREDICT FUNCTION
# ==========================
def predict_price_with_hard_routing(bundle_path: str, raw_df: pd.DataFrame) -> np.ndarray:
    bundle = joblib.load(bundle_path)

    classifier = bundle["classifier"]
    label_encoder = bundle.get("label_encoder")
    expert_models = bundle["expert_models"]
    global_regressor = bundle["global_regressor"]
    features = bundle["features"]

    raw_df = raw_df.copy()

    missing = [c for c in features if c not in raw_df.columns]
    if missing:
        raise ValueError(f"raw_df is missing columns: {missing}")

    for col in NUM_COLS:
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

    for col in CAT_COLS:
        raw_df[col] = raw_df[col].astype("string").fillna("UNKNOWN")

    raw_df["damage_secondary"] = raw_df["damage_secondary"].replace({"UNKNOWN": "NONE"})
    mask_none = raw_df["damage_secondary"] == "NONE"
    raw_df.loc[mask_none, "damage_secondary_severity"] = "NONE"
    raw_df["damage_secondary_severity"] = raw_df["damage_secondary_severity"].fillna("UNKNOWN")

    X_new = raw_df[features].copy()

    pred_cls_encoded = classifier.predict(X_new)
    if label_encoder is not None:
        pred_cls = label_encoder.inverse_transform(pred_cls_encoded)
    else:
        pred_cls = pred_cls_encoded

    preds = []
    for i in range(len(X_new)):
        row = X_new.iloc[[i]]
        cls = pred_cls[i]

        if cls in expert_models:
            pred = expert_models[cls].predict(row)[0]
        else:
            pred = global_regressor.predict(row)[0]

        preds.append(pred)

    return np.array(preds)


if __name__ == "__main__":
    print("\nScript completed successfully.")
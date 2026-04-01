import joblib
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


# ==========================
# 1) CONFIG
# ==========================
CSV_PATH = "../scraper/final_results_parsed.csv"
OUTPUT_PATH = "price_blend_bundle.pkl"

TARGET = "sold_price"

FEATURES = [
    "year",
    "make",
    "model",
    "mileage",
    "engine_volume",
    "cylinders",
    "fuel_type",
    "transmission",
    "drive_type",
    "primary_damage",
    "secondary_damage",
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

TEST_SIZE = 0.2
RANDOM_STATE = 42

MIN_PRICE = 1000
MAX_PRICE = 50000

# Веса blend
BLEND_LINEAR_WEIGHT = 0.2
BLEND_LOG_WEIGHT = 0.8

if not np.isclose(BLEND_LINEAR_WEIGHT + BLEND_LOG_WEIGHT, 1.0):
    raise ValueError("BLEND_LINEAR_WEIGHT + BLEND_LOG_WEIGHT должны давать 1.0")


# ==========================
# 2) HELPERS
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


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.dropna(subset=[TARGET])

    if "currency" in df.columns:
        cad_mask = df["currency"] == "CAD"
        df.loc[cad_mask, "sold_price"] = df.loc[cad_mask, "sold_price"] * 0.74  # курс

    df = df[df["sold_price"].notna()]
    df = df[df["sold_price"] >= 1000]

    df = df[df["year"].notna()]
    df = df[(df["year"] >= 1980) & (df["year"] <= 2026)]

    df = df[df["mileage"].notna()]
    df = df[df["mileage"] >= 0]
    df = df[df["mileage"] <= 400000]

    q = df["sold_price"].quantile(0.995)
    df["sold_price"] = df["sold_price"].clip(upper=q)

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"В датасете отсутствуют колонки: {missing}")

    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CAT_COLS:
        df[col] = df[col].astype("string").fillna("UNKNOWN")

    # df["damage_secondary"] = df["damage_secondary"].replace({"UNKNOWN": "NONE"})
    # mask_none = df["damage_secondary"] == "NONE"
    # df.loc[mask_none, "damage_secondary_severity"] = "NONE"
    # df["damage_secondary_severity"] = df["damage_secondary_severity"].fillna("UNKNOWN")

    return df


def preprocess_raw_df(raw_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    raw_df = raw_df.copy()

    missing = [c for c in features if c not in raw_df.columns]
    if missing:
        raise ValueError(f"В raw_df отсутствуют колонки: {missing}")

    for col in NUM_COLS:
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

    for col in CAT_COLS:
        raw_df[col] = raw_df[col].astype("string").fillna("UNKNOWN")

    raw_df["damage_secondary"] = raw_df["damage_secondary"].replace({"UNKNOWN": "NONE"})
    mask_none = raw_df["damage_secondary"] == "NONE"
    raw_df.loc[mask_none, "damage_secondary_severity"] = "NONE"
    raw_df["damage_secondary_severity"] = raw_df["damage_secondary_severity"].fillna("UNKNOWN")

    return raw_df[features].copy()


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
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def blend_predictions(
    pred_linear: np.ndarray,
    pred_log: np.ndarray,
    linear_weight: float,
    log_weight: float,
    min_price: float = MIN_PRICE,
    max_price: float = MAX_PRICE,
) -> np.ndarray:
    pred_final = linear_weight * pred_linear + log_weight * pred_log
    pred_final = np.clip(pred_final, min_price, max_price)
    return pred_final


def predict_with_blend(
    X_input: pd.DataFrame,
    linear_model,
    log_model,
    linear_weight: float,
    log_weight: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    pred_linear = linear_model.predict(X_input)
    pred_linear = np.clip(pred_linear, MIN_PRICE, MAX_PRICE)

    pred_log_raw = log_model.predict(X_input)
    pred_log = np.exp(pred_log_raw)
    pred_log = np.clip(pred_log, MIN_PRICE, MAX_PRICE)

    pred_final = blend_predictions(
        pred_linear=pred_linear,
        pred_log=pred_log,
        linear_weight=linear_weight,
        log_weight=log_weight,
        min_price=MIN_PRICE,
        max_price=MAX_PRICE,
    )

    debug_df = pd.DataFrame({
        "pred_linear": pred_linear,
        "pred_log_raw": pred_log_raw,
        "pred_log": pred_log,
        "pred_final": pred_final,
    }, index=X_input.index)

    return pred_final, debug_df


# ==========================
# 3) LOAD DATA
# ==========================
df = pd.read_csv(CSV_PATH)
df = prepare_dataframe(df)

print("Dataset size after filters:", len(df))
print("Price min/max:", df[TARGET].min(), df[TARGET].max())

X = df[FEATURES].copy()
y_linear = df[TARGET].astype(float).copy()
y_log = np.log(df[TARGET].astype(float))

X_train, X_test, y_train_linear, y_test_linear, y_train_log, y_test_log = train_test_split(
    X,
    y_linear,
    y_log,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

print("\nTrain size:", len(X_train))
print("Test size :", len(X_test))


# ==========================
# 4) TRAIN LINEAR MODEL
# ==========================
print("\nTraining linear_model...")
linear_model = build_regressor()
linear_model.fit(X_train, y_train_linear)

pred_train_linear = linear_model.predict(X_train)
pred_test_linear = linear_model.predict(X_test)

pred_train_linear = np.clip(pred_train_linear, MIN_PRICE, MAX_PRICE)
pred_test_linear = np.clip(pred_test_linear, MIN_PRICE, MAX_PRICE)

linear_metrics = evaluate_regression(
    y_test_linear,
    pred_test_linear,
    title="LINEAR MODEL"
)


# ==========================
# 5) TRAIN LOG MODEL
# ==========================
print("\nTraining log_model...")
log_model = build_regressor()
log_model.fit(X_train, y_train_log)

pred_train_log_raw = log_model.predict(X_train)
pred_test_log_raw = log_model.predict(X_test)

pred_train_log = np.exp(pred_train_log_raw)
pred_test_log = np.exp(pred_test_log_raw)

pred_train_log = np.clip(pred_train_log, MIN_PRICE, MAX_PRICE)
pred_test_log = np.clip(pred_test_log, MIN_PRICE, MAX_PRICE)

log_metrics = evaluate_regression(
    y_test_linear,
    pred_test_log,
    title="LOG MODEL"
)


# ==========================
# 6) BLEND MODEL
# ==========================
pred_test_blend = blend_predictions(
    pred_linear=pred_test_linear,
    pred_log=pred_test_log,
    linear_weight=BLEND_LINEAR_WEIGHT,
    log_weight=BLEND_LOG_WEIGHT,
    min_price=MIN_PRICE,
    max_price=MAX_PRICE,
)

blend_metrics = evaluate_regression(
    y_test_linear,
    pred_test_blend,
    title=f"BLEND MODEL ({BLEND_LINEAR_WEIGHT:.2f} linear + {BLEND_LOG_WEIGHT:.2f} log)"
)


# ==========================
# 7) SEGMENT ANALYSIS
# ==========================
eval_df = pd.DataFrame({
    "true_price": y_test_linear.values,
    "pred_linear": pred_test_linear,
    "pred_log": pred_test_log,
    "pred_blend": pred_test_blend,
}, index=y_test_linear.index)

eval_df["true_segment"] = pd.cut(
    eval_df["true_price"],
    bins=[100, 2000, 5000, 10000, 20001],
    right=False,
    include_lowest=True,
).astype(str)

segment_rows = []

for segment, g in eval_df.groupby("true_segment"):
    segment_rows.append({
        "true_segment": segment,
        "count": len(g),
        "linear_MAE": mean_absolute_error(g["true_price"], g["pred_linear"]),
        "linear_RMSE": np.sqrt(mean_squared_error(g["true_price"], g["pred_linear"])),
        "log_MAE": mean_absolute_error(g["true_price"], g["pred_log"]),
        "log_RMSE": np.sqrt(mean_squared_error(g["true_price"], g["pred_log"])),
        "blend_MAE": mean_absolute_error(g["true_price"], g["pred_blend"]),
        "blend_RMSE": np.sqrt(mean_squared_error(g["true_price"], g["pred_blend"])),
    })

segment_comparison_df = pd.DataFrame(segment_rows).sort_values("true_segment")

print("\nSEGMENT COMPARISON")
print(segment_comparison_df)


# ==========================
# 8) OVERALL COMPARISON
# ==========================
comparison_df = pd.DataFrame([
    {"model": "linear_model", **linear_metrics},
    {"model": "log_model", **log_metrics},
    {"model": f"blend_{BLEND_LINEAR_WEIGHT:.2f}_{BLEND_LOG_WEIGHT:.2f}", **blend_metrics},
])

print("\nOVERALL COMPARISON")
print(comparison_df.sort_values("RMSE"))


# ==========================
# 9) SAVE BUNDLE
# ==========================
bundle = {
    "saved_at": datetime.now(),
    "features": FEATURES,
    "num_cols": NUM_COLS,
    "cat_cols": CAT_COLS,
    "target": TARGET,
    "min_price": MIN_PRICE,
    "max_price": MAX_PRICE,
    "linear_weight": BLEND_LINEAR_WEIGHT,
    "log_weight": BLEND_LOG_WEIGHT,
    "linear_model": linear_model,
    "log_model": log_model,
    "overall_metrics": comparison_df.to_dict(orient="records"),
    "segment_metrics": segment_comparison_df.to_dict(orient="records"),
}

#joblib.dump(bundle, OUTPUT_PATH)
#print(f"\nSaved bundle to: {OUTPUT_PATH}")


# ==========================
# 10) PREDICT FUNCTION
# ==========================
def predict_price_with_blend_bundle(
    bundle_path: str,
    raw_df: pd.DataFrame,
) -> np.ndarray:
    bundle = joblib.load(bundle_path)

    linear_model = bundle["linear_model"]
    log_model = bundle["log_model"]
    linear_weight = bundle["linear_weight"]
    log_weight = bundle["log_weight"]
    features = bundle["features"]
    min_price = bundle["min_price"]
    max_price = bundle["max_price"]

    X_new = preprocess_raw_df(raw_df=raw_df, features=features)

    pred_linear = linear_model.predict(X_new)
    pred_linear = np.clip(pred_linear, min_price, max_price)

    pred_log_raw = log_model.predict(X_new)
    pred_log = np.exp(pred_log_raw)
    pred_log = np.clip(pred_log, min_price, max_price)

    pred_final = (
        linear_weight * pred_linear +
        log_weight * pred_log
    )
    pred_final = np.clip(pred_final, min_price, max_price)

    return pred_final
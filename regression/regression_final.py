import re
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings("ignore")


# ==========================
# 1) CONFIG
# ==========================
CSV_PATH = "./cleaned.csv"
OUTPUT_PATH = "benchmark_results.pkl"

TARGET = "sold_price"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5

MIN_PRICE = 1000
MAX_PRICE = 38000

BLEND_LINEAR_WEIGHT = 0.4
BLEND_LOG_WEIGHT = 0.6

if not np.isclose(BLEND_LINEAR_WEIGHT + BLEND_LOG_WEIGHT, 1.0):
    raise ValueError("BLEND_LINEAR_WEIGHT + BLEND_LOG_WEIGHT turi sudaryti 1.0")

CURRENT_YEAR = 2026
CAD_TO_USD = 0.73

CAT_COLS = [
    "make",
    "model",
    "fuel_type",
    "transmission",
    "drive_type",
    "primary_damage",
    "secondary_damage",
]

NUM_COLS = [
    "mileage",
    "engine_volume",
    "cylinders",
    "year",
    "primary_damage_severity",
    "secondary_damage_severity",
    "primary_severity_was_missing",
    "secondary_severity_was_missing",
    "secondary_damage_present",
    "make_model_year_price_median",
]

FEATURES = CAT_COLS + NUM_COLS


# ==========================
# 2) METRICS / HELPERS
# ==========================
def accuracy_within_pct(y_true, y_pred, pct=0.2):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan

    error = np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])
    return np.mean(error <= pct)


def evaluate_regression(y_true, y_pred, title="MODEL"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    acc_10 = accuracy_within_pct(y_true, y_pred, pct=0.10)
    acc_20 = accuracy_within_pct(y_true, y_pred, pct=0.20)
    acc_30 = accuracy_within_pct(y_true, y_pred, pct=0.30)

    print(f"\n{title}")
    print(f"MAE    : {mae:.2f}")
    print(f"RMSE   : {rmse:.2f}")
    print(f"R2     : {r2:.4f}")
    print(f"ACC_10 : {acc_10:.4f} ({acc_10 * 100:.2f}%)")
    print(f"ACC_20 : {acc_20:.4f} ({acc_20 * 100:.2f}%)")
    print(f"ACC_30 : {acc_30:.4f} ({acc_30 * 100:.2f}%)")

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "ACC_10": acc_10,
        "ACC_20": acc_20,
        "ACC_30": acc_30,
    }


def blend_predictions(
    pred_linear: np.ndarray,
    pred_log: np.ndarray,
    linear_weight: float,
    log_weight: float,
    min_price: float = MIN_PRICE,
    max_price: float = MAX_PRICE,
) -> np.ndarray:
    pred_final = linear_weight * pred_linear + log_weight * pred_log
    return np.clip(pred_final, min_price, max_price)


# ==========================
# 3) TEXT / DAMAGE HELPERS
# ==========================
def clean_text_unknown(x):
    if pd.isna(x):
        return "UNKNOWN"
    x = str(x).strip().upper()
    x = re.sub(r"\s+", " ", x)
    return x if x else "UNKNOWN"


def clean_text_none(x):
    if pd.isna(x):
        return "NONE"
    x = str(x).strip().upper()
    x = re.sub(r"\s+", " ", x)
    return x if x else "NONE"


def damage_score_fn(dmg: str) -> int:
    score_map = {
        "NONE": 0,
        "UNKNOWN": 0,
        "MINOR DENT/SCRATCHES": 1,
        "NORMAL WEAR & TEAR": 1,
        "MISSING/ALTERED VIN": 1,
        "REPLACED VIN": 1,
        "REPOSSESSION": 1,
        "CASH FOR CLUNKERS": 1,
        "HAIL": 1,
        "VANDALISM": 1,
        "DAMAGE HISTORY": 1,
        "PARTIAL REPAIR": 1,
        "REAR END": 2,
        "SIDE": 2,
        "ROOF": 2,
        "UNDERCARRIAGE": 2,
        "SUSPENSION": 2,
        "STORM DAMAGE": 2,
        "FRONT END": 3,
        "ALL OVER": 3,
        "FRONT & REAR": 3,
        "WATER/FLOOD": 4,
        "THEFT": 4,
        "STRIPPED": 4,
        "MECHANICAL": 4,
        "ELECTRICAL": 4,
        "ENGINE DAMAGE": 4,
        "TRANSMISSION DAMAGE": 4,
        "FRAME DAMAGE": 5,
        "ROLLOVER": 5,
        "BURN": 5,
        "BURN - ENGINE": 5,
        "BURN - INTERIOR": 5,
        "BIOHAZARD": 5,
    }
    return score_map.get(dmg, 2)


NON_VISUAL = {
    "BIOHAZARD",
    "DAMAGE HISTORY",
    "ELECTRICAL",
    "ENGINE DAMAGE",
    "FRAME DAMAGE",
    "MECHANICAL",
    "MISSING/ALTERED VIN",
    "NORMAL WEAR & TEAR",
    "CASH FOR CLUNKERS",
    "REPOSSESSION",
    "SUSPENSION",
    "THEFT",
    "MINOR",
    "TRANSMISSION DAMAGE",
    "UNKNOWN",
    "WATER/FLOOD",
    "REPLACED VIN",
    "UNDERCARRIAGE",
}


def fit_damage_severity_maps(train_df: pd.DataFrame) -> dict:
    result = {}

    for dmg_col, sev_col in [
        ("primary_damage", "primary_damage_severity"),
        ("secondary_damage", "secondary_damage_severity"),
    ]:
        tmp = train_df.copy()
        tmp[dmg_col] = tmp[dmg_col].astype(str).str.upper().str.strip()
        tmp[sev_col] = pd.to_numeric(tmp[sev_col], errors="coerce")

        valid = tmp[tmp[sev_col].notna() & (tmp[sev_col] > 0)].copy()

        damage_to_median = valid.groupby(dmg_col)[sev_col].median().to_dict()
        global_median = float(valid[sev_col].median()) if len(valid) else 1.0

        result[f"{sev_col}_map"] = damage_to_median
        result[f"{sev_col}_global"] = global_median

    return result


def normalize_primary_damage_severity(
    damage_col: pd.Series,
    severity_col: pd.Series,
    damage_to_median: dict,
    global_median: float,
) -> tuple[pd.Series, pd.Series]:
    damage = damage_col.fillna("UNKNOWN").astype(str).str.upper().str.strip()
    severity = pd.to_numeric(severity_col, errors="coerce")

    was_missing = (severity.isna()) | (severity <= 0)
    result = severity.copy()

    none_like = damage.isin(["NONE", "UNKNOWN"])
    result.loc[none_like & was_missing] = 0

    non_visual_mask = damage.isin(NON_VISUAL)
    result.loc[non_visual_mask & was_missing & ~none_like] = 1

    remaining = result.isna() | (result <= 0)
    result.loc[remaining] = (
        damage[remaining].map(damage_to_median).fillna(global_median).values
    )

    result = result.fillna(global_median)
    return result.astype(float), was_missing.astype(int)


def normalize_secondary_damage_severity(
    damage_col: pd.Series,
    severity_col: pd.Series,
    damage_to_median: dict,
    global_median: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    damage = damage_col.fillna("NONE").astype(str).str.upper().str.strip()
    severity = pd.to_numeric(severity_col, errors="coerce")

    was_missing = (severity.isna()) | (severity <= 0)
    result = severity.copy()

    no_secondary = damage.isin(["NONE", "NO DAMAGE", "NULL", "NAN", ""])
    secondary_present = (~no_secondary).astype(int)

    result.loc[no_secondary] = 0

    missing_with_damage = (~no_secondary) & was_missing

    non_visual_mask = damage.isin(NON_VISUAL)
    result.loc[missing_with_damage & non_visual_mask] = 1

    still_missing = (result.isna() | (result <= 0)) & (~no_secondary)
    result.loc[still_missing] = (
        damage[still_missing].map(damage_to_median).fillna(global_median).values
    )

    result.loc[no_secondary] = 0
    result = result.fillna(0)

    return result.astype(float), was_missing.astype(int), secondary_present.astype(int)


# ==========================
# 4) OOF HELPERS
# ==========================
def make_group_key(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    return frame[cols].apply(tuple, axis=1)


def make_oof_group_median(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_cols: list[str],
    target_col: str,
    n_splits: int = 5,
    random_state: int = 42,
):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof = pd.Series(index=train_df.index, dtype=float)
    global_val = float(train_df[target_col].median())

    for tr_idx, val_idx in kf.split(train_df):
        tr = train_df.iloc[tr_idx]
        val = train_df.iloc[val_idx]

        grp_map = tr.groupby(group_cols)[target_col].median().to_dict()
        val_key = make_group_key(val, group_cols)

        oof.iloc[val_idx] = val_key.map(grp_map).fillna(global_val).values

    full_map = train_df.groupby(group_cols)[target_col].median().to_dict()
    test_key = make_group_key(test_df, group_cols)
    test_feat = test_key.map(full_map).fillna(global_val)

    return oof, test_feat, full_map, global_val


# ==========================
# 5) BASE PREPROCESS
# ==========================
TEXT_COLS_UNKNOWN = [
    "make",
    "model",
    "primary_damage",
    "transmission",
    "fuel_type",
    "drive_type",
    "seller_type",
    "color",
]

TEXT_COLS_NONE = [
    "secondary_damage",
]

NUMERIC_PARSE_COLS = [
    "sold_price",
    "mileage",
    "year",
    "engine_volume",
    "cylinders",
    "primary_damage_severity",
    "secondary_damage_severity",
]


def preprocess_base_rowwise(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df = df.replace("", np.nan)
    df["mileage"] = df["mileage"].replace(1, np.nan)

    for col in TEXT_COLS_UNKNOWN:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df[col].apply(clean_text_unknown)

    for col in TEXT_COLS_NONE:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df[col].apply(clean_text_none)

    if "currency" not in df.columns:
        df["currency"] = "USD"
    else:
        df["currency"] = df["currency"].apply(clean_text_unknown)

    for col in NUMERIC_PARSE_COLS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    mask_cad = df["currency"].eq("CAD")
    df.loc[mask_cad, "sold_price"] = df.loc[mask_cad, "sold_price"] * CAD_TO_USD
    df.loc[mask_cad, "currency"] = "USD"

    return df.reset_index(drop=True)


def fit_base_preprocess_bundle(train_df_rowwise: pd.DataFrame) -> dict:
    df = train_df_rowwise.copy()

    df = df[df["sold_price"].notna()]
    df = df[df["sold_price"] >= MIN_PRICE]

    df = df[df["year"].notna()]
    df = df[(df["year"] >= 1980) & (df["year"] <= CURRENT_YEAR)]

    df = df[df["mileage"].notna()]
    df = df[df["mileage"] >= 0]
    df = df[df["mileage"] <= 400000]

    upper_q = float(df["sold_price"].quantile(0.995)) if len(df) else MAX_PRICE
    engine_volume_median = float(df["engine_volume"].median()) if df["engine_volume"].notna().any() else 2.0
    cylinders_median = float(df["cylinders"].median()) if df["cylinders"].notna().any() else 4.0

    rare_thresholds = {
        "make": 100,
        "model": 20,
        "color": 100,
        "damage_combo": 50,
    }

    tmp = df.copy()
    tmp["damage_combo"] = (
        tmp["primary_damage"].fillna("UNKNOWN") + "__" + tmp["secondary_damage"].fillna("NONE")
    )

    rare_maps = {}
    for col, min_count in rare_thresholds.items():
        vc = tmp[col].value_counts(dropna=False)
        keep_values = set(vc[vc >= min_count].index.tolist())
        rare_maps[col] = keep_values

    severity_maps_bundle = fit_damage_severity_maps(df)

    return {
        "upper_price_quantile": upper_q,
        "engine_volume_median": engine_volume_median,
        "cylinders_median": cylinders_median,
        "rare_maps": rare_maps,
        "severity_maps_bundle": severity_maps_bundle,
    }


def apply_base_preprocess(
    df_rowwise: pd.DataFrame,
    base_bundle: dict,
    is_training: bool = True,
) -> pd.DataFrame:
    df = df_rowwise.copy()

    df = df[df["year"].notna()]
    df = df[(df["year"] >= 1980) & (df["year"] <= CURRENT_YEAR)]

    df = df[df["mileage"].notna()]
    df = df[df["mileage"] >= 0]
    df = df[df["mileage"] <= 400000]

    if is_training:
        df = df[df["sold_price"].notna()]
        df = df[df["sold_price"] >= MIN_PRICE]
        df = df[df["sold_price"] <= base_bundle["upper_price_quantile"]]

    df["engine_volume"] = df["engine_volume"].fillna(base_bundle["engine_volume_median"])
    df["cylinders"] = df["cylinders"].fillna(base_bundle["cylinders_median"])

    df["primary_damage"] = df["primary_damage"].apply(clean_text_unknown)
    df["secondary_damage"] = df["secondary_damage"].apply(clean_text_none)

    df["damage_combo"] = (
        df["primary_damage"].fillna("UNKNOWN") + "__" + df["secondary_damage"].fillna("NONE")
    )

    sev_maps = base_bundle["severity_maps_bundle"]

    (
        df["primary_damage_severity"],
        df["primary_severity_was_missing"],
    ) = normalize_primary_damage_severity(
        df["primary_damage"],
        df["primary_damage_severity"],
        sev_maps["primary_damage_severity_map"],
        sev_maps["primary_damage_severity_global"],
    )

    (
        df["secondary_damage_severity"],
        df["secondary_severity_was_missing"],
        df["secondary_damage_present"],
    ) = normalize_secondary_damage_severity(
        df["secondary_damage"],
        df["secondary_damage_severity"],
        sev_maps["secondary_damage_severity_map"],
        sev_maps["secondary_damage_severity_global"],
    )

    df["primary_damage_score"] = df["primary_damage"].apply(damage_score_fn)
    df["secondary_damage_score"] = df["secondary_damage"].apply(damage_score_fn)

    df["primary_damage_weighted_score"] = df["primary_damage_score"] * df["primary_damage_severity"]
    df["secondary_damage_weighted_score"] = df["secondary_damage_score"] * df["secondary_damage_severity"]

    df["damage_score"] = df["primary_damage_weighted_score"] + df["secondary_damage_weighted_score"]
    df["max_damage_score"] = df[
        ["primary_damage_weighted_score", "secondary_damage_weighted_score"]
    ].max(axis=1)

    df["engine_volume"] = df["engine_volume"].fillna(base_bundle["engine_volume_median"])
    df["cylinders"] = df["cylinders"].fillna(base_bundle["cylinders_median"])

    df.loc[df["fuel_type"].isin(["ELECTRIC", "OTHER"]), ["engine_volume", "cylinders"]] = 0

    for col in ["make", "model", "color", "damage_combo"]:
        keep_values = base_bundle["rare_maps"][col]
        df[col] = df[col].where(df[col].isin(keep_values), "OTHER")

    for col in CAT_COLS:
        if col not in df.columns:
            df[col] = "UNKNOWN"
        df[col] = df[col].astype("string").fillna("UNKNOWN")

    numeric_cols_to_clean = set(NUM_COLS + [TARGET])
    for col in numeric_cols_to_clean:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df.reset_index(drop=True)


# ==========================
# 6) FEATURE ENGINEERING
# ==========================
def build_train_test_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = TARGET,
    n_splits: int = 5,
    random_state: int = 42,
):
    train_df = train_df.copy()
    test_df = test_df.copy()

    (
        train_df["make_model_year_price_median"],
        test_df["make_model_year_price_median"],
        make_model_year_price_median_map,
        make_model_year_price_median_global,
    ) = make_oof_group_median(
        train_df=train_df,
        test_df=test_df,
        group_cols=["make", "model", "year"],
        target_col=target_col,
        n_splits=n_splits,
        random_state=random_state,
    )

    feature_stats_bundle = {
        "make_model_year_price_median_map": make_model_year_price_median_map,
        "make_model_year_price_median_global": make_model_year_price_median_global,
    }

    for col in CAT_COLS:
        train_df[col] = train_df[col].astype("string").fillna("UNKNOWN")
        test_df[col] = test_df[col].astype("string").fillna("UNKNOWN")

    for col in NUM_COLS:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    return train_df, test_df, feature_stats_bundle


def apply_saved_feature_engineering(df: pd.DataFrame, feature_stats_bundle: dict) -> pd.DataFrame:
    df = df.copy()

    key_mmy = make_group_key(df, ["make", "model", "year"])
    df["make_model_year_price_median"] = key_mmy.map(
        feature_stats_bundle["make_model_year_price_median_map"]
    ).fillna(feature_stats_bundle["make_model_year_price_median_global"])

    for col in CAT_COLS:
        df[col] = df[col].astype("string").fillna("UNKNOWN")

    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    return df


# ==========================
# 7) PREPROCESSORS
# ==========================
def build_ohe_preprocessor() -> ColumnTransformer:
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


def prepare_catboost_inputs(X_train: pd.DataFrame, X_test: pd.DataFrame):
    X_train_cb = X_train.copy()
    X_test_cb = X_test.copy()

    for col in CAT_COLS:
        X_train_cb[col] = X_train_cb[col].astype("string").fillna("UNKNOWN").astype(str)
        X_test_cb[col] = X_test_cb[col].astype("string").fillna("UNKNOWN").astype(str)

    for col in NUM_COLS:
        median_val = pd.to_numeric(X_train_cb[col], errors="coerce").median()
        X_train_cb[col] = pd.to_numeric(X_train_cb[col], errors="coerce").fillna(median_val)
        X_test_cb[col] = pd.to_numeric(X_test_cb[col], errors="coerce").fillna(median_val)

    cat_feature_indices = [X_train_cb.columns.get_loc(col) for col in CAT_COLS]
    return X_train_cb, X_test_cb, cat_feature_indices


# ==========================
# 8) MODEL BUILDERS
# ==========================
def build_ohe_model_pipeline(model_name: str) -> Pipeline:
    model_name = model_name.lower()

    if model_name == "linear_regression":
        reg = LinearRegression()

    elif model_name == "random_forest":
        reg = RandomForestRegressor(
            n_estimators=1,
            max_depth=1,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    elif model_name == "xgboost":
        reg = XGBRegressor(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=9,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    elif model_name == "lightgbm":
        if not HAS_LGBM:
            raise ImportError("lightgbm nėra įdiegtas")
        reg = LGBMRegressor(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )

    else:
        raise ValueError(f"Unknown OHE model_name: {model_name}")

    return Pipeline([
        ("prep", build_ohe_preprocessor()),
        ("reg", reg),
    ])


def build_catboost_regressor():
    if not HAS_CATBOOST:
        raise ImportError("catboost nėra įdiegtas")

    return CatBoostRegressor(
        iterations=1,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=RANDOM_STATE,
        verbose=0,
    )


# ==========================
# 9) TRAIN / EVAL HELPERS
# ==========================
def train_and_evaluate_ohe_model(
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_linear: pd.Series,
    y_test_linear: pd.Series,
    y_train_log: pd.Series,
):
    print(f"\n{'=' * 70}")
    print(f"MODEL: {model_name}")
    print(f"{'=' * 70}")

    linear_model = build_ohe_model_pipeline(model_name)
    linear_model.fit(X_train, y_train_linear)

    pred_test_linear = linear_model.predict(X_test)
    pred_test_linear = np.clip(pred_test_linear, MIN_PRICE, MAX_PRICE)

    linear_metrics = evaluate_regression(
        y_test_linear,
        pred_test_linear,
        title=f"{model_name.upper()} | LINEAR TARGET",
    )

    log_model = build_ohe_model_pipeline(model_name)
    log_model.fit(X_train, y_train_log)

    pred_test_log_raw = log_model.predict(X_test)
    pred_test_log = np.exp(pred_test_log_raw)
    pred_test_log = np.clip(pred_test_log, MIN_PRICE, MAX_PRICE)

    log_metrics = evaluate_regression(
        y_test_linear,
        pred_test_log,
        title=f"{model_name.upper()} | LOG TARGET",
    )

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
        title=f"{model_name.upper()} | BLEND",
    )

    return {
        "model_name": model_name,
        "linear_model": linear_model,
        "log_model": log_model,
        "linear_metrics": linear_metrics,
        "log_metrics": log_metrics,
        "blend_metrics": blend_metrics,
    }


def train_and_evaluate_catboost(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_linear: pd.Series,
    y_test_linear: pd.Series,
    y_train_log: pd.Series,
):
    print(f"\n{'=' * 70}")
    print("MODEL: catboost")
    print(f"{'=' * 70}")

    X_train_cb, X_test_cb, cat_feature_indices = prepare_catboost_inputs(X_train, X_test)

    linear_model = build_catboost_regressor()
    linear_model.fit(
        X_train_cb,
        y_train_linear,
        cat_features=cat_feature_indices,
    )

    pred_test_linear = linear_model.predict(X_test_cb)
    pred_test_linear = np.clip(pred_test_linear, MIN_PRICE, MAX_PRICE)

    linear_metrics = evaluate_regression(
        y_test_linear,
        pred_test_linear,
        title="CATBOOST | LINEAR TARGET",
    )

    log_model = build_catboost_regressor()
    log_model.fit(
        X_train_cb,
        y_train_log,
        cat_features=cat_feature_indices,
    )

    pred_test_log_raw = log_model.predict(X_test_cb)
    pred_test_log = np.exp(pred_test_log_raw)
    pred_test_log = np.clip(pred_test_log, MIN_PRICE, MAX_PRICE)

    log_metrics = evaluate_regression(
        y_test_linear,
        pred_test_log,
        title="CATBOOST | LOG TARGET",
    )

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
        title="CATBOOST | BLEND",
    )

    return {
        "model_name": "catboost",
        "linear_model": linear_model,
        "log_model": log_model,
        "linear_metrics": linear_metrics,
        "log_metrics": log_metrics,
        "blend_metrics": blend_metrics,
        "cat_feature_indices": cat_feature_indices,
    }


# ==========================
# 10) FEATURE IMPORTANCE
# ==========================
def get_grouped_feature_importance_df(
    model_pipeline: Pipeline,
    cat_cols: list[str],
) -> pd.DataFrame | None:
    prep = model_pipeline.named_steps["prep"]
    reg = model_pipeline.named_steps["reg"]

    if not hasattr(reg, "feature_importances_"):
        return None

    feature_names = prep.get_feature_names_out()
    importances = reg.feature_importances_

    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })

    def restore_base_feature(name: str) -> str:
        if name.startswith("num__"):
            return name.replace("num__", "")

        if name.startswith("cat__"):
            raw = name.replace("cat__", "")
            for col in cat_cols:
                if raw == col or raw.startswith(col + "_"):
                    return col
            return raw

        return name

    fi_df["base_feature"] = fi_df["feature"].apply(restore_base_feature)

    grouped_fi = (
        fi_df.groupby("base_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False, ignore_index=True)
    )

    return grouped_fi


def get_catboost_feature_importance_df(model, feature_names: list[str]) -> pd.DataFrame | None:
    if not hasattr(model, "get_feature_importance"):
        return None

    importances = model.get_feature_importance()
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False, ignore_index=True)

    return fi_df


# ==========================
# 11) INFERENCE HELPERS
# ==========================
def preprocess_raw_df_for_bundle(raw_df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    df = preprocess_base_rowwise(raw_df)
    df = apply_base_preprocess(df, bundle["base_preprocess_bundle"], is_training=False)
    df = apply_saved_feature_engineering(df, bundle["feature_stats_bundle"])

    missing = [c for c in bundle["features"] if c not in df.columns]
    if missing:
        raise ValueError(f"Po feature engineering trūksta stulpelių: {missing}")

    return df[bundle["features"]].copy()


# ==========================
# 12) MAIN BENCHMARK
# ==========================
if __name__ == "__main__":
    df_raw = pd.read_csv(CSV_PATH)

    raw_train_df, raw_test_df = train_test_split(
        df_raw,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    raw_train_df = raw_train_df.reset_index(drop=True)
    raw_test_df = raw_test_df.reset_index(drop=True)

    train_rowwise = preprocess_base_rowwise(raw_train_df)
    test_rowwise = preprocess_base_rowwise(raw_test_df)

    base_preprocess_bundle = fit_base_preprocess_bundle(train_rowwise)

    train_df = apply_base_preprocess(train_rowwise, base_preprocess_bundle, is_training=True)
    test_df = apply_base_preprocess(test_rowwise, base_preprocess_bundle, is_training=True)

    print("Train size after base preprocess:", len(train_df))
    print("Test size after base preprocess :", len(test_df))
    print("Train price min/max:", train_df[TARGET].min(), train_df[TARGET].max())
    print("Mean test price:", test_df[TARGET].mean())

    train_fe, test_fe, feature_stats_bundle = build_train_test_features(
        train_df=train_df,
        test_df=test_df,
        target_col=TARGET,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
    )

    missing_train = [c for c in FEATURES if c not in train_fe.columns]
    missing_test = [c for c in FEATURES if c not in test_fe.columns]
    if missing_train:
        raise ValueError(f"train_fe trūksta stulpelių: {missing_train}")
    if missing_test:
        raise ValueError(f"test_fe trūksta stulpelių: {missing_test}")

    X_train = train_fe[FEATURES].copy()
    X_test = test_fe[FEATURES].copy()

    y_train_linear = train_fe[TARGET].astype(float).copy()
    y_test_linear = test_fe[TARGET].astype(float).copy()

    y_train_log = np.log(train_fe[TARGET].astype(float))

    model_names = [
        "linear_regression",
        "random_forest",
        "xgboost",
    ]

    if HAS_LGBM:
        model_names.append("lightgbm")
    else:
        print("\n[WARN] lightgbm nėra įdiegtas, praleidžiu.")

    benchmark_rows = []
    fitted_results = {}

    for model_name in model_names:
        result = train_and_evaluate_ohe_model(
            model_name=model_name,
            X_train=X_train,
            X_test=X_test,
            y_train_linear=y_train_linear,
            y_test_linear=y_test_linear,
            y_train_log=y_train_log,
        )

        fitted_results[model_name] = result

        benchmark_rows.append({
            "model": f"{model_name}_linear",
            **result["linear_metrics"],
        })
        benchmark_rows.append({
            "model": f"{model_name}_log",
            **result["log_metrics"],
        })
        benchmark_rows.append({
            "model": f"{model_name}_blend",
            **result["blend_metrics"],
        })

    if HAS_CATBOOST:
        cb_result = train_and_evaluate_catboost(
            X_train=X_train,
            X_test=X_test,
            y_train_linear=y_train_linear,
            y_test_linear=y_test_linear,
            y_train_log=y_train_log,
        )

        fitted_results["catboost"] = cb_result

        benchmark_rows.append({
            "model": "catboost_linear",
            **cb_result["linear_metrics"],
        })
        benchmark_rows.append({
            "model": "catboost_log",
            **cb_result["log_metrics"],
        })
        benchmark_rows.append({
            "model": "catboost_blend",
            **cb_result["blend_metrics"],
        })
    else:
        print("[WARN] catboost nėra įdiegtas, praleidžiu.")

    comparison_df = pd.DataFrame(benchmark_rows)

    print("\nOVERALL COMPARISON")
    print(comparison_df.sort_values("MAE").to_string(index=False))

    best_row = comparison_df.sort_values("MAE", ascending=True).iloc[0]
    print("\nBEST MODEL:")
    print(best_row.to_string())

    for model_name, result in fitted_results.items():
        if model_name == "catboost":
            fi_linear = get_catboost_feature_importance_df(result["linear_model"], FEATURES)
            fi_log = get_catboost_feature_importance_df(result["log_model"], FEATURES)

            if fi_linear is not None:
                print("\nCATBOOST FEATURE IMPORTANCE | LINEAR")
                print(fi_linear.head(20).to_string(index=False))

            if fi_log is not None:
                print("\nCATBOOST FEATURE IMPORTANCE | LOG")
                print(fi_log.head(20).to_string(index=False))

        else:
            fi_linear = get_grouped_feature_importance_df(result["linear_model"], CAT_COLS)
            fi_log = get_grouped_feature_importance_df(result["log_model"], CAT_COLS)

            if fi_linear is not None:
                print(f"\nGROUPED FEATURE IMPORTANCE | {model_name.upper()} | LINEAR")
                print(fi_linear.head(20).to_string(index=False))

            if fi_log is not None:
                print(f"\nGROUPED FEATURE IMPORTANCE | {model_name.upper()} | LOG")
                print(fi_log.head(20).to_string(index=False))

    bundle = {
        "features": FEATURES,
        "num_cols": NUM_COLS,
        "cat_cols": CAT_COLS,
        "target": TARGET,
        "min_price": MIN_PRICE,
        "max_price": MAX_PRICE,
        "linear_weight": BLEND_LINEAR_WEIGHT,
        "log_weight": BLEND_LOG_WEIGHT,
        "base_preprocess_bundle": base_preprocess_bundle,
        "feature_stats_bundle": feature_stats_bundle,
        "comparison_df": comparison_df.to_dict(orient="records"),
        "best_model_row": best_row.to_dict(),
    }

    # joblib.dump(bundle, OUTPUT_PATH)
    # print(f"\nSaved benchmark bundle to: {OUTPUT_PATH}")
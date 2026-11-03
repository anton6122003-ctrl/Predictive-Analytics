import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# NASA C-MAPSS FD001 - FINAL RUL MODEL TRAINING
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

TRAIN_PATH = DATA_DIR / "train_FD001.txt"
TEST_PATH = DATA_DIR / "test_FD001.txt"
RUL_PATH = DATA_DIR / "RUL_FD001.txt"

MODEL_PATH = MODEL_DIR / "rul_model.pkl"
SENSORS_PATH = MODEL_DIR / "selected_sensors.pkl"

SELECTED_SENSORS = [
    "sensor_11",
    "sensor_4",
    "sensor_12",
    "sensor_7",
    "sensor_15",
    "sensor_21",
    "sensor_20",
    "sensor_2",
    "sensor_17",
    "sensor_3",
    "sensor_8",
    "sensor_13",
]

COLUMNS = (
    ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_fd001(path):
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
        engine="python",
    )


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


print("=" * 68)
print("NASA C-MAPSS FD001 - FINAL RUL MODEL TRAINING")
print("=" * 68)

for path in (TRAIN_PATH, TEST_PATH, RUL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------
print("\n[1/7] Loading FD001 data...")

train_df = load_fd001(TRAIN_PATH)
test_df = load_fd001(TEST_PATH)
test_rul = pd.read_csv(RUL_PATH, sep=r"\s+", header=None, names=["RUL"])

print(f"Training rows : {len(train_df):,}")
print(f"Training units: {train_df['unit'].nunique()}")
print(f"Test rows     : {len(test_df):,}")
print(f"Test units    : {test_df['unit'].nunique()}")

# ------------------------------------------------------------
# 2. Create training RUL target
# ------------------------------------------------------------
print("\n[2/7] Creating training RUL target...")

max_cycle = train_df.groupby("unit")["cycle"].transform("max")
train_df["RUL"] = (max_cycle - train_df["cycle"]).clip(upper=125)

print(f"RUL range: {train_df['RUL'].min():.0f} to {train_df['RUL'].max():.0f} cycles")

# ------------------------------------------------------------
# 3. Prepare features
# ------------------------------------------------------------
print("\n[3/7] Preparing the 12 selected sensor features...")

missing = [s for s in SELECTED_SENSORS if s not in train_df.columns]
if missing:
    raise ValueError(f"Missing selected sensors: {missing}")

train_df = train_df.dropna(subset=SELECTED_SENSORS + ["RUL"]).copy()

X_all = train_df[SELECTED_SENSORS]
y_all = train_df["RUL"]

print(f"Features: {', '.join(SELECTED_SENSORS)}")
print(f"Usable training rows: {len(X_all):,}")

# ------------------------------------------------------------
# 4. Engine-level validation
# ------------------------------------------------------------
print("\n[4/7] Running engine-level validation...")

# Split by engine, not by individual rows, to reduce leakage.
units = np.array(sorted(train_df["unit"].unique()))
rng = np.random.RandomState(42)
rng.shuffle(units)

split_at = int(len(units) * 0.80)
train_units = set(units[:split_at])
valid_units = set(units[split_at:])

train_part = train_df[train_df["unit"].isin(train_units)]
valid_part = train_df[train_df["unit"].isin(valid_units)]

validation_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=18,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

validation_model.fit(
    train_part[SELECTED_SENSORS],
    train_part["RUL"],
)

valid_pred = validation_model.predict(valid_part[SELECTED_SENSORS])

val_mae = mean_absolute_error(valid_part["RUL"], valid_pred)
val_rmse = rmse(valid_part["RUL"], valid_pred)
val_r2 = r2_score(valid_part["RUL"], valid_pred)

print(f"Validation engines: {len(valid_units)}")
print(f"Validation MAE : {val_mae:.2f} cycles")
print(f"Validation RMSE: {val_rmse:.2f} cycles")
print(f"Validation R²  : {val_r2:.4f}")

# ------------------------------------------------------------
# 5. Train final model on all training engines
# ------------------------------------------------------------
print("\n[5/7] Training final Random Forest on all training data...")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=18,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_all, y_all)

print("Final model training completed.")

# ------------------------------------------------------------
# 6. Evaluate on the official FD001 test set
# ------------------------------------------------------------
print("\n[6/7] Evaluating on official FD001 test engines...")

# For each test engine, the last observed cycle is the prediction point.
last_test_rows = (
    test_df.sort_values(["unit", "cycle"])
    .groupby("unit", as_index=False)
    .tail(1)
    .sort_values("unit")
)

if len(last_test_rows) != len(test_rul):
    raise ValueError(
        "Number of test engines does not match the number of RUL targets: "
        f"{len(last_test_rows)} vs {len(test_rul)}"
    )

test_pred = model.predict(last_test_rows[SELECTED_SENSORS])
test_true = test_rul["RUL"].to_numpy(dtype=float)

test_mae = mean_absolute_error(test_true, test_pred)
test_rmse = rmse(test_true, test_pred)
test_r2 = r2_score(test_true, test_pred)

print(f"Official test MAE : {test_mae:.2f} cycles")
print(f"Official test RMSE: {test_rmse:.2f} cycles")
print(f"Official test R²  : {test_r2:.4f}")

# ------------------------------------------------------------
# 7. Save model and metadata
# ------------------------------------------------------------
print("\n[7/7] Saving final model files...")

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open(SENSORS_PATH, "wb") as f:
    pickle.dump(SELECTED_SENSORS, f)

print(f"Saved model   : {MODEL_PATH}")
print(f"Saved sensors : {SENSORS_PATH}")

print("\n" + "=" * 68)
print("TRAINING COMPLETE")
print("=" * 68)
print("The dashboard can now use models/rul_model.pkl.")

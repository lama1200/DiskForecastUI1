import os, random, json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from neuralprophet import NeuralProphet, load
import pickle

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = ""      # CPU فقط
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTORCH_LIGHTNING_DISABLE_PROGRESS_BAR"] = "1"

random.seed(SEED); np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
except Exception:
    pass

TRAIN_START = pd.Timestamp("2024-05-01")
TRAIN_END   = pd.Timestamp("2025-04-30")
TEST_START  = pd.Timestamp("2025-05-01")
TEST_END    = pd.Timestamp("2025-07-31")
ANCHOR_END  = TEST_END
FORECAST_TO = pd.Timestamp("2025-12-31")
EPS = 1e-9
EPOCHS_FULL = 250

def inv_log1p_safe(a): return np.expm1(np.clip(a, -50, 50))
def rmse_safe(y_true, y_pred): return np.sqrt(mean_squared_error(y_true, y_pred))

def prepare_series(df_monthly, label):
    df = df_monthly.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    y_mb = inv_log1p_safe(df["y"].to_numpy())
    if label.upper().startswith("DATA"):
        y_mb = np.maximum.accumulate(y_mb)
    elif label.upper().startswith("LOG"):
        y_mb = pd.Series(y_mb).rolling(3, min_periods=1).median().to_numpy()
    df["y"] = np.log1p(np.maximum(y_mb, 0.0))
    return df[["ds", "y"]]

def _normalize_optimizer(name):
    name = name.strip().lower()
    return {"adam": "AdamW", "adamw": "AdamW", "sgd": "SGD"}.get(name, "AdamW")

def build_model(nc, tr, yearly, lr, batch_size, optimizer):
    return NeuralProphet(
        n_changepoints=nc,
        changepoints_range=0.95,
        yearly_seasonality=yearly,
        weekly_seasonality=False,
        daily_seasonality=False,
        loss_func="Huber",
        trend_reg=tr,
        learning_rate=lr,
        batch_size=batch_size,
        optimizer=_normalize_optimizer(optimizer),
        trainer_config={
            "accelerator": "cpu",
            "enable_checkpointing": False,
            "logger": False,
            "enable_model_summary": False,
            "deterministic": True,
            "enable_progress_bar": True,
        },
    )

def time_range_mask(df, start, end): return (df["ds"] >= start) & (df["ds"] <= end)

# ------------------ تدريب + تنبؤ (قديم) ------------------
def train_and_forecast(path_csv, path_json, label, output_dir=".", months=6):
    df = pd.read_csv(path_csv)
    df["ds"] = pd.to_datetime(df["ds"])
    with open(path_json, "r", encoding="utf-8") as f:
        best = json.load(f)

    df = prepare_series(df, label)
    train_df = df[time_range_mask(df, TRAIN_START, TRAIN_END)].copy()
    test_df  = df[time_range_mask(df, TEST_START, TEST_END)].copy()

    tr2, val = (train_df.iloc[:-1], train_df.iloc[-1:]) if len(train_df) >= 2 else (train_df.copy(), None)
    m = build_model(
        int(best["n_changepoints"]), float(best["trend_reg"]), bool(best["yearly"]),
        float(best["learning_rate"]), len(tr2), best.get("optimizer", "AdamW")
    )
    m.fit(tr2, freq="ME", epochs=EPOCHS_FULL, progress="none", validation_df=val)

    # تقييم الاختبار
    fcst = m.predict(test_df)
    y_true = inv_log1p_safe(test_df["y"].to_numpy())
    y_pred = inv_log1p_safe(fcst["yhat1"].to_numpy())

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = rmse_safe(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, EPS))) * 100.0)
    wmape = float(100.0 * np.sum(np.abs(y_true - y_pred)) / max(EPS, np.sum(np.abs(y_true))))
    acc = 100.0 - wmape

    test_csv = os.path.join(output_dir, f"test_forecast_{label}.csv")
    pd.DataFrame({"ds": test_df["ds"], "Actual_MB": y_true, "Pred_MB": y_pred}).to_csv(test_csv, index=False)

    # --- التنبؤ المستقبل (عدد أشهر ديناميكي) ---
    fut_df = m.make_future_dataframe(df[df["ds"] <= ANCHOR_END], periods=months, n_historic_predictions=False)
    fut = m.predict(fut_df)
    fut = fut[fut["ds"] > ANCHOR_END][["ds", "yhat1"]].copy()
    fut["SizeInMB"] = inv_log1p_safe(fut["yhat1"].to_numpy())

    future_csv = os.path.join(output_dir, f"future_{label}_{months}M.csv")
    fut.to_csv(future_csv, index=False)

    return {
        "test_csv": test_csv,
        "future_csv": future_csv,
        "metrics": {"MAE": mae, "RMSE": rmse, "MAPE": mape, "ACC": acc}
    }

def run_full_pipeline(data_csv, log_csv, params_data, params_log, out_dir=".", months=6):
    out1 = train_and_forecast(data_csv, params_data, "DATA", output_dir=out_dir, months=months)
    out2 = train_and_forecast(log_csv,  params_log,  "LOG",  output_dir=out_dir, months=months)
    return {
        "DATA": out1,
        "LOG":  out2,
        "paths": [
            out1["future_csv"],
            out2["future_csv"],
            out1["test_csv"],
            out2["test_csv"],
        ],
    }

# ------------------ تدريب وحفظ (جديد) ------------------

def train_and_save_models(data_csv, log_csv, params_data, params_log, out_dir="."):
    """يدرب المودلات مرة وحدة ويحفظها باستخدام pickle (model_DATA.pkl, model_LOG.pkl)."""
    results = {}
    for label, path_csv, path_json in [
        ("DATA", data_csv, params_data),
        ("LOG", log_csv, params_log),
    ]:
        df = pd.read_csv(path_csv)
        df["ds"] = pd.to_datetime(df["ds"])
        with open(path_json, "r", encoding="utf-8") as f:
            best = json.load(f)

        df = prepare_series(df, label)
        train_df = df[time_range_mask(df, TRAIN_START, TRAIN_END)].copy()

        m = build_model(
            int(best["n_changepoints"]),
            float(best["trend_reg"]),
            bool(best["yearly"]),
            float(best["learning_rate"]),
            len(train_df),
            best.get("optimizer", "AdamW")
        )
        m.fit(train_df, freq="ME", epochs=EPOCHS_FULL, progress="none")

        # حفظ المودل باستخدام pickle
        model_path = os.path.join(out_dir, f"model_{label}.np")
        with open(model_path, "wb") as f:
         pickle.dump(m, f)


        results[label] = {"model_path": model_path}
    return results


# ------------------ تنبؤ فقط (جديد) ------------------
def forecast_only(model_path, df_csv, label, months, out_dir="."):
    """يحمّل المودل المحفوظ (pickle) ويعمل تنبؤ لعدد أشهر محدد."""
    import pickle
    with open(model_path, "rb") as f:
        m = pickle.load(f)

    df = pd.read_csv(df_csv)
    df["ds"] = pd.to_datetime(df["ds"])
    df = prepare_series(df, label)

    fut_df = m.make_future_dataframe(df, periods=months, n_historic_predictions=False)
    fut = m.predict(fut_df)
    fut = fut[fut["ds"] > ANCHOR_END][["ds", "yhat1"]].copy()
    fut["SizeInMB"] = inv_log1p_safe(fut["yhat1"].to_numpy())

    future_csv = os.path.join(out_dir, f"future_{label}_{months}M.csv")
    fut.to_csv(future_csv, index=False)
    return future_csv



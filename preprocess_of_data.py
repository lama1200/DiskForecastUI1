import pandas as pd
import numpy as np
import os

def monthly_eom(df_part):
    """تجميع يومي ثم لقطة نهاية كل شهر (EOM) مع ffill للأيام المفقودة."""
    if df_part.empty:
        return pd.DataFrame(columns=["ds", "y"])

    daily = (
        df_part.assign(DateRecorded=df_part["DateRecorded"].dt.normalize())
               .groupby("DateRecorded")["SizeInMB"].sum()
               .sort_index()
    )
    idx = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq="D")
    daily_full = daily.reindex(idx).ffill()
    monthly = daily_full.resample("ME").last()
    out = pd.DataFrame({"ds": monthly.index, "y": np.log1p(monthly.values)})
    return out

def run_preprocess(excel_path: str, server_name: str, out_dir: str = "."):
    """يشغّل التحضير على نفس ملف الإكسل والاسم المحدد."""
    df = pd.read_excel(excel_path)

    # اختيار السيرفر
    df = df[df["ServerName"] == server_name].copy()
    df["DateRecorded"] = pd.to_datetime(df["DateRecorded"])
    df["SizeInMB"] = pd.to_numeric(df["SizeInMB"], errors="coerce")
    df = df.dropna(subset=["DateRecorded", "SizeInMB"])

    is_data = df["FileType"].str.contains("data", case=False, na=False)
    is_log  = df["FileType"].str.contains("log",  case=False, na=False)
    df_data = df[is_data].copy()
    df_log  = df[is_log].copy()

    data_monthly = monthly_eom(df_data)
    log_monthly  = monthly_eom(df_log)

    data_csv = os.path.join(out_dir, "data_preprocessed.csv")
    log_csv  = os.path.join(out_dir, "log_preprocessed.csv")
    data_monthly.to_csv(data_csv, index=False)
    log_monthly.to_csv(log_csv, index=False)

    return data_csv, log_csv

import os, json, re, hashlib, uuid
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, send_from_directory, flash, redirect, url_for, session
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np

from preprocess_of_data import run_preprocess
# استيراد الدوال الجديدة من model.py (تدريب + حفظ) و (تنبؤ فقط)
from model import run_full_pipeline, forecast_only, train_and_save_models

ALLOWED_EXT = {"xlsx", "xls"}
BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"
UPLOADS_DIR = RUNS_DIR / "_uploads"

app = Flask(__name__)
app.secret_key = "change-me-please"
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
RUNS_DIR.mkdir(exist_ok=True, parents=True)
UPLOADS_DIR.mkdir(exist_ok=True, parents=True)

# ------------------- Helpers -------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def canonical_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"\s+", "-", name)
    name = re.sub(r"[^A-Za-z0-9\-_]+", "", name)
    return name or "server"

def server_dir_path(server_name: str) -> Path:
    return RUNS_DIR / canonical_name(server_name)

def server_dir(server_name: str) -> Path:
    d = server_dir_path(server_name)
    d.mkdir(parents=True, exist_ok=True)
    return d

def file_md5(p: Path) -> str:
    import hashlib
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

@app.after_request
def no_cache(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp
# ------------------------------------------------

# -------------------- الصفحات --------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", prepared=False, server_name=None)

@app.route("/multi", methods=["GET"])
def multi_page():
    return render_template("multi.html", server_list=None)

# -------------------- سيرفر واحد --------------------
@app.post("/prepare")
def prepare():
    if "excel_file" not in request.files:
        flash("لم يتم رفع ملف الإكسل.", "error")
        return redirect(url_for("index"))
    
    up = request.files["excel_file"]
    server_name = (request.form.get("server_name") or "").strip()
    if not server_name:
        flash("رجاءً أدخلي اسم السيرفر.", "error")
        return redirect(url_for("index"))
    if up.filename == "" or not allowed_file(up.filename):
        flash("ملف الإكسل غير صالح.", "error")
        return redirect(url_for("index"))

    work = server_dir(server_name)
    excel_path = work / "DataOfServer.xlsx"

    # 🔹 تحقق إذا الموديلات محفوظة مسبقاً
    model_data_pkl = work / "model_DATA.np"
    model_log_pkl  = work / "model_LOG.np"
    if model_data_pkl.exists() and model_log_pkl.exists():
    
     return render_template(
        "index.html",
        prepared=True,
        server_name=server_name,
        already_trained=True  
    )


    # 🔹 إذا ما فيه موديلات، نكمل تجهيز البيانات
    up.save(str(excel_path))
    if not excel_path.exists():
        flash("فشل حفظ ملف الإكسل. أعيدي المحاولة.", "error")
        return redirect(url_for("index"))

    # تنظيف مخلفات قديمة داخل مجلد السيرفر فقط (نُبقي ملفات الموديل والباراميترز إن وجدت)
    for p in work.glob("*.csv"):
        try:
            p.unlink()
        except:
            pass
    for p in work.glob("*.json"):
        if p.name not in ("best_params_DATA.json", "best_params_LOG.json"):
            try:
                p.unlink()
            except:
                pass

    meta = {"server": canonical_name(server_name), "excel_md5": file_md5(excel_path)}
    (work / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    try:
        _data_csv, _log_csv = run_preprocess(str(excel_path), server_name, out_dir=str(work))
        flash("تم تجهيز البيانات بنجاح. يمكنك بدء التدريب ثم اختيار مدة التنبؤ.", "success")
        return render_template("index.html", prepared=True, server_name=server_name)
    except Exception as e:
        flash(f"حدث خطأ أثناء التحضير: {e}", "error")
        return redirect(url_for("index"))


@app.post("/run")
def run():
    """يدرب الموديلات مرة واحدة ويحفظها فقط (بدون توليد تنبؤ)."""
    server_name = (request.form.get("server_name") or "").strip()
    if not server_name:
        flash("رجاءً أدخلي اسم السيرفر.", "error"); return redirect(url_for("index"))

    work = server_dir(server_name)
    meta_path = work / "meta.json"
    if not meta_path.exists():
        flash("يجب تحضير البيانات أولاً.", "error")
        return redirect(url_for("index"))

    data_csv = work / "data_preprocessed.csv"
    log_csv  = work / "log_preprocessed.csv"
    if not data_csv.exists() or not log_csv.exists():
        flash("يجب تحضير البيانات أولاً.", "error")
        return redirect(url_for("index"))

    try:
        # تدريب وحفظ الموديلات
        _ = train_and_save_models(
            data_csv=str(data_csv),
            log_csv=str(log_csv),
            params_data=str(BASE_DIR / "best_params_DATA.json"),
            params_log=str(BASE_DIR / "best_params_LOG.json"),
            out_dir=str(work)
        )
        flash("تم تدريب المودل وحفظه. اختاري مدة التنبؤ وشغّلي /forecast.", "success")
        return render_template("index.html", prepared=True, server_name=server_name)
    except Exception as e:
        flash(f"حدث خطأ أثناء التدريب: {e}", "error")
        return redirect(url_for("index"))

@app.post("/forecast")
def forecast():
    """يولّد ملفات التنبؤ من الموديلات المحفوظة حسب المدة المختارة."""
    server_name = (request.form.get("server_name") or "").strip()
    months = int(request.form.get("months") or 6)

    work = server_dir(server_name)
    meta_path = work / "meta.json"
    if not meta_path.exists():
        flash("يجب إعادة تحضير البيانات لهذا السيرفر أولاً.", "error")
        return redirect(url_for("index"))

    data_csv = work / "data_preprocessed.csv"
    log_csv  = work / "log_preprocessed.csv"
    if not data_csv.exists() or not log_csv.exists():
        flash("يجب تحضير البيانات أولاً.", "error")
        return redirect(url_for("index"))

    model_data_path = work / "model_DATA.np"
    model_log_path  = work / "model_LOG.np"
    if not model_data_path.exists() or not model_log_path.exists():
        flash("لا توجد موديلات محفوظة بعد. قومي بتشغيل التدريب أولاً.", "error")
        return redirect(url_for("index"))

    try:
        # تنبؤ فقط باستخدام الموديلات المحفوظة
        _ = forecast_only(str(model_data_path), str(data_csv), "DATA", months, out_dir=str(work))
        _ = forecast_only(str(model_log_path),  str(log_csv),  "LOG",  months, out_dir=str(work))
        return _render_results_page(work, server_name, months)
    except Exception as e:
        flash(f"حدث خطأ أثناء التنبؤ: {e}", "error")
        return redirect(url_for("index"))

@app.post("/view")
def view():
    """استعراض النتائج الحالية للمدة المختارة. إن لم يوجد ملف للمدة ولكن يوجد موديل محفوظ → يولّد فوراً."""
    server_name = (request.form.get("server_name") or "").strip()
    months = int(request.form.get("months") or 6)

    if not server_name:
        flash("رجاءً أدخلي اسم السيرفر لاستعراض النتائج.", "error")
        return redirect(url_for("index"))

    work = server_dir_path(server_name)
    future_data_csv = work / f"future_DATA_{months}M.csv"

    # إن لم يوجد ملف التنبؤ لكن يوجد موديل محفوظ → نولّد سريعاً
    if not future_data_csv.exists():
        model_data_path = work / "model_DATA.np"
        model_log_path  = work / "model_LOG.np"
        data_csv = work / "data_preprocessed.csv"
        log_csv  = work / "log_preprocessed.csv"
        if model_data_path.exists() and model_log_path.exists() and data_csv.exists() and log_csv.exists():
            try:
                _ = forecast_only(str(model_data_path), str(data_csv), "DATA", months, out_dir=str(work))
                _ = forecast_only(str(model_log_path),  str(log_csv),  "LOG",  months, out_dir=str(work))
            except Exception as e:
                flash(f"لا توجد نتائج محفوظة لهذه المدة ولم نتمكن من توليدها: {e}", "error")
                return redirect(url_for("index"))
        else:
            flash("لا توجد نتائج محفوظة لهذه المدة. شغّلي التدريب ثم التنبؤ أولاً.", "error")
            return redirect(url_for("index"))

    return _render_results_page(work, server_name, months)

def _render_results_page(work: Path, server_name: str, months: int):
   
    future_data_csv = work / f"future_DATA_{months}M.csv"
    if not future_data_csv.exists():
        flash("⚠️ لا يوجد ملف تنبؤ لهذه المدة.", "error")
        return redirect(url_for("index"))

    df = pd.read_csv(future_data_csv)
    mb_col = "SizeInMB" if "SizeInMB" in df.columns else "yhat1"
    df = df[["ds", mb_col]].rename(columns={"ds": "التاريخ", mb_col: "الحجم (MB)"})

    # ---------------- معالجة وتحويل القيم الرقمية مع التعامل مع الأخطاء ----------------
    try:
        # تحويل القيم النصية إلى أرقام
        df["الحجم (MB)"] = pd.to_numeric(df["الحجم (MB)"], errors="coerce")
        df["الحجم (GB)"] = (df["الحجم (MB)"] / 1024).round(2)
    except Exception as e:
        flash(f"⚠️ حدث خطأ أثناء معالجة ملف التنبؤ: {str(e)}", "error")
        return redirect(url_for("index"))
    # -----------------------------------------------------------------------------------

    chart_labels = df["التاريخ"].tolist()
    chart_values = df["الحجم (GB)"].tolist()

    # -------- قراءة البيانات التاريخية من data_preprocessed.csv --------
    chart_labels_history, chart_values_history = [], []
    history_csv = work / "data_preprocessed.csv"
    if history_csv.exists():
        df_hist = pd.read_csv(history_csv)
        df_hist["ds"] = pd.to_datetime(df_hist["ds"])
        # نعتمد نفس منطقك الحالي قبل نقطة البدء التاريخية للتنبؤ
        df_hist = df_hist[df_hist["ds"] < pd.to_datetime("2025-08-01")]
        df_hist["y"] = np.expm1(df_hist["y"].clip(-50, 50))
        df_hist["GB"] = (df_hist["y"] / 1024).round(2)
        chart_labels_history = df_hist["ds"].dt.strftime("%Y-%m").tolist()
        chart_values_history = df_hist["GB"].tolist()
    # ------------------------------------------------------------------

    files = []
    for fname in [
        f"future_DATA_{months}M.csv",
        f"future_LOG_{months}M.csv",
        "test_forecast_DATA.csv",  # قد تكون موجودة من تجارب سابقة
        "test_forecast_LOG.csv",
    ]:
        if (work / fname).exists():
            files.append(fname)

    return render_template(
        "results.html",
        server_name=server_name,
        work_rel=work.name,
        months=months,
        table=df.to_dict(orient="records"),
        files=files,
        chart_labels=chart_labels,
        chart_values=chart_values,
        chart_labels_history=chart_labels_history,
        chart_values_history=chart_values_history
    )


# -------------------- أكثر من سيرفر --------------------
@app.post("/list_servers")
def list_servers():
    if "excel_file_multi" not in request.files:
        flash("لم يتم رفع ملف الإكسل.", "error"); return redirect(url_for("multi_page"))
    up = request.files["excel_file_multi"]
    if up.filename == "" or not allowed_file(up.filename):
        flash("ملف الإكسل غير صالح.", "error"); return redirect(url_for("multi_page"))

    token = uuid.uuid4().hex
    saved = UPLOADS_DIR / f"{token}.xlsx"
    up.save(str(saved))
    if not saved.exists():
        flash("فشل حفظ ملف الإكسل.", "error"); return redirect(url_for("multi_page"))

    session["multi_excel_path"] = str(saved)

    try:
        df = pd.read_excel(str(saved))
        cols = {c.lower(): c for c in df.columns}
        server_col = cols.get("servername") or cols.get("server_name")
        if not server_col:
            flash("لم يتم العثور على عمود ServerName في الملف.", "error")
            return redirect(url_for("multi_page"))
        names = sorted({str(x).strip() for x in df[server_col].dropna().unique() if str(x).strip()})
    except Exception as e:
        flash(f"تعذّر قراءة الملف: {e}", "error")
        return redirect(url_for("multi_page"))

    if not names:
        flash("لا يوجد أسماء سيرفرات في الملف.", "error")
        return redirect(url_for("multi_page"))

    return render_template("multi.html", server_list=names)

@app.post("/forecast_multi")
def forecast_multi():
    months = int(request.form.get("months") or 6)
    selected = request.form.getlist("servers")

    if not selected:
        flash("رجاءً اختاري سيرفرًا واحدًا على الأقل.", "error")
        return redirect(url_for("multi_page"))

    all_server_data = {}      # للتوقع فقط (يبني الجدول)
    full_label_set = set()    # لكل التواريخ (ماضي + توقع) → للشارت
    datasets = []

    green_shades = [
        "rgba(14,138,72,1)",   # أخضر غامق
        "rgba(16,185,129,1)",  # أخضر ساطع
        "rgba(52,211,153,1)",  # أخضر متوسط
        "rgba(110,231,183,1)", # أخضر فاتح
    ]

    excel_path = session.get("multi_excel_path")

    # -------- تجهيز بيانات كل سيرفر --------
    for idx, name in enumerate(selected):
        try:
            work = server_dir(name)
            data_csv = work / "data_preprocessed.csv"
            log_csv  = work / "log_preprocessed.csv"

            # ✅ إذا السيرفر غير محضر → حضر البيانات الآن
            if not data_csv.exists() or not log_csv.exists():
                if excel_path and Path(excel_path).exists():
                    try:
                        _data_csv, _log_csv = run_preprocess(str(excel_path), name, out_dir=str(work))
                        data_csv, log_csv = Path(_data_csv), Path(_log_csv)
                    except Exception as e:
                        flash(f"⚠️ السيرفر {name}: فشل تحضير البيانات ({e})", "error")
                        continue
                else:
                    flash(f"⚠️ السيرفر {name}: لا يوجد ملف إكسل صالح للتحضير.", "error")
                    continue

            future_file = work / f"future_DATA_{months}M.csv"
            model_data_path = work / "model_DATA.np"
            model_log_path  = work / "model_LOG.np"

            # ✅ إذا ما عنده ملف تنبؤ → نحاول إنشاؤه
            if not future_file.exists():
                try:
                    if model_data_path.exists() and model_log_path.exists():
                        _ = forecast_only(str(model_data_path), str(data_csv), "DATA", months, out_dir=str(work))
                        _ = forecast_only(str(model_log_path),  str(log_csv),  "LOG",  months, out_dir=str(work))
                    else:
                        run_full_pipeline(
                            data_csv=str(data_csv),
                            log_csv=str(log_csv),
                            params_data=str(BASE_DIR / "best_params_DATA.json"),
                            params_log=str(BASE_DIR / "best_params_LOG.json"),
                            out_dir=str(work),
                            months=months
                        )
                except Exception as e:
                    msg = str(e)
                    if "singular value" in msg.lower():
                        flash(f"⚠️ السيرفر {name}: البيانات ثابتة جدًا ولا يمكن تدريب النموذج.", "error")
                    else:
                        flash(f"⚠️ السيرفر {name}: فشل التدريب ({msg[:120]}...)", "error")
                    continue

            # -------- بيانات النمو السابق --------
            hist_map = {}
            hist_path = work / "data_preprocessed.csv"
            if hist_path.exists():
                df_hist = pd.read_csv(hist_path)
                df_hist["ds"] = pd.to_datetime(df_hist["ds"])
                df_hist = df_hist[df_hist["ds"] <= pd.Timestamp("2025-07-31")]
                df_hist["y"] = np.expm1(df_hist["y"].clip(-50, 50))
                df_hist["GB"] = (df_hist["y"] / 1024).round(2)
                df_hist["label"] = df_hist["ds"].dt.strftime("%Y-%m")
                hist_map = dict(zip(df_hist["label"], df_hist["GB"]))

            # -------- بيانات التوقع --------
            if not future_file.exists():
                flash(f"⚠️ السيرفر {name}: لم يتم إنشاء ملف التنبؤ.", "error")
                continue

            df_fut = pd.read_csv(future_file)
            df_fut["ds"] = pd.to_datetime(df_fut["ds"])
            df_fut["GB"] = (df_fut["SizeInMB"] / 1024).round(2)
            df_fut["label"] = df_fut["ds"].dt.strftime("%Y-%m")
            forecast_map = dict(zip(df_fut["label"], df_fut["GB"]))

            # -------- ندمج كل التواريخ للشارت --------
            all_labels = sorted(set(hist_map.keys()) | set(forecast_map.keys()))

            # خط النمو السابق (رمادي)
            datasets.append({
                "label": f"{name} (النمو السابق)",
                "data": [hist_map.get(l, None) for l in all_labels],
                "fill": False,
                "borderColor": "gray",
                "borderWidth": 2,
                "pointRadius": 3,
                "tension": 0.2
            })

            # خط التوقع (أخضر)
            color = green_shades[idx % len(green_shades)]
            datasets.append({
                "label": f"{name} (توقع)",
                "data": [forecast_map.get(l, None) for l in all_labels],
                "fill": False,
                "borderColor": color,
                "borderWidth": 3,
                "pointRadius": 5,
                "tension": 0.3
            })

            # 👇 الجدول يعتمد فقط على التوقع
            all_server_data[name] = forecast_map  
            full_label_set.update(all_labels)

        except Exception as e:
            flash(f"⚠️ السيرفر {name}: حدث خطأ غير متوقع ({str(e)[:100]}...)", "error")
            continue

    # -------- بناء الجدول (فقط التوقع بعد آخر تاريخ فعلي) --------
    if not all_server_data:
        flash("❌ لم ينجح أي سيرفر في إنشاء التنبؤ.", "error")
        return redirect(url_for("multi_page"))

    labels_all = sorted({l for f in all_server_data.values() for l in f.keys()})

    # نحدد آخر شهر فعلي من كل السيرفرات
    last_hist_dates = []
    for name in all_server_data.keys():
        hist_path = server_dir(name) / "data_preprocessed.csv"
        if hist_path.exists():
            df_hist = pd.read_csv(hist_path)
            df_hist["ds"] = pd.to_datetime(df_hist["ds"])
            if not df_hist.empty:
                last_hist = df_hist["ds"].max().strftime("%Y-%m")
                last_hist_dates.append(last_hist)

    cutoff = max(last_hist_dates) if last_hist_dates else None
    labels_forecast_only = [l for l in labels_all if not cutoff or l > cutoff]

    # نبني الجدول من التوقع فقط
    rows, totals = [], {l: 0.0 for l in labels_forecast_only}
    for name, fdata in all_server_data.items():
        row = {"اسم السيرفر": name}
        for l in labels_forecast_only:
            val = fdata.get(l)
            if val is not None:
                row[l] = val
                totals[l] += val
        rows.append(row)

    total_row = {"اسم السيرفر": "المجموع"}
    for l, val in totals.items():
        total_row[l] = round(val, 2)
    if rows:
        rows.append(total_row)

    summary_df = pd.DataFrame(rows)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_work = f"_multi_{stamp}"
    out_dir = RUNS_DIR / summary_work
    out_dir.mkdir(exist_ok=True)
    summary_file = f"summary_MULTI_{months}M.csv"
    summary_df.to_csv(out_dir / summary_file, index=False, encoding="utf-8-sig")

    flash("✅ تم إكمال التحليل للسيرفرات الممكنة. السيرفرات الفاشلة تم تجاهلها.", "success")

    return render_template(
        "multi.html",
        server_list=None,
        multi_table=summary_df.to_dict(orient="records"),
        summary_work=summary_work,
        summary_file=summary_file,
        chart_labels_multi=sorted(full_label_set),  # 👈 الشارت = ماضي + توقع
        chart_datasets_multi=datasets,
        months=months,
        labels_forecast_only=labels_forecast_only   # 👈 نرسل للجدول فقط
    )

# -------------------- التنزيل --------------------
@app.get("/download/<work>/<path:fname>")
def download(work, fname):
    d = RUNS_DIR / secure_filename(work)
    return send_from_directory(d, fname, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)

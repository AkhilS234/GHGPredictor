import os, io, re
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from textblob import TextBlob
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

app = Flask(__name__)
app.secret_key = "replace-me"

TARGET = "GHG Emissions/Car"
NUMERIC_BASE = [
    "Total Polarity",
    "Total GHG Emissions",
    "Net Income (Billions)",
    "Invested Money (Billions)",
]
ENGINEERED = ["GHG_LAG1", "TGE_SLOPE", "NI_SLOPE", "INV_SLOPE"]
X_NUM = ["Year"] + NUMERIC_BASE + ENGINEERED
X_CAT = ["Company"]
X_COLS = X_CAT + X_NUM
USE_CALIBRATION = True
EMISSION_KEYWORDS = [
    "reduce emissions", "carbon reduction", "cut emissions", "net zero",
    "decarbonize", "decarbonization", "carbon neutrality", "lower emissions",
    "clean energy", "hydrogen economy", "zero emissions", "eco-friendly",
    "sustainable", "sustainability", "climate-friendly", "green technology",
    "carbon footprint", "electric vehicle", "ev", "hydrogen vehicle",
    "renewable energy", "fuel cell", "environmental improvement"
]
_EMISSION_KWS_LOWER = [k.lower() for k in EMISSION_KEYWORDS]
_SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+")

def norm_company(s: str) -> str:
    """Normalize company names so 'Mercedes Benz' matches 'Mercedes-Benz'."""
    s = str(s).strip().lower()
    s = s.replace("&", "and")
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def read_csv_upload(file_storage):
    if not file_storage or not file_storage.filename:
        return None
    data = file_storage.read()
    return pd.read_csv(io.BytesIO(data))

def coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def last_available(sub, col, years=(2022, 2021, 2020)):
    for y in years:
        v = sub.loc[sub["Year"] == y, col].dropna()
        if not v.empty:
            return float(v.iloc[0])
    return np.nan

def compute_slope(x_years, x_vals):
    df = pd.DataFrame({"Year": x_years, "Val": x_vals}).dropna()
    if len(df) >= 2:
        s, _ = np.polyfit(df["Year"], df["Val"], 1)
        return float(s)
    return np.nan

def extrapolate_to_2023(sub_df, col):
    f = sub_df.loc[sub_df["Year"].isin([2020, 2021, 2022]), ["Year", col]].dropna()
    if len(f) >= 2:
        slope, intercept = np.polyfit(f["Year"], f[col], 1)
        return float(slope * 2023 + intercept)
    return last_available(sub_df, col)

def make_moving_features(hist_df):
    """Adds GHG_LAG1 and slopes up to prior year for numeric drivers."""
    df = hist_df.sort_values(["Company", "Year"]).copy()
    df["GHG_LAG1"] = df.groupby("Company")[TARGET].shift(1)

    out = []
    for comp, g in df.groupby("Company"):
        g = g.copy()
        def slope_prior(col, row_year):
            use = g[g["Year"] <= row_year - 1][["Year", col]].dropna()
            if len(use) >= 2:
                s, _ = np.polyfit(use["Year"], use[col], 1)
                return float(s)
            return np.nan

        g["TGE_SLOPE"] = g.apply(lambda r: slope_prior("Total GHG Emissions", r["Year"]), axis=1)
        g["NI_SLOPE"]  = g.apply(lambda r: slope_prior("Net Income (Billions)", r["Year"]), axis=1)
        g["INV_SLOPE"] = g.apply(lambda r: slope_prior("Invested Money (Billions)", r["Year"]), axis=1)
        out.append(g)

    return pd.concat(out, ignore_index=True)

def file_to_text(file_storage):
    raw = file_storage.read()
    file_storage.stream.seek(0)
    for enc in ("utf-8-sig", "utf-8", "utf-16", "utf-16le", "utf-16be", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")

def textblob_polarity_01_from_text(text: str) -> float:
    sentences = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return 0.5
    relevant = [s for s in sentences if any(kw in s.lower() for kw in _EMISSION_KWS_LOWER)]
    if relevant:
        scores = [TextBlob(s).sentiment.polarity for s in relevant]
        avg = float(np.mean(scores))
    else:
        avg = TextBlob(text).sentiment.polarity
    return float(np.clip((avg + 1.0) / 2.0, 0.0, 1.0))

def infer_company_from_filename(filename):
    """
    Extract company from filename; keep hyphens (so 'Mercedes-Benz' stays).
    Examples:
      'Ford_2023_Report.txt' -> 'Ford'
      'Mercedes-Benz.txt'    -> 'Mercedes-Benz'
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    name = re.sub(r"[_\-\s]?(19|20)\d{2}.*$", "", name, flags=re.IGNORECASE)  
    name = re.sub(r"[_]+", " ", name).strip()  
    return name

def polarity_df_from_txt_files(files):
    rows = []
    for f in files:
        if not f or not f.filename.lower().endswith(".txt"):
            continue
        company = infer_company_from_filename(f.filename)
        if not company:
            continue
        text = file_to_text(f)
        score01 = textblob_polarity_01_from_text(text)
        rows.append({
            "Company": company,
            "CompanyKey": norm_company(company),
            "Total Polarity": score01
        })
    return pd.DataFrame(rows)

def build_2023_row(company_hist, polarity_2023):
    tge_2023 = extrapolate_to_2023(company_hist, "Total GHG Emissions")
    ni_2023  = extrapolate_to_2023(company_hist, "Net Income (Billions)")
    inv_2023 = extrapolate_to_2023(company_hist, "Invested Money (Billions)")

    g = company_hist
    g_2022 = g[g["Year"] <= 2022]
    tge_slope = compute_slope(g_2022["Year"], g_2022["Total GHG Emissions"])
    ni_slope  = compute_slope(g_2022["Year"], g_2022["Net Income (Billions)"])
    inv_slope = compute_slope(g_2022["Year"], g_2022["Invested Money (Billions)"])

    ghg_lag1 = last_available(company_hist, TARGET, years=(2022, 2021, 2020))

    return {
        "Year": 2023,
        "Total Polarity": polarity_2023,
        "Total GHG Emissions": tge_2023,
        "Net Income (Billions)": ni_2023,
        "Invested Money (Billions)": inv_2023,
        "TGE_SLOPE": tge_slope if not np.isnan(tge_slope) else 0.0,
        "NI_SLOPE":  ni_slope  if not np.isnan(ni_slope)  else 0.0,
        "INV_SLOPE": inv_slope if not np.isnan(inv_slope) else 0.0,
        "GHG_LAG1": ghg_lag1
    }

def train_and_predict(historical_df, polarity_df):
    
    for d in (historical_df,):
        d.columns = d.columns.str.strip()
        d["Company"] = d["Company"].astype(str).str.strip()
        d["CompanyKey"] = d["Company"].apply(norm_company)

    polarity_df = polarity_df.copy()
    historical_df = coerce_numeric(historical_df, ["Year"] + NUMERIC_BASE + [TARGET])
    polarity_df   = coerce_numeric(polarity_df, ["Total Polarity"])

    historical_df = make_moving_features(historical_df)

    df_fit = historical_df[historical_df["Year"].isin([2020, 2021, 2022])].copy()
    need_cols = ["Year", "Total Polarity", "Total GHG Emissions", TARGET, "GHG_LAG1",
                 "TGE_SLOPE", "NI_SLOPE", "INV_SLOPE"]
    df_fit = df_fit.dropna(subset=need_cols)
    if df_fit.empty:
        return pd.DataFrame(columns=["Company", "Year", "Predicted GHG Emissions/Car"])

    pre = ColumnTransformer([
        ("num", StandardScaler(), X_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), X_CAT),
    ])
    gbr = GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05,
        max_depth=3, subsample=0.9, random_state=0
    )
    model = Pipeline([("pre", pre), ("reg", gbr)])
    model.fit(df_fit[X_COLS], df_fit[TARGET])

    hist_keys = set(historical_df["CompanyKey"])
    pol_keys  = set(polarity_df["CompanyKey"])
    keys_to_predict = sorted(hist_keys & pol_keys)

    preds = []
    for key in keys_to_predict:
        canon_name = historical_df.loc[historical_df["CompanyKey"] == key, "Company"].iloc[0]
        sub_all = historical_df[historical_df["CompanyKey"] == key]
        pol_row = polarity_df[polarity_df["CompanyKey"] == key]
        if pol_row.empty or pd.isna(pol_row["Total Polarity"]).all():
            continue
        pol_2023 = float(pol_row["Total Polarity"].iloc[0])

        x2023 = build_2023_row(sub_all, pol_2023)
        if np.isnan(x2023.get("GHG_LAG1", np.nan)) or np.isnan(x2023.get("Total GHG Emissions", np.nan)):
            continue

        row_df = pd.DataFrame([{**x2023, "Company": canon_name}])[X_COLS]
        yhat = float(model.predict(row_df)[0])
        preds.append({"Company": canon_name, "Year": 2023, "Predicted GHG Emissions/Car": yhat})

    if not preds:
        return pd.DataFrame(columns=["Company", "Year", "Predicted GHG Emissions/Car"])

    pred_df = pd.DataFrame(preds).sort_values("Company").reset_index(drop=True)
    return pred_df

def merge_preds_with_actuals(actual_df, pred_df):
    act = actual_df.copy()
    act.columns = act.columns.str.strip()
    act["Company"] = act["Company"].astype(str).str.strip()
    for c in ["Year", TARGET]:
        if c in act.columns:
            act[c] = pd.to_numeric(act[c], errors="coerce")
    act["CompanyKey"] = act["Company"].apply(norm_company)

    pred = pred_df.copy()
    pred["CompanyKey"] = pred["Company"].apply(norm_company)

    act_2023 = (
        act.loc[act["Year"] == 2023, ["CompanyKey", TARGET]]
           .dropna()
           .groupby("CompanyKey", as_index=False)[TARGET].mean()
           .rename(columns={TARGET: "Actual GHG Emissions/Car"})
    )

    merged = (pred.merge(act_2023, on="CompanyKey", how="left")
                  .drop(columns=["CompanyKey"])
                  .sort_values("Company")
                  .reset_index(drop=True))
    return merged

def calibrate_predictions_inplace(merged_df):
    elig = merged_df.dropna(subset=["Actual GHG Emissions/Car"])
    if len(elig) < 3:
        return None  
    A = np.vstack([np.ones(len(elig)), elig["Predicted GHG Emissions/Car"]]).T
    y = elig["Actual GHG Emissions/Car"].values
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coeffs
    merged_df["Predicted GHG Emissions/Car"] = a + b * merged_df["Predicted GHG Emissions/Car"]
    return float(a), float(b)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        hist_file   = request.files.get("historical_csv")
        actual_file = request.files.get("actual_csv")
        txt_files   = request.files.getlist("reports")

        if not hist_file or not actual_file or not txt_files or not any(f.filename for f in txt_files):
            flash("Upload Historical CSV, Actual 2023 CSV, and at least one TXT report.", "danger")
            return redirect(url_for("index"))

        hist_df = read_csv_upload(hist_file)
        act_df  = read_csv_upload(actual_file)
        pol_df  = polarity_df_from_txt_files(txt_files)

        if pol_df.empty:
            flash("Could not compute polarity from TXT files.", "danger")
            return redirect(url_for("index"))

        pred_df = train_and_predict(hist_df, pol_df)
        if pred_df.empty:
            flash("No predictions could be generated. Check missing historical values.", "warning")
            return redirect(url_for("index"))

        merged = merge_preds_with_actuals(act_df, pred_df)

  
        if USE_CALIBRATION:
            calibrate_predictions_inplace(merged)

        
        r2 = None
        eligible = merged.dropna(subset=["Actual GHG Emissions/Car"])
        if len(eligible) >= 2:
            r2 = float(r2_score(
                eligible["Actual GHG Emissions/Car"],
                eligible["Predicted GHG Emissions/Car"]
            ))

       
        for col in ["Predicted GHG Emissions/Car", "Actual GHG Emissions/Car"]:
            if col in merged.columns:
                merged[col] = merged[col].round(4)

        return render_template(
            "results.html",
            headers=merged.columns.tolist(),
            tables=merged.to_dict(orient="records"),
            r2=r2
        )

    except Exception as e:
        flash(str(e), "danger")
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
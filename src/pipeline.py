"""Fraud detection pipeline: time-based validation, causal card-history features,
CatBoost, Utility-optimal threshold, submission.csv.

Usage (from project folder):
    python src\pipeline.py                   # validate + final fit + submission
    python src\pipeline.py --validate-only
    python src\pipeline.py --isolated-test   # test history uses only test rows
    python src\pipeline.py --no-history      # ablation: base features only
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "transaction-fraud-detection"
OUT = ROOT / "outputs"

CAT_COLS = ["merchant", "category", "state", "job", "gender"]
TRAIN_END = pd.Timestamp("2025-01-01")   # fit: < 2025-01-01
TUNE_END = pd.Timestamp("2025-04-01")    # tune/threshold: 2025-01..03; holdout: 2025-04..06


# ---------------------------------------------------------------- utility
def utility(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    return int(50 * tp - 100 * fn - 5 * fp)


def best_threshold(y, p):
    """Scan every cut-off of the ranked probabilities; return (threshold, utility)."""
    y = np.asarray(y)
    order = np.argsort(-p, kind="stable")
    ys, ps = y[order], p[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    pos = y.sum()
    u = 50 * tp - 100 * (pos - tp) - 5 * fp
    u = np.concatenate([[-100 * pos], u])          # index k = flag the top-k
    k = int(np.argmax(u))
    if k == 0:
        return 1.0 + 1e-9, int(u[0])
    nxt = ps[k] if k < len(ps) else 0.0
    return float((ps[k - 1] + nxt) / 2), int(u[k])


def report(name, y, p, thr):
    pred = (p >= thr).astype(int)
    y = np.asarray(y)
    tp = int(((y == 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    print(f"{name:<22} thr={thr:.4f} TP={tp} FN={fn} FP={fp} Utility={utility(y, pred)}")


# ---------------------------------------------------------------- data
def load(isolated_test):
    tr = pd.read_csv(DATA / "fraudTrain.csv", dtype={"cc_num": str, "zip": str})
    te = pd.read_csv(DATA / "fraudTest.csv", dtype={"cc_num": str, "zip": str})
    tr["dob"] = pd.to_datetime(tr["dob"], format="%Y-%m-%d")
    te["dob"] = pd.to_datetime(te["dob"], format="%m/%d/%Y")
    tr["is_test"], te["is_test"] = 0, 1
    te["is_fraud"] = -1
    tr["row_order"] = np.arange(len(tr))
    te["row_order"] = np.arange(len(te)) + len(tr)
    df = pd.concat([tr, te], ignore_index=True)
    # Test cc_num is mangled by scientific notation, so identify cardholders by
    # name + dob + street (consistent across both files).
    df["card"] = (df["first"] + "|" + df["last"] + "|" + df["dob"].dt.strftime("%Y%m%d")
                  + "|" + df["street"])
    if isolated_test:
        df.loc[df["is_test"] == 1, "card"] += "|T"
    df["ts"] = pd.to_datetime(df["unix_time"], unit="s")
    return df.sort_values(["unix_time", "row_order"], kind="stable").reset_index(drop=True)


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def build_features(df):
    f = df.copy()
    f["log_amt"] = np.log1p(f["amt"])
    f["hour"] = f["ts"].dt.hour
    f["dow"] = f["ts"].dt.dayofweek
    f["month"] = f["ts"].dt.month
    f["is_weekend"] = (f["dow"] >= 5).astype(int)
    f["age"] = (f["ts"] - f["dob"]).dt.days / 365.25
    f["dist_home_merch"] = haversine(f["lat"], f["long"], f["merch_lat"], f["merch_long"])

    n = len(f)
    t = f["unix_time"].to_numpy(dtype=np.int64)
    amt = f["amt"].to_numpy(dtype=float)
    mlat, mlon = f["merch_lat"].to_numpy(), f["merch_long"].to_numpy()
    names = ["secs_since_prev", "cnt_1h", "cnt_24h", "sum_24h", "n_prior", "mean_amt_prior",
             "amt_ratio_prior", "dist_prev_merch", "speed_kmh"]
    cols = {k: np.full(n, np.nan) for k in names}
    # Only earlier rows of the same card are used (no labels, nothing from the future).
    for idx in f.groupby("card", sort=False).indices.values():
        idx = np.sort(idx)
        tt, aa = t[idx], amt[idx]
        k = len(idx)
        pos = np.arange(k)
        cs = np.concatenate([[0.0], np.cumsum(aa)])          # cs[i] = sum(aa[:i])
        for w, key in ((3600, "cnt_1h"), (86400, "cnt_24h")):
            left = np.searchsorted(tt, tt - w, side="left")
            cols[key][idx] = pos - left
            if w == 86400:
                cols["sum_24h"][idx] = cs[pos] - cs[left]
        cols["n_prior"][idx] = pos
        mean = np.where(pos > 0, cs[pos] / np.maximum(pos, 1), np.nan)
        cols["mean_amt_prior"][idx] = mean
        cols["amt_ratio_prior"][idx] = aa / (mean + 1e-6)
        if k > 1:
            dt = np.diff(tt).astype(float)
            d = haversine(mlat[idx[:-1]], mlon[idx[:-1]], mlat[idx[1:]], mlon[idx[1:]])
            cols["secs_since_prev"][idx[1:]] = dt
            cols["dist_prev_merch"][idx[1:]] = d
            cols["speed_kmh"][idx[1:]] = d / np.maximum(dt, 60) * 3600
    for k_, v in cols.items():
        f[k_] = v
    f["new_merchant"] = (f.groupby(["card", "merchant"]).cumcount() == 0).astype(int)
    f["new_category"] = (f.groupby(["card", "category"]).cumcount() == 0).astype(int)
    # Unlabelled "regime" descriptors: trailing 7-day mix of ALL transactions.
    r = pd.DataFrame({"morning": (f["hour"] < 12).astype(float),
                      "night": f["hour"].isin([22, 23, 0, 1, 2, 3]).astype(float),
                      "lamt": f["log_amt"], "one": 1.0}).set_index(f["ts"])
    roll = r.rolling("7D", min_periods=1)
    f["reg_morning"] = roll["morning"].mean().to_numpy()
    f["reg_night"] = roll["night"].mean().to_numpy()
    f["reg_lamt"] = roll["lamt"].mean().to_numpy()
    f["reg_vol"] = roll["one"].sum().to_numpy() / 7
    f["amt_vs_reg"] = f["log_amt"] - f["reg_lamt"]
    # Trailing relative frequency of the same value among ALL transactions (no labels):
    # a value that is rare in the current window is very likely fraud.
    g_all = np.arange(n)
    for dim in REL_DIMS:
        for w, tag in ((86400, "1d"), (7 * 86400, "7d")):
            tot = g_all - np.searchsorted(t, t - w, side="left")
            cnt = np.zeros(n)
            for idx in f.groupby(dim, sort=False).indices.values():
                idx = np.sort(idx)
                tt = t[idx]
                cnt[idx] = np.arange(len(idx)) - np.searchsorted(tt, tt - w, side="left")
            f[f"rel_{dim}_{tag}"] = (cnt + 0.5) / (tot + 1)
            f[f"lcnt_{dim}_{tag}"] = np.log1p(cnt)
    return f


REL_DIMS = ["category", "merchant", "hour", "state", "job"]
NUM_REL = [f"{p}_{d}_{w}" for d in REL_DIMS for w in ("1d", "7d") for p in ("rel", "lcnt")]
NUM_REG = ["reg_morning", "reg_night", "reg_lamt", "reg_vol", "amt_vs_reg"]


NUM_BASE = ["amt", "log_amt", "hour", "dow", "month", "is_weekend", "age", "city_pop",
            "lat", "long", "merch_lat", "merch_long", "dist_home_merch"]
NUM_HIST = ["secs_since_prev", "cnt_1h", "cnt_24h", "sum_24h", "n_prior", "mean_amt_prior",
            "amt_ratio_prior", "dist_prev_merch", "speed_kmh", "new_merchant", "new_category"]


# ---------------------------------------------------------------- models
def fit_catboost(X, y, Xv=None, yv=None, iterations=2000):
    from catboost import CatBoostClassifier, Pool
    es = Xv is not None
    m = CatBoostClassifier(iterations=iterations, learning_rate=0.08, depth=6,
                           eval_metric="PRAUC", random_seed=42, verbose=200,
                           thread_count=-1, **({"od_type": "Iter", "od_wait": 100} if es else {}))
    cats = [c for c in CAT_COLS if c in X.columns]
    ev = Pool(Xv, yv, cat_features=cats) if es else None
    m.fit(Pool(X, y, cat_features=cats), eval_set=ev, use_best_model=es)
    return m


def fit_logreg(X, y):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    num = [c for c in X.columns if c not in CAT_COLS]
    ct = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50),
         ["category", "state", "gender"]),
    ])
    return Pipeline([("ct", ct), ("lr", LogisticRegression(max_iter=300))]).fit(X, y)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--isolated-test", action="store_true",
                    help="test-period history ignores train rows and earlier... uses test rows only")
    ap.add_argument("--no-history", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    df = build_features(load(args.isolated_test))
    feats = NUM_BASE + ([] if args.no_history else NUM_HIST) + CAT_COLS
    lab = df[df["is_test"] == 0]
    fit_m = lab["ts"] < TRAIN_END
    tune_m = (lab["ts"] >= TRAIN_END) & (lab["ts"] < TUNE_END)
    hold_m = lab["ts"] >= TUNE_END
    X, y = lab[feats], lab["is_fraud"].astype(int)
    print("fraud rate fit/tune/holdout:", [round(y[m].mean(), 5) for m in (fit_m, tune_m, hold_m)])
    print(lab.groupby(lab["ts"].dt.to_period("M"))["is_fraud"].agg(["sum", "count"]).T.to_string())

    lr = fit_logreg(X[fit_m], y[fit_m])
    p_t, p_h = lr.predict_proba(X[tune_m])[:, 1], lr.predict_proba(X[hold_m])[:, 1]
    thr, _ = best_threshold(y[tune_m], p_t)
    report("LogReg tune", y[tune_m], p_t, thr)
    report("LogReg holdout", y[hold_m], p_h, thr)

    cb = fit_catboost(X[fit_m], y[fit_m], X[tune_m], y[tune_m])
    best_iter = cb.get_best_iteration() + 1
    p_t, p_h = cb.predict_proba(X[tune_m])[:, 1], cb.predict_proba(X[hold_m])[:, 1]
    thr, _ = best_threshold(y[tune_m], p_t)
    print(f"\nbest_iter={best_iter}  (theoretical threshold = 0.0323)")
    report("CatBoost tune", y[tune_m], p_t, thr)
    report("CatBoost holdout", y[hold_m], p_h, thr)
    vm = lab[tune_m | hold_m]
    pv = pd.Series(np.concatenate([p_t, p_h]), index=vm.index)
    for per, g in vm.groupby(vm["ts"].dt.to_period("M")):
        report(f"  month {per}", y.loc[g.index], pv.loc[g.index].to_numpy(), thr)
    pd.Series(dict(zip(feats, cb.get_feature_importance()))).sort_values(
        ascending=False).to_csv(OUT / "feature_importance.csv", header=["importance"])

    if args.validate_only:
        return

    final = fit_catboost(X, y, iterations=int(best_iter * 1.2))
    te = df[df["is_test"] == 1]
    prob = pd.Series(final.predict_proba(te[feats])[:, 1], index=te["ID"].to_numpy())
    ids = pd.read_csv(DATA / "fraudTest.csv", usecols=["ID"])["ID"]   # original order
    prob = prob.loc[ids]
    sub = pd.DataFrame({"ID": ids.to_numpy(), "is_fraud": (prob.to_numpy() >= thr).astype(int)})
    sub.to_csv(OUT / "submission.csv", index=False)
    prob.rename("prob").to_csv(OUT / "test_probabilities.csv", index_label="ID")

    assert len(sub) == len(ids) and sub["ID"].is_unique and sub["ID"].equals(ids)
    assert sub["ID"].is_unique and set(sub["is_fraud"].unique()) <= {0, 1}
    print(f"\nsubmission rows={len(sub)} flagged={int(sub['is_fraud'].sum())} "
          f"({sub['is_fraud'].mean():.4%}) -> {OUT / 'submission.csv'}")


if __name__ == "__main__":
    main()

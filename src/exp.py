"""Experiment harness: cache features once, compare feature sets / params by pooled OOF utility."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as P  # noqa: E402

CACHE = P.OUT / "feats.pkl"
DROP = {"month", "n_prior", "lat", "long", "merch_lat", "merch_long", "city_pop", "age",
        "merchant", "job", "state", "gender"}


def extra_features(f):
    """Additional causal features (card history + trailing same-category amount norms)."""
    t = f["unix_time"].to_numpy(dtype=np.int64)
    amt = f["amt"].to_numpy(dtype=float)
    n = len(f)
    out = {k: np.full(n, np.nan) for k in
           ["card_max_prior", "amt_over_max", "nmerch_24h", "ncat_24h", "secs_same_cat",
            "amt_rank_card", "std_amt_prior", "amt_z", "night_frac_prior", "mean_amt_7d",
            "amt_ratio_7d"]}
    night = f["hour"].isin([22, 23, 0, 1, 2, 3]).to_numpy().astype(float)
    mer = f["merchant"].astype("category").cat.codes.to_numpy()
    cat = f["category"].astype("category").cat.codes.to_numpy()
    for idx in f.groupby("card", sort=False).indices.values():
        idx = np.sort(idx)
        tt, aa, nn = t[idx], amt[idx], night[idx]
        k = len(idx)
        pos = np.arange(k)
        cmax = np.maximum.accumulate(aa)
        prev_max = np.concatenate([[np.nan], cmax[:-1]])
        out["card_max_prior"][idx] = prev_max
        out["amt_over_max"][idx] = aa / (prev_max + 1e-6)
        cs = np.concatenate([[0.0], np.cumsum(aa)])
        cs2 = np.concatenate([[0.0], np.cumsum(aa ** 2)])
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = cs[pos] / pos
            var = cs2[pos] / pos - mean ** 2
        sd = np.sqrt(np.maximum(var, 0))
        out["std_amt_prior"][idx] = np.where(pos > 1, sd, np.nan)
        out["amt_z"][idx] = np.where(pos > 1, (aa - mean) / (sd + 1.0), np.nan)
        cn = np.concatenate([[0.0], np.cumsum(nn)])
        out["night_frac_prior"][idx] = np.where(pos > 0, cn[pos] / np.maximum(pos, 1), np.nan)
        l7 = np.searchsorted(tt, tt - 7 * 86400, side="left")
        m7 = np.where(pos - l7 > 0, (cs[pos] - cs[l7]) / np.maximum(pos - l7, 1), np.nan)
        out["mean_amt_7d"][idx] = m7
        out["amt_ratio_7d"][idx] = aa / (m7 + 1e-6)
        l24 = np.searchsorted(tt, tt - 86400, side="left")
        mm, cc = mer[idx], cat[idx]
        last_cat = {}
        for i in range(k):
            out["nmerch_24h"][idx[i]] = len(set(mm[l24[i]:i]))
            out["ncat_24h"][idx[i]] = len(set(cc[l24[i]:i]))
            lc = last_cat.get(cc[i])
            out["secs_same_cat"][idx[i]] = tt[i] - lc if lc is not None else np.nan
            last_cat[cc[i]] = tt[i]
        out["amt_rank_card"][idx] = [(aa[:i] < aa[i]).mean() if i else np.nan for i in range(k)]
    for k_, v in out.items():
        f[k_] = v

    # trailing 7d amount norm of the same category across ALL cards
    t7 = 7 * 86400
    ca = np.full(n, np.nan)
    cr = np.full(n, np.nan)
    for idx in f.groupby("category", sort=False).indices.values():
        idx = np.sort(idx)
        tt, aa = t[idx], amt[idx]
        cs = np.concatenate([[0.0], np.cumsum(np.log1p(aa))])
        l = np.searchsorted(tt, tt - t7, side="left")
        pos = np.arange(len(idx))
        mean = np.where(pos - l > 0, (cs[pos] - cs[l]) / np.maximum(pos - l, 1), np.nan)
        ca[idx] = mean
        cr[idx] = np.log1p(aa) - mean
    f["cat_lamt_7d"] = ca
    f["amt_vs_cat_7d"] = cr
    return f


EXTRA = ["card_max_prior", "amt_over_max", "nmerch_24h", "ncat_24h", "secs_same_cat",
         "amt_rank_card", "std_amt_prior", "amt_z", "night_frac_prior", "mean_amt_7d",
         "amt_ratio_7d", "cat_lamt_7d", "amt_vs_cat_7d"]


def get_data():
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    t0 = time.time()
    df = extra_features(P.build_features(P.load(False)))
    df.to_pickle(CACHE)
    print(f"features built in {time.time() - t0:.0f}s", flush=True)
    return df


def cv(lab, feats, params, name, folds=range(6)):
    from catboost import CatBoostClassifier, Pool
    cats = [c for c in P.CAT_COLS if c in feats]
    y = lab["is_fraud"].astype(int).to_numpy()
    mi = (lab["ts"].dt.year - 2024) * 12 + lab["ts"].dt.month - 1
    fold = (mi // 3).to_numpy()
    oof = np.full(len(lab), np.nan)
    t0 = time.time()
    for k in folds:
        va = fold == k
        m = CatBoostClassifier(random_seed=42, verbose=0, thread_count=-1,
                               allow_writing_files=False, **params)
        m.fit(Pool(lab.loc[~va, feats], y[~va], cat_features=cats))
        oof[va] = m.predict_proba(Pool(lab.loc[va, feats], cat_features=cats))[:, 1]
    ok = ~np.isnan(oof)
    thr, u = P.best_threshold(y[ok], oof[ok])
    fu = [P.utility(y[fold == k], (oof[fold == k] >= thr).astype(int)) for k in folds]
    print(f"[{name}] feats={len(feats)} thr={thr:.4f} pooledU={u} folds={fu} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return oof, thr, u


if __name__ == "__main__":
    P.OUT.mkdir(exist_ok=True)
    df = get_data()
    lab = df[df["is_test"] == 0].reset_index(drop=True)
    base = [f for f in P.NUM_BASE + P.NUM_HIST + P.CAT_COLS + P.NUM_REL if f not in DROP]
    p0 = dict(iterations=300, learning_rate=0.1, depth=6)
    which = sys.argv[1:] or ["base", "extra", "extra_big"]
    if "base" in which:
        cv(lab, base, p0, "base")
    if "extra" in which:
        cv(lab, base + EXTRA, p0, "base+extra")
    if "extra_big" in which:
        cv(lab, base + EXTRA, dict(iterations=800, learning_rate=0.05, depth=8), "extra big")
    if "final" in which:
        from catboost import CatBoostClassifier, Pool
        feats = base + EXTRA
        params = dict(iterations=800, learning_rate=0.05, depth=8)
        oof, thr, u = cv(lab, feats, params, "final-cv")
        y = lab["is_fraud"].astype(int).to_numpy()
        for t in (0.01, 0.0175, 0.025, 0.04, 0.06):
            print(f"  thr {t}: {P.utility(y, (oof >= t).astype(int))}")
        cats = [c for c in P.CAT_COLS if c in feats]
        m = CatBoostClassifier(random_seed=42, verbose=0, thread_count=-1,
                               allow_writing_files=False, **params)
        m.fit(Pool(lab[feats], y, cat_features=cats))
        te = df[df["is_test"] == 1]
        prob = pd.Series(m.predict_proba(Pool(te[feats], cat_features=cats))[:, 1],
                         index=te["ID"].to_numpy())
        ids = pd.read_csv(P.DATA / "fraudTest.csv", usecols=["ID"])["ID"]
        prob = prob.loc[ids]
        sub = pd.DataFrame({"ID": ids.to_numpy(), "is_fraud": (prob.to_numpy() >= thr).astype(int)})
        assert len(sub) == len(ids) and sub["ID"].is_unique
        sub.to_csv(P.OUT / "submission.csv", index=False)
        prob.rename("prob").to_csv(P.OUT / "test_probabilities.csv", index_label="ID")
        print(f"submission rows={len(sub)} flagged={sub['is_fraud'].mean():.4%} "
              f"OOF flagged={(oof >= thr).mean():.4%} sum_p={prob.sum():.0f}")

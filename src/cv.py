"""Month-blocked CV (6 folds x 3 months) over all labelled data.
The test period reuses regimes seen in 2024, so validating on one future block is misleading.
Usage: python src\cv.py [--iters 400] [--drop month,n_prior]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--drop", default="month,n_prior")
    ap.add_argument("--keep", default="", help="comma list; if set, use exactly these features")
    ap.add_argument("--reg", action="store_true", help="add regime features")
    ap.add_argument("--final", action="store_true", help="refit on all data, write submission")
    ap.add_argument("--rel", action="store_true", help="add relative-frequency features")
    ap.add_argument("--isolated-test", action="store_true")
    args = ap.parse_args()
    from catboost import CatBoostClassifier, Pool

    df = P.build_features(P.load(args.isolated_test))
    lab = df[df["is_test"] == 0].reset_index(drop=True)
    drop = set(filter(None, args.drop.split(",")))
    allf = P.NUM_BASE + P.NUM_HIST + P.CAT_COLS + (P.NUM_REG if args.reg else [])
    if args.rel:
        allf += P.NUM_REL
    feats = [f for f in allf if f not in drop]
    if args.keep:
        feats = args.keep.split(",")
    print("features:", feats, flush=True)
    cats = [c for c in P.CAT_COLS if c in feats]
    y = lab["is_fraud"].astype(int).to_numpy()
    mi = (lab["ts"].dt.year - 2024) * 12 + lab["ts"].dt.month - 1   # 0..17
    fold = (mi // 3).to_numpy()
    oof = np.zeros(len(lab))
    for k in range(6):
        va = fold == k
        m = CatBoostClassifier(iterations=args.iters, learning_rate=0.1, depth=6,
                               random_seed=42, verbose=0, thread_count=-1,
                               allow_writing_files=False)
        m.fit(Pool(lab.loc[~va, feats], y[~va], cat_features=cats))
        oof[va] = m.predict_proba(Pool(lab.loc[va, feats], cat_features=cats))[:, 1]
        thr_k, u_k = P.best_threshold(y[va], oof[va])
        print(f"fold {k} own-best thr={thr_k:.4f} utility={u_k}", flush=True)
    thr, u = P.best_threshold(y, oof)
    print(f"\npooled OOF best thr={thr:.4f} utility={u}  (theory 0.0323)")
    for k in range(6):
        va = fold == k
        P.report(f" fold {k}", y[va], oof[va], thr)
    for t in (0.03, 0.05, 0.1, 0.2, 0.3, 0.5):
        print(f" fixed thr {t}: utility={P.utility(y, (oof >= t).astype(int))}")
    pd.DataFrame({"ID": lab["ID"], "fold": fold, "y": y, "oof": oof}).to_csv(
        P.OUT / "oof.csv", index=False)
    if not args.final:
        return

    m = CatBoostClassifier(iterations=args.iters, learning_rate=0.1, depth=6,
                           random_seed=42, verbose=0, thread_count=-1, allow_writing_files=False)
    m.fit(Pool(lab[feats], y, cat_features=cats))
    te = df[df["is_test"] == 1]
    prob = pd.Series(m.predict_proba(Pool(te[feats], cat_features=cats))[:, 1],
                     index=te["ID"].to_numpy())
    ids = pd.read_csv(P.DATA / "fraudTest.csv", usecols=["ID"])["ID"]
    prob = prob.loc[ids]
    sub = pd.DataFrame({"ID": ids.to_numpy(), "is_fraud": (prob.to_numpy() >= thr).astype(int)})
    assert len(sub) == len(ids) and sub["ID"].is_unique and set(sub["is_fraud"]) <= {0, 1}
    sub.to_csv(P.OUT / "submission.csv", index=False)
    prob.rename("prob").to_csv(P.OUT / "test_probabilities.csv", index_label="ID")
    te_m = te.assign(flag=prob.loc[te["ID"]].to_numpy() >= thr)
    print(f"\nsubmission rows={len(sub)} flagged={int(sub['is_fraud'].sum())} "
          f"({sub['is_fraud'].mean():.4%}); OOF flagged rate "
          f"{(oof >= thr).mean():.4%}, OOF true fraud rate {y.mean():.4%}")
    print(te_m.groupby(te_m['ts'].dt.to_period('M'))['flag'].mean().round(5).to_string())


if __name__ == "__main__":
    P.OUT.mkdir(exist_ok=True)
    main()

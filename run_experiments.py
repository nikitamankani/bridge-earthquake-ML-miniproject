"""Run the complete workflow: cleaning -> EDA -> evaluation -> interpretation -> final model.

Usage (from the repository root):
    python run_experiments.py              # full run, 5 CV repeats
    python run_experiments.py --repeats 2  # quicker run

Evaluation protocols
  A. Original protocol (kept as a reference baseline): one stratified random
     70/30 split of ROWS. Rows from the same bridge can appear in both train
     and test, so this estimate is optimistic.
  B. Row-level CV: repeated stratified 5-fold CV on rows, with nested tuning.
  C. Bridge-grouped CV (PRIMARY): repeated StratifiedGroupKFold (5 folds) with
     all rows of a bridge kept in the same fold, nested grouped tuning.
     B and C use identical models and tuning, so their difference isolates the
     effect of letting a bridge appear in both train and test.
"""

import argparse
import json
from collections import Counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (GridSearchCV, StratifiedGroupKFold, StratifiedKFold,
                                     train_test_split)

from src.data_loader import (BINARY_TARGET, CATEGORICAL_COLS, DAMAGE_LEVEL_SHORT, FEATURE_COLS,
                             GROUP_COL, MULTI_TARGET, NUMERIC_COLS, PROJECT_ROOT, load_clean)
from src.evaluate import binary_metrics, multiclass_report, plot_confusion_grid, plot_roc_pr
from src.models import RANDOM_STATE, binary_models, multiclass_model

RESULTS = PROJECT_ROOT / "results"
FIG = RESULTS / "figures"
MET = RESULTS / "metrics"
BIN_LABELS = ["Undamaged", "Damaged"]
METRIC_COLS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


def damage_score(model, X):
    """Continuous score for the damaged class: probability if available, else SVM decision value."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def fit_model(pipe, grid, X, y, groups, inner_cv):
    """Fit one pipeline; tune with GridSearchCV (F1) on the training rows only."""
    if grid is None:
        return clone(pipe).fit(X, y), {}
    search = GridSearchCV(clone(pipe), grid, scoring="f1", cv=inner_cv, n_jobs=-1)
    search.fit(X, y, groups=groups)
    return search.best_estimator_, search.best_params_


# --------------------------------------------------------------------------- data + EDA
def eda(df):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    df[BINARY_TARGET].value_counts().sort_index().plot.bar(ax=axes[0], color=["steelblue", "crimson"])
    axes[0].set_xticklabels(["Undamaged (0)", "Damaged (1)"], rotation=0)
    axes[0].set_title(f"Binary target: {(df.damaged == 0).sum()} undamaged vs {(df.damaged == 1).sum()} damaged")
    df[MULTI_TARGET].value_counts().sort_index().plot.bar(ax=axes[1], color="teal")
    axes[1].set_xticklabels(DAMAGE_LEVEL_SHORT, rotation=0)
    axes[1].set_title("Damage level (y2)")
    for ax in axes:
        for p in ax.patches:
            ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(FIG / "target_balance.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, c in zip(axes, NUMERIC_COLS):
        for v, col in zip([0, 1], ["steelblue", "crimson"]):
            ax.hist(df.loc[df.damaged == v, c], bins=20, alpha=0.6, color=col, density=True, label=BIN_LABELS[v])
        ax.set_title(c); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "feature_distributions.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for v, col, m in zip([0, 1], ["steelblue", "crimson"], ["o", "X"]):
        s = df[df.damaged == v]
        ax.scatter(s.magnitude, s.distance_mi, c=col, marker=m, s=60, alpha=0.6, edgecolor="k", label=BIN_LABELS[v])
    ax.set_xlabel("Earthquake magnitude"); ax.set_ylabel("Distance to epicenter (miles)")
    ax.set_title("Magnitude vs distance by damage outcome"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "mag_vs_dist.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, c in zip(axes, CATEGORICAL_COLS):
        pd.crosstab(df[c], df.damaged, normalize="index").plot.bar(ax=ax, stacked=True, color=["steelblue", "crimson"])
        n = df[c].value_counts()
        ax.set_xticklabels([f"{t.get_text()}\n(n={n[t.get_text()]})" for t in ax.get_xticklabels()], rotation=0)
        ax.set_title(f"{c}: share damaged"); ax.set_ylabel("proportion"); ax.legend(BIN_LABELS)
    fig.tight_layout(); fig.savefig(FIG / "cat_vs_target.png", dpi=150); plt.close(fig)


# --------------------------------------------------------------------------- protocol A
def protocol_a(df):
    X, y, g = df[FEATURE_COLS], df[BINARY_TARGET].values, df[GROUP_COL].values
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.3, stratify=y, random_state=RANDOM_STATE)
    overlap = sorted(set(g[idx_tr]) & set(g[idx_te]))
    info = {"train_rows": len(idx_tr), "test_rows": len(idx_te), "test_damaged": int(y[idx_te].sum()),
            "test_bridges": int(len(set(g[idx_te]))), "test_bridges_also_in_train": len(overlap)}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, cms = {}, {}
    for name, (pipe, grid) in binary_models(y[idx_tr]).items():
        model, _ = fit_model(pipe, grid, X.iloc[idx_tr], y[idx_tr], None, cv)
        prob = damage_score(model, X.iloc[idx_te])
        pred = model.predict(X.iloc[idx_te])
        rows[name] = binary_metrics(y[idx_te], pred, prob)
        cms[name] = confusion_matrix(y[idx_te], pred, labels=[0, 1])
    table = pd.DataFrame(rows).T

    # multi-class on the same split
    ym = df[MULTI_TARGET].values
    tr_m, te_m = train_test_split(np.arange(len(df)), test_size=0.3, stratify=ym, random_state=RANDOM_STATE)
    mc = multiclass_model(ym[tr_m]).fit(X.iloc[tr_m], ym[tr_m])
    pred_m = mc.predict(X.iloc[te_m])
    mc_rep = multiclass_report(ym[te_m], pred_m, DAMAGE_LEVEL_SHORT)
    mc_cm = confusion_matrix(ym[te_m], pred_m, labels=[0, 1, 2])
    return table, cms, info, mc_rep, mc_cm


# --------------------------------------------------------------------------- protocols B + C
def repeated_cv(df, grouped, repeats, collect_importance=False):
    X, y, g = df[FEATURE_COLS], df[BINARY_TARGET].values, df[GROUP_COL].values
    names = list(binary_models(y).keys())
    per_repeat, oof0, params, folds, perm = [], None, {n: [] for n in names}, [], []
    for r in range(repeats):
        seed = RANDOM_STATE + r
        if grouped:
            outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
        else:
            outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
        prob = {n: np.zeros(len(y)) for n in names}
        pred = {n: np.zeros(len(y), dtype=int) for n in names}
        for k, (tr, te) in enumerate(outer.split(X, y, g if grouped else None)):
            folds.append({"repeat": r, "fold": k, "test_rows": len(te), "test_damaged": int(y[te].sum()),
                          "test_bridges": len(set(g[te])),
                          "test_bridges_also_in_train": len(set(g[te]) & set(g[tr]))})
            for name, (pipe, grid) in binary_models(y[tr]).items():
                model, bp = fit_model(pipe, grid, X.iloc[tr], y[tr], g[tr] if grouped else None, inner)
                prob[name][te] = damage_score(model, X.iloc[te])
                pred[name][te] = model.predict(X.iloc[te])
                if bp:
                    params[name].append(json.dumps(bp, sort_keys=True))
                if collect_importance and r == 0 and name != "Dummy (majority)":
                    pi = permutation_importance(model, X.iloc[te], y[te], scoring="average_precision",
                                                n_repeats=20, random_state=RANDOM_STATE)
                    for f, v in zip(FEATURE_COLS, pi.importances_mean):
                        perm.append({"model": name, "fold": k, "feature": f, "importance": v})
        m = {n: binary_metrics(y, pred[n], prob[n]) for n in names}
        per_repeat.append(m)
        if r == 0:
            oof0 = (prob, pred)
        print(f"  repeat {r + 1}/{repeats} done ({'grouped' if grouped else 'row-level'})")

    summary = {}
    for n in names:
        d = pd.DataFrame([rep[n] for rep in per_repeat])
        summary[n] = {**{f"{c}_mean": d[c].mean() for c in d.columns}, **{f"{c}_std": d[c].std(ddof=0) for c in METRIC_COLS}}
    common = {n: (Counter(p).most_common(1)[0] if p else None) for n, p in params.items()}
    return pd.DataFrame(summary).T, oof0, common, pd.DataFrame(folds), pd.DataFrame(perm)


def multiclass_grouped(df, repeats):
    X, ym, g = df[FEATURE_COLS], df[MULTI_TARGET].values, df[GROUP_COL].values
    reps, cm_total = [], np.zeros((3, 3), dtype=int)
    for r in range(repeats):
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + r)
        pred = np.zeros(len(ym), dtype=int)
        for tr, te in outer.split(X, ym, g):
            pred[te] = multiclass_model(ym[tr]).fit(X.iloc[tr], ym[tr]).predict(X.iloc[te])
        reps.append(multiclass_report(ym, pred, DAMAGE_LEVEL_SHORT))
        cm_total += confusion_matrix(ym, pred, labels=[0, 1, 2])
    stacked = pd.concat(reps)
    mean = stacked.groupby(level=0, sort=False).mean()
    std = stacked.groupby(level=0, sort=False).std(ddof=0)
    return mean, std, cm_total


# --------------------------------------------------------------------------- main
def fmt_table(t, cols, std=False):
    out = pd.DataFrame(index=t.index)
    for c in cols:
        out[c] = (t[f"{c}_mean"].map("{:.3f}".format) + " ± " + t[f"{c}_std"].map("{:.3f}".format)) if std else t[c].map("{:.3f}".format)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5, help="number of repeated CV runs (default 5)")
    args = ap.parse_args()
    for d in (FIG, MET, RESULTS / "models"):
        d.mkdir(parents=True, exist_ok=True)

    df = load_clean()
    df.to_csv(RESULTS / "cleaned_data.csv", index=False)
    print(f"Clean data: {len(df)} rows | damaged={df.damaged.sum()} undamaged={(df.damaged == 0).sum()} | "
          f"levels I/II/III={df.damage_level.value_counts().reindex(['I', 'II', 'III']).tolist()} | "
          f"raw bridge names={df.bridge.nunique()} physical bridges={df[GROUP_COL].nunique()}")
    eda(df)

    print("Protocol A: original stratified 70/30 row split ...")
    a_table, a_cms, a_info, a_mc, a_mc_cm = protocol_a(df)
    a_table.to_csv(MET / "A_random_split_binary.csv")
    a_mc.to_csv(MET / "A_random_split_multiclass.csv")
    print(a_info); print(a_table[METRIC_COLS].round(3))
    plot_confusion_grid({n: a_cms[n] for n in ["Random Forest + SMOTE", "XGBoost (weighted)", "SVM (RBF) + SMOTE"]},
                        BIN_LABELS, "Protocol A (single random 70/30 split; optimistic)", FIG / "A_confusion_matrices.png")

    print("Protocol B: repeated row-level 5-fold CV ...")
    b_table, _, _, b_folds, _ = repeated_cv(df, grouped=False, repeats=args.repeats)
    b_table.to_csv(MET / "B_rowlevel_cv_binary.csv")

    print("Protocol C: repeated bridge-grouped 5-fold CV (primary) ...")
    c_table, (c_prob, c_pred), c_params, c_folds, perm = repeated_cv(df, grouped=True, repeats=args.repeats,
                                                                     collect_importance=True)
    c_table.to_csv(MET / "C_grouped_cv_binary.csv")
    c_folds.to_csv(MET / "C_grouped_fold_composition.csv", index=False)
    print(fmt_table(c_table, METRIC_COLS, std=True))

    y = df[BINARY_TARGET].values
    models_plot = ["LogReg (magnitude only)", "LogReg (all features)", "Random Forest + SMOTE",
                   "XGBoost (weighted)", "SVM (RBF) + SMOTE"]
    plot_roc_pr({n: (y, c_prob[n]) for n in models_plot}, "(bridge-grouped CV, repeat 1)",
                FIG / "C_roc_curves.png", FIG / "C_pr_curves.png")
    plot_confusion_grid({n.split(" (")[0].replace(" + SMOTE", ""): confusion_matrix(y, c_pred[n], labels=[0, 1])
                         for n in ["LogReg (all features)", "Random Forest + SMOTE", "XGBoost (weighted)", "SVM (RBF) + SMOTE"]},
                        BIN_LABELS, "Protocol C: bridge-grouped CV, out-of-fold predictions for all 249 rows (repeat 1)",
                        FIG / "C_confusion_matrices.png")
    oof = df[[GROUP_COL, "earthquake", "magnitude", "distance_mi", "age", BINARY_TARGET]].copy()
    for n in models_plot:
        oof[f"score_{n}"] = c_prob[n].round(4)  # probability, or SVM decision value
    oof.to_csv(MET / "C_grouped_oof_predictions_repeat1.csv", index=False)

    # comparison A vs B vs C (F1 and PR-AUC)
    cmp = pd.DataFrame({
        "A_split_F1": a_table["f1"], "B_rowCV_F1": b_table["f1_mean"], "C_groupCV_F1": c_table["f1_mean"],
        "A_split_PR_AUC": a_table["pr_auc"], "B_rowCV_PR_AUC": b_table["pr_auc_mean"], "C_groupCV_PR_AUC": c_table["pr_auc_mean"],
    })
    cmp.to_csv(MET / "protocol_comparison.csv")

    # permutation importance (grouped CV, repeat 1, held-out folds)
    perm_mean = perm.groupby(["model", "feature"]).importance.mean().unstack()
    perm_std = perm.groupby(["model", "feature"]).importance.std(ddof=0).unstack()
    perm_mean.to_csv(MET / "C_permutation_importance.csv")
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.2), sharex=True)
    for ax, n in zip(axes, ["LogReg (all features)", "Random Forest + SMOTE", "XGBoost (weighted)", "SVM (RBF) + SMOTE"]):
        s = perm_mean.loc[n].sort_values()
        ax.barh(s.index, s.values, xerr=perm_std.loc[n, s.index].values, color="steelblue", capsize=3)
        ax.axvline(0, color="k", lw=0.8); ax.set_title(n, fontsize=10); ax.set_xlabel("drop in PR-AUC when permuted")
    fig.suptitle("Permutation importance on held-out bridges (grouped CV, mean ± std over 5 folds)", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "C_permutation_importance.png", dpi=150); plt.close(fig)

    print("Multi-class (damage level) with bridge-grouped CV ...")
    mc_mean, mc_std, mc_cm = multiclass_grouped(df, args.repeats)
    mc_mean.to_csv(MET / "C_grouped_multiclass_mean.csv"); mc_std.to_csv(MET / "C_grouped_multiclass_std.csv")
    plot_confusion_grid({"A: random 70/30 split (test set)": a_mc_cm,
                         f"C: grouped CV (summed over {args.repeats} repeats)": mc_cm},
                        DAMAGE_LEVEL_SHORT, "Multi-class damage level (XGBoost + SMOTE)", FIG / "multiclass_confusion.png")
    print(mc_mean.round(3))

    # final model: best mean PR-AUC under grouped CV, refit on all rows with grouped tuning
    cand = c_table.drop(index="Dummy (majority)")
    best = cand["pr_auc_mean"].idxmax()
    pipe, grid = binary_models(y)[best]
    final, final_params = fit_model(pipe, grid, df[FEATURE_COLS], y, df[GROUP_COL].values,
                                    StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE))
    model_path = RESULTS / "models" / "final_model.joblib"
    joblib.dump(final, model_path)
    reloaded = joblib.load(model_path)
    assert np.allclose(damage_score(reloaded, df[FEATURE_COLS]), damage_score(final, df[FEATURE_COLS]))
    assert (reloaded.predict(df[FEATURE_COLS]) == final.predict(df[FEATURE_COLS])).all()
    print(f"Final model: {best} {final_params} -> {model_path.relative_to(PROJECT_ROOT)} (reload check passed)")

    summary = {
        "data": {"rows": len(df), "damaged": int(y.sum()), "undamaged": int((y == 0).sum()),
                 "level_counts": df.damage_level.value_counts().reindex(["I", "II", "III"]).astype(int).tolist(),
                 "raw_bridge_names": int(df.bridge.nunique()), "physical_bridges": int(df[GROUP_COL].nunique())},
        "repeats": args.repeats, "protocol_A_split_info": a_info,
        "grouped_fold_test_damaged_min_max": [int(c_folds.test_damaged.min()), int(c_folds.test_damaged.max())],
        "grouped_fold_bridge_overlap_max": int(c_folds.test_bridges_also_in_train.max()),
        "rowlevel_fold_bridge_overlap_mean": float(b_folds.test_bridges_also_in_train.mean()),
        "most_common_params_grouped": {k: v for k, v in c_params.items() if v},
        "final_model": best, "final_params": final_params,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    with open(RESULTS / "results_summary.md", "w") as f:
        f.write("# Results summary (auto-generated by run_experiments.py)\n\n")
        f.write(f"Data: {json.dumps(summary['data'])}\n\n")
        f.write(f"## Protocol A: single stratified random 70/30 row split (optimistic reference)\n\n{json.dumps(a_info)}\n\n")
        f.write(a_table[METRIC_COLS + ["tp", "fp", "fn", "tn"]].round(3).to_markdown() + "\n\n")
        f.write(f"## Protocol B: row-level stratified 5-fold CV, {args.repeats} repeats (mean ± std)\n\n")
        f.write(fmt_table(b_table, METRIC_COLS, std=True).to_markdown() + "\n\n")
        f.write(f"## Protocol C: bridge-grouped stratified 5-fold CV, {args.repeats} repeats (mean ± std) — PRIMARY\n\n")
        f.write(fmt_table(c_table, METRIC_COLS, std=True).to_markdown() + "\n\n")
        f.write("Mean pooled confusion counts per repeat (TP / FP / FN / TN):\n\n")
        f.write(c_table[["tp_mean", "fp_mean", "fn_mean", "tn_mean"]].round(1).to_markdown() + "\n\n")
        f.write("## Protocol comparison\n\n" + cmp.round(3).to_markdown() + "\n\n")
        f.write("## Multi-class: Protocol A test set (single split)\n\n" + a_mc.round(3).to_markdown() + "\n\n")
        f.write(f"## Multi-class: Protocol C grouped CV, mean over {args.repeats} repeats\n\n" + mc_mean.round(3).to_markdown() + "\n\n")
        f.write("std over repeats:\n\n" + mc_std.round(3).to_markdown() + "\n\n")
        f.write("## Permutation importance (grouped CV, repeat 1; drop in PR-AUC)\n\n" + perm_mean.round(3).to_markdown() + "\n\n")
        f.write(f"## Final model\n\n{best}, params {final_params}\n")
    print("Done. See results/results_summary.md")


if __name__ == "__main__":
    main()

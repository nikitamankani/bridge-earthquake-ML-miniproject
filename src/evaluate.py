"""Metric and plotting helpers (binary + multi-class)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_fscore_support, precision_score, recall_score, roc_auc_score,
    roc_curve, precision_recall_curve,
)


def binary_metrics(y_true, y_pred, y_proba) -> dict:
    """Metrics for the positive (damaged) class plus threshold-free ROC-AUC / PR-AUC."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def multiclass_report(y_true, y_pred, labels) -> pd.DataFrame:
    """Per-class precision / recall / F1 / support plus accuracy, macro and weighted F1."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    rep = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=labels)
    rep.loc["accuracy"] = [np.nan, np.nan, accuracy_score(y_true, y_pred), len(y_true)]
    rep.loc["macro avg"] = [p.mean(), r.mean(), f1_score(y_true, y_pred, average="macro", zero_division=0), len(y_true)]
    rep.loc["weighted avg"] = [np.average(p, weights=s), np.average(r, weights=s),
                               f1_score(y_true, y_pred, average="weighted", zero_division=0), len(y_true)]
    return rep


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_grid(cms: dict, labels, title, path):
    """cms: {panel title: confusion matrix (2-D array)}."""
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.4), squeeze=False)
    for ax, (name, cm) in zip(axes[0], cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(name, fontsize=10)
        ax.tick_params(axis="x", rotation=0 if len(labels) == 2 else 20)
        ax.tick_params(axis="y", rotation=0 if len(labels) == 3 else 90)
    fig.suptitle(title, fontsize=11)
    _save(fig, path)


def plot_roc_pr(curves: dict, title_suffix, roc_path, pr_path):
    """curves: {model name: (y_true, y_proba)} pooled out-of-fold predictions."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, (yt, yp) in curves.items():
        fpr, tpr, _ = roc_curve(yt, yp)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc_score(yt, yp):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curves {title_suffix}", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _save(fig, roc_path)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, (yt, yp) in curves.items():
        prec, rec, _ = precision_recall_curve(yt, yp)
        ax.plot(rec, prec, lw=2, label=f"{name} (AP={average_precision_score(yt, yp):.3f})")
    base = np.mean(list(curves.values())[0][0])
    ax.axhline(base, color="k", ls="--", alpha=0.4, label=f"Chance (damaged rate={base:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-recall curves {title_suffix}", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _save(fig, pr_path)

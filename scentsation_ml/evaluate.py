"""Plots and reports for trained models."""

import logging
import os
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

CLASS_NAMES = ["NEUTRAL", "RELAXED", "STRESSED", "FOCUSED"]
CLASS_COLORS = ["#95a5a6", "#3498db", "#e74c3c", "#2ecc71"]


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list = None,
    save_path: str = "results/confusion_matrix.png",
    normalize: bool = True,
) -> str:
    """Save confusion matrix PNG."""
    if class_names is None:
        class_names = CLASS_NAMES
    if normalize:
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        row_sums[row_sums == 0] = 1
        cm_plot = cm.astype(float) / row_sums
        cm_plot = np.nan_to_num(cm_plot)
        fmt = ".2f"
        title = "Confusion Matrix (Normalized)"
    else:
        cm_plot = cm
        fmt = "d"
        title = "Confusion Matrix (Counts)"
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm_plot.shape[1]),
        yticks=np.arange(cm_plot.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True",
        xlabel="Predicted",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm_plot.max() / 2.0 if cm_plot.size else 0
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            ax.text(
                j,
                i,
                format(cm_plot[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm_plot[i, j] > thresh else "black",
            )
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_roc_curves(
    y_test: np.ndarray,
    y_proba: np.ndarray,
    class_names: list = None,
    save_path: str = "results/roc_curves.png",
) -> str:
    """One-vs-rest ROC curves."""
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import label_binarize

    if class_names is None:
        class_names = CLASS_NAMES
    n_classes = len(class_names)
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, (name, color) in enumerate(zip(class_names, CLASS_COLORS)):
        if y_bin.shape[1] <= i:
            break
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("One-vs-Rest ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_feature_importance(
    importance: np.ndarray,
    feature_names: list,
    save_path: str = "results/feature_importance.png",
) -> str:
    """Horizontal bar chart of importances."""
    order = np.argsort(importance)
    names = [feature_names[i] for i in order]
    values = importance[order]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(names)), values, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(names))))
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def generate_report(
    results: Dict[str, Dict[str, Any]],
    feature_names: list,
    report_path: str = "results/report.txt",
) -> str:
    """Write text summary."""
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    lines = ["Scentsation ML report", "=" * 50, ""]
    best = max(results, key=lambda k: results[k].get("f1_macro", 0))
    lines.append(f"Best (by val F1): {best}\n")
    for name, m in results.items():
        lines.append(f"{name}: acc={m.get('accuracy', 0):.4f} f1={m.get('f1_macro', 0):.4f}")
    lines.append("")
    lines.append("Features:")
    for i, fn in enumerate(feature_names, 1):
        lines.append(f"  {i}. {fn}")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    return report_path

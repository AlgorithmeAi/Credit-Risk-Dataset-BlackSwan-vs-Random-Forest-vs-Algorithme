import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from sklearn.calibration import calibration_curve
from matplotlib.backends.backend_pdf import PdfPages

def plot_model_performances_auc(df, target_col, confidence_labels, pdf_name='model_performance_auc_output.pdf'):
    def compute_metrics(y_true, y_scores):
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)
        youden = tpr - fpr
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        accuracy_by_threshold = []
        for thresh in thresholds_roc:
            y_pred = (y_scores >= thresh).astype(int)
            acc = np.mean(y_pred == y_true)
            accuracy_by_threshold.append(acc)
        return {
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "youden": youden,
            "thresholds_roc": thresholds_roc,
            "thresholds_pr": thresholds_pr,
            "accuracy": np.array(accuracy_by_threshold),
        }

    def format_auc(v):
        return "{:.3f}".format(v).replace(".", ",")

    y_true = df[target_col].astype(int)
    model_colors = ['purple'] + ['green'] + ['gold'] + (['purple'] * 5)
    model_styles = ['-', ':', '--', '--', '--', '--', '--', '--', ':', ':', ':', ':', ':']
    labels = [label.replace('_', ' ').title() for label in confidence_labels]

    metrics_list = []
    for conf_label in confidence_labels:
        scores = df[conf_label] / 100 if df[conf_label].max() > 1 else df[conf_label]
        metrics_list.append(compute_metrics(y_true, scores))

    aucs_roc = [m["auc"] for m in metrics_list]
    aucs_recall = [np.trapz(m["recall"][:-1], m["thresholds_pr"]) for m in metrics_list]
    aucs_f1 = [np.trapz(m["f1"][:-1], m["thresholds_pr"]) for m in metrics_list]
    aucs_precision = [np.trapz(m["precision"][:-1], m["thresholds_pr"]) for m in metrics_list]

    aucs_youden = []
    for m in metrics_list:
        sorted_indices = np.argsort(m["thresholds_roc"])
        sorted_thresholds = m["thresholds_roc"][sorted_indices]
        sorted_youden = m["youden"][sorted_indices]
        interp_youden = np.interp(np.linspace(0, 1, len(sorted_thresholds)), sorted_thresholds, sorted_youden)
        aucs_youden.append(abs(np.trapz(interp_youden, np.linspace(0, 1, len(sorted_thresholds)))))

    winners = {
        "roc": np.argmax(aucs_roc),
        "recall": np.argmax(aucs_recall),
        "f1": np.argmax(aucs_f1),
        "youden": np.argmax(aucs_youden),
        "precision": np.argmax(aucs_precision)
    }

    fig, axs = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle("Model Performance Evaluation (Backtest Set) Focus on AUC-Based Metrics", fontsize=14)

    for i, (label, color, style, m) in enumerate(zip(labels, model_colors, model_styles, metrics_list)):
        star = "*" if i == winners["roc"] else ""
        axs[0, 0].plot(m["fpr"], m["tpr"], linestyle=style, color=color,
                       label="{}{} ROC AUC = {}".format(star, label, format_auc(aucs_roc[i])), linewidth=1.5)

        star = "*" if i == winners["youden"] else ""
        axs[0, 1].plot(m["thresholds_roc"], m["youden"], linestyle=style, color=color,
                       label="{}{} Youden'J".format(star, label), linewidth=1.5)

        star = "*" if i == winners["precision"] else ""
        axs[0, 2].plot(m["thresholds_pr"], m["precision"][:-1], linestyle=style, color=color,
                       label="{}{} Precision AUC = {}".format(star, label, format_auc(aucs_precision[i])), linewidth=1.5)

        star = "*" if i == winners["recall"] else ""
        axs[1, 0].plot(m["thresholds_pr"], m["recall"][:-1], linestyle=style, color=color,
                       label="{}{} Recall AUC = {}".format(star, label, format_auc(aucs_recall[i])), linewidth=1.5)

        star = "*" if i == winners["f1"] else ""
        axs[1, 1].plot(m["thresholds_pr"], m["f1"][:-1], linestyle=style, color=color,
                       label="{}{} F1 AUC = {}".format(star, label, format_auc(aucs_f1[i])), linewidth=1.5)

        scores = df[confidence_labels[i]] / 100 if df[confidence_labels[i]].max() > 1 else df[confidence_labels[i]]
        prob_true, prob_pred = calibration_curve(y_true, scores, n_bins=10, strategy='uniform')
        axs[1, 2].plot(prob_pred, prob_true, linestyle=style, color=color, marker='o', label=label, linewidth=1.5)

        axs[2, 0].plot(m["thresholds_roc"], m["accuracy"], linestyle=style, color=color,
                       label="{} Accuracy".format(label), linewidth=1.5)

    axs[0, 0].set_title("ROC AUC")
    axs[0, 0].set_xlabel("False Positive Rate")
    axs[0, 0].set_ylabel("True Positive Rate")
    axs[0, 0].legend()

    axs[0, 1].set_title("Youden's Index (TPR - FPR) by Threshold")
    axs[0, 1].set_xlabel("Threshold")
    axs[0, 1].set_ylabel("Youden Index")
    axs[0, 1].legend()

    axs[0, 2].set_title("Precision by Threshold (AUC)")
    axs[0, 2].set_xlabel("Threshold")
    axs[0, 2].set_ylabel("Precision")
    axs[0, 2].legend()

    axs[1, 0].set_title("Recall by Threshold (AUC)")
    axs[1, 0].set_xlabel("Threshold")
    axs[1, 0].set_ylabel("Recall")
    axs[1, 0].legend()

    axs[1, 1].set_title("F1-score by Threshold (AUC)")
    axs[1, 1].set_xlabel("Threshold")
    axs[1, 1].set_ylabel("F1-score")
    axs[1, 1].legend()

    axs[1, 2].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    axs[1, 2].set_title("Calibration Curve")
    axs[1, 2].set_xlabel("Predicted Probability")
    axs[1, 2].set_ylabel("True Frequency")
    axs[1, 2].legend()

    axs[2, 0].set_title("Accuracy by Threshold")
    axs[2, 0].set_xlabel("Threshold")
    axs[2, 0].set_ylabel("Accuracy")
    axs[2, 0].legend()

    fig.delaxes(axs[2, 1])
    fig.delaxes(axs[2, 2])

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    with PdfPages(pdf_name) as pdf:
        pdf.savefig(fig)
        plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python plot_model.py <csv_path> <target_column> <confidence_col1> <confidence_col2> ...")
        sys.exit(1)

    csv_path = sys.argv[1]
    target_column = sys.argv[2]
    confidence_cols = sys.argv[3:]

    df = pd.read_csv(csv_path)
    filename = os.path.splitext(os.path.basename(csv_path))[0]
    output_pdf = "{}_performance_auc.pdf".format(filename)

    plot_model_performances_auc(df, target_column, confidence_cols, output_pdf)

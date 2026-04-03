import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report
import json
import os

def plot_global_performance(predictions_csv, output_dir):
    """
    Generates ROC/PR curves and Confusion Matrix for the diagnostic system.
    """
    if not os.path.exists(predictions_csv):
        print(f"Error: {predictions_csv} not found.")
        return

    df = pd.read_csv(predictions_csv)
    # Target the test set for scientific validation
    test_df = df[df['split'] == 'test'] if 'test' in df['split'].unique() else df
    
    # Label mapping: NORMAL=0, PNEUMONIA=1
    y_true = (test_df['true_label'] == 'PNEUMONIA').astype(int)
    y_scores = test_df['prob_pneumonia']
    y_pred = (test_df['predicted_label'] == 'PNEUMONIA').astype(int)

    os.makedirs(output_dir, exist_ok=True)

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # 4. Statistical Metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    
    metrics = {
        'auc_roc': float(roc_auc),
        'auc_pr': float(pr_auc),
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    with open(os.path.join(output_dir, 'statistical_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # 5. Combined Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ROC Plot
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Receiver Operating Characteristic')
    axes[0].legend(loc="lower right")

    # PR Plot
    axes[1].plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend(loc="lower left")

    # Confusion Matrix Plot
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
                xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('True')
    axes[2].set_title('Confusion Matrix')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_plots.png'))
    print(f"Performance plots saved to {output_dir}")

if __name__ == "__main__":
    plot_global_performance(
        predictions_csv='results/predictions.csv',
        output_dir='outputs'
    )

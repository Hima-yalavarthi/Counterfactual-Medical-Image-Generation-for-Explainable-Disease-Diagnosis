import pandas as pd
import json
import os
import numpy as np

def generate_global_summary(predictions_csv, metrics_json, output_path):
    """
    Aggregates all project metrics into a single global summary for the dashboard.
    """
    if not os.path.exists(predictions_csv):
        print(f"Error: {predictions_csv} not found.")
        return

    df = pd.read_csv(predictions_csv)
    
    # 1. Classifier Performance
    # Filter for test set if available, otherwise use all
    test_df = df[df['split'] == 'test'] if 'test' in df['split'].unique() else df
    
    correct = (test_df['true_label'] == test_df['predicted_label']).sum()
    accuracy = correct / len(test_df)
    
    # Class-wise accuracy
    class_acc = {}
    for label in test_df['true_label'].unique():
        class_df = test_df[test_df['true_label'] == label]
        class_correct = (class_df['true_label'] == class_df['predicted_label']).sum()
        class_acc[label] = class_correct / len(class_df)

    # 2. Confidence Stats
    mean_conf_correct = test_df[test_df['true_label'] == test_df['predicted_label']]['confidence'].mean()
    mean_conf_incorrect = test_df[test_df['true_label'] != test_df['predicted_label']]['confidence'].mean()

    # 3. GAN Metrics (from metrics_json)
    gan_metrics = {}
    if os.path.exists(metrics_json):
        with open(metrics_json, 'r') as f:
            data = json.load(f)
            gan_metrics['mean_ssim'] = data.get('summary', {}).get('mean_ssim', 0)
            gan_metrics['mean_lpips'] = data.get('summary', {}).get('mean_lpips', 0)
            gan_metrics['flip_rate'] = data.get('flip_rate_summary', {}).get('flip_rate', 0)

    # 4. Final Summary object
    summary = {
        'classifier': {
            'overall_accuracy': float(accuracy),
            'class_accuracy': class_acc,
            'mean_confidence_correct': float(mean_conf_correct),
            'mean_confidence_incorrect': float(mean_conf_incorrect),
            'total_samples': len(test_df)
        },
        'generator': gan_metrics,
        'metadata': {
            'generated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset': 'Chest X-Ray (Pneumonia)'
        }
    }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=4)
        
    print(f"Global summary saved to {output_path}")

if __name__ == "__main__":
    generate_global_summary(
        predictions_csv='results/predictions.csv',
        metrics_json='results/evaluation_metrics.json',
        output_path='results/global_dataset_summary.json'
    )

import pandas as pd
from sklearn.metrics import (
    recall_score, precision_score, f1_score, accuracy_score, cohen_kappa_score, confusion_matrix
)

for model in ['svm', 'lstm', 'bert']:
    # Load predictions
    df = pd.read_csv(f'preds/val_predictions_{model}.csv')
    y_true = df['true_label']
    y_pred = df['predicted_label']
    labels = sorted(df['true_label'].unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Per-class metrics
    per_class = []
    for i, label in enumerate(labels):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        sensitivity = recall  # same as recall
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_class.append({
            'class': label,
            'Recall': recall,
            'Precision': precision,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'F-measure': f1
        })

    per_class_df = pd.DataFrame(per_class)
    print(f"Per-class metrics for {model}:")
    print(per_class_df)

    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    print(f"\nOverall metrics for {model}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Cohen's kappa: {kappa:.4f}")
    print(f"F1_score_macro: {f1_macro:.4f}")
    print(f"F1_score_micro: {f1_micro:.4f}")
    print(f"F1_score_weighted: {f1_weighted:.4f}")
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from sklearn.model_selection import KFold
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc
)
import numpy as np
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from dataset import *
from model import *

def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + 1e-8)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-8)

    mcc_num = (tp * tn) - (fp * fn)
    mcc_den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = mcc_num / (mcc_den + 1e-8)

    roc_auc = roc_auc_score(y_true, y_prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_curve, precision_curve)
    
    return {
        'Sn': sensitivity,
        'Sp': specificity,
        'Acc': accuracy,
        'MCC': mcc,
        'Precision': precision,
        'F1': f1,
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc
    }

sequences, labels = load_encoding_from_txt('')
features = load_features_from_txt('')

sequences = np.array(sequences)
features = np.array(features)
labels = np.array(labels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
lr = 1e-5
lambda_hsic = 0.6
n_epochs = 100
k_folds = 10

criterion = nn.CrossEntropyLoss()

kf = KFold(n_splits=k_folds, shuffle=True, random_state=64)

fold_metrics = [] 

for fold, (train_idx, val_idx) in enumerate(kf.split(sequences)):
    print(f"\n====== Fold {fold + 1}/{k_folds} ======")

    train_seq, val_seq = sequences[train_idx], sequences[val_idx]
    train_feat, val_feat = features[train_idx], features[val_idx]
    train_lbl, val_lbl = labels[train_idx], labels[val_idx]

    dataset_train = MyDataSet(train_seq, train_feat, train_lbl)
    dataset_val = MyDataSet(val_seq, val_feat, val_lbl)

    train_loader = Data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = Data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    model = FusionPepNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    def train_model():
        model.train()
        total_loss = 0.0
        for input_ids, sequence_features, labels in train_loader:
            input_ids = input_ids.to(device)
            sequence_features = sequence_features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs, feat1, feat2 = model(input_ids, sequence_features)
            loss_ce = criterion(outputs, labels)
            loss_hsic = hsic_loss(feat1, feat2)
            loss = loss_ce + lambda_hsic * loss_hsic
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def evaluate_model():
        model.eval()
        total_loss = 0.0
        all_probs = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for input_ids, sequence_features, labels in val_loader:
                input_ids = input_ids.to(device)
                sequence_features = sequence_features.to(device)
                labels = labels.to(device)
                
                outputs, _, _ = model(input_ids, sequence_features)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        metrics = calculate_metrics(all_labels, all_preds, all_probs)
        metrics['Val Loss'] = total_loss / len(val_loader)
        return metrics

    best_metrics = None
    best_acc = 0.0
    
    for epoch in range(n_epochs):
        train_loss = train_model()
        val_metrics = evaluate_model()
        current_acc = val_metrics['Acc']

        if current_acc > best_acc:
            best_acc = current_acc
            best_metrics = val_metrics
            torch.save(model.state_dict(), f"fold_{fold+1}.pth")

        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['Val Loss']:.4f} | "
              f"Acc: {val_metrics['Acc']:.4f} | "
              f"MCC: {val_metrics['MCC']:.4f} | "
              f"ROC-AUC: {val_metrics['ROC-AUC']:.4f}")

    fold_metrics.append(best_metrics)
    for k, v in best_metrics.items():
        if k != 'Val Loss':
            print(f"{k:10}: {v:.4f}")

metric_names = ['Sn', 'Sp', 'Acc', 'MCC', 'Precision', 'F1', 'ROC-AUC', 'PR-AUC']
for name in metric_names:
    values = [m[name] for m in fold_metrics]
    print(f"{name:10}: {np.mean(values):.4f} ± {np.std(values):.4f}")
"""
Stretch Tuesday — Manual Evaluation Harness.

Implement these without using Trainer.predict, sklearn metrics helpers, or
Hugging Face evaluate. The goal is to make the math explicit.
"""

import numpy as np
import torch


def manual_predict(model, tokenizer, texts: list, batch_size: int = 8):
    """
    Run manual PyTorch inference over a list of texts.

    Returns (preds, probs):
      preds: shape (N,), int class indices
      probs: shape (N, num_classes), probabilities (post-softmax)
    """
    all_preds = []
    all_probs = []
    model.eval()
    device = next(model.parameters()).device
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs_batch = torch.softmax(logits, dim=-1).cpu().numpy()
            preds_batch = np.argmax(probs_batch, axis=-1)
        
        all_preds.extend(preds_batch)
        all_probs.extend(probs_batch)
    
    return np.array(all_preds), np.array(all_probs)


def compute_classification_report_from_arrays(y_true, y_pred) -> dict:
    """
    Compute accuracy, per-class precision/recall/F1, and macro-F1 from numpy
    primitives only — no sklearn, no Hugging Face evaluate.

    Returns:
      {
        "accuracy": float,
        "macro_f1": float,
        "per_class": {label_index: {"precision": ..., "recall": ..., "f1": ...}, ...},
      }
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    num_samples = len(y_true)
    
    per_class = {}
    f1_scores = []
    
    for label in unique_labels:
        TP = np.sum((y_true == label) & (y_pred == label))
        FP = np.sum((y_true != label) & (y_pred == label))
        FN = np.sum((y_true == label) & (y_pred != label))
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class[int(label)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }
        f1_scores.append(f1)
    
    accuracy = float(np.sum(y_true == y_pred) / num_samples)
    macro_f1 = float(np.mean(f1_scores))
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class
    }

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


@torch.no_grad()
def test_model(model, test_loader, device):
    model.eval()
    preds, trues = [], []
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits, _, _ = model(x)
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        trues.extend(y.cpu().numpy())
    acc = accuracy_score(trues, preds)
    pre = precision_score(trues, preds, average="macro")
    rec = recall_score(trues, preds, average="macro")
    f1 = f1_score(trues, preds, average="macro")
    print(f"\n--- Test Set ---")
    print(f"Acc: {acc:.4f}  Pre: {pre:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}")
    return acc, pre, rec, f1
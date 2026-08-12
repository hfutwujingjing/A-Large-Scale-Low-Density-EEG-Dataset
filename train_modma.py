import os
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        trues.extend(y.cpu().numpy())
    return accuracy_score(trues, preds), f1_score(trues, preds, average="macro")


def train_modma_teacher(model, train_loader, val_loader, epochs, lr, device, save_path):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    best_val_f1 = 0.0
    for ep in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"MODMA E{ep+1}/{epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": total_loss / (pbar.n + 1)})
        tr_acc, tr_f1 = evaluate(model, train_loader, device)
        va_acc, va_f1 = evaluate(model, val_loader, device)
        print(f"MODMA E{ep+1:>3}  Train Acc:{tr_acc:.4f} F1:{tr_f1:.4f}  Val Acc:{va_acc:.4f} F1:{va_f1:.4f}")
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.encoder.state_dict(), save_path)
    model.encoder.load_state_dict(torch.load(save_path))
    print(f"Best Val F1: {best_val_f1:.4f}")
    return model.encoder
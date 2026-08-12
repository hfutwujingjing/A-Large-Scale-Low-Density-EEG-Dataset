import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    preds, trues = [], []
    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        trues.extend(y.cpu().numpy())
    return total_loss / len(loader), accuracy_score(trues, preds)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds, trues = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _, _ = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        trues.extend(y.cpu().numpy())
    return total_loss / len(loader), accuracy_score(trues, preds), f1_score(trues, preds, average="macro")


def train(model, train_loader, val_loader, epochs, lr, device, freeze_general=False, freeze_class=False):
    model = model.to(device)
    model.set_general_freeze(freeze_general)
    model.set_class_freeze(freeze_class)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    best_f1 = 0.0
    for ep in range(epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, va_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"E{ep+1:>3}  Train Loss:{tr_loss:.4f} Acc:{tr_acc:.4f}  "
              f"Val Loss:{va_loss:.4f} Acc:{va_acc:.4f} F1:{va_f1:.4f}")
        if va_f1 > best_f1:
            best_f1 = va_f1
            os.makedirs("./pth", exist_ok=True)
            torch.save(model.state_dict(), "./pth/hbn_modma_best.pth")
    model.load_state_dict(torch.load("./pth/hbn_modma_best.pth"))
    print(f"Best Val F1: {best_f1:.4f}")
    return model
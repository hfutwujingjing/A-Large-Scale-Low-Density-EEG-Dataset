import os
import torch
import torch.nn as nn
from tqdm import tqdm


def pretrain(model, loader, epochs, lr, device, save_path):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    for ep in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Pretrain E{ep+1}/{epochs}", leave=False)
        for x, _ in pbar:
            x = x.to(device)
            rec = model(x)
            loss = criterion(rec, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": total_loss / (pbar.n + 1)})
        avg_loss = total_loss / len(loader)
        print(f"Pretrain E{ep+1:>3}  MSE: {avg_loss:.6f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.encoder.state_dict(), save_path)
    model.encoder.load_state_dict(torch.load(save_path))
    print(f"Best MSE: {best_loss:.6f}")
    return model.encoder
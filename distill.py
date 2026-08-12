import os
import torch
import torch.nn as nn
from tqdm import tqdm


def distill(distill_model, loader, epochs, lr, device, save_path):
    distill_model = distill_model.to(device)
    optimizer = torch.optim.AdamW(distill_model.student.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    for ep in range(epochs):
        distill_model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Distill E{ep+1}/{epochs}", leave=False)
        for x_128, x_3 in pbar:
            x_128, x_3 = x_128.to(device), x_3.to(device)
            s_feat, t_feat = distill_model(x_128, x_3)
            loss = criterion(s_feat, t_feat.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": total_loss / (pbar.n + 1)})
        avg_loss = total_loss / len(loader)
        print(f"Distill E{ep+1:>3}  MSE: {avg_loss:.6f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(distill_model.student.state_dict(), save_path)
    distill_model.student.load_state_dict(torch.load(save_path))
    print(f"Best MSE: {best_loss:.6f}")
    return distill_model.student
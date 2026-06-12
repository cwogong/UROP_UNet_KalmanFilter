import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from model.Vanilla_UNet import VanillaUNet
from experiments.phase2_test import create_synthetic_frame_sequence


def create_dataset(num_samples=200, image_size=128):
    frames, masks, _ = create_synthetic_frame_sequence(num_frames=num_samples, image_size=image_size)
    X = torch.cat(frames, dim=0)  # (N, C, H, W)
    Y = torch.from_numpy(np.stack(masks, axis=0)).unsqueeze(1).float()  # (N,1,H,W)
    return TensorDataset(X, Y)


def train_unet(epochs=3, batch_size=8, lr=1e-3, save_path='checkpoints/unet_trained_demo.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    unet_config = {'in_channels': 3, 'start_out_channels': 16, 'num_class': 1, 'size': 3, 'padding': 1}
    model = VanillaUNet(**unet_config).to(device)

    dataset = create_dataset(num_samples=200, image_size=128)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(1, epochs+1):
        running_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        avg_loss = running_loss / len(loader.dataset)
        print(f'Epoch {ep}/{epochs} - Loss: {avg_loss:.6f}')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print('Saved model to', save_path)


if __name__ == '__main__':
    train_unet()

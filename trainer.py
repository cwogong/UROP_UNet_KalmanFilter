import argparse
import yaml
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from model.Vanilla_UNet import VanillaUNet
from experiments.phase2_test import create_synthetic_frame_sequence
import os


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def frames_to_dataset(frames, masks):
    # frames: list of torch tensors (1, C, H, W)
    # masks: list of numpy arrays (H, W)
    X = torch.cat(frames, dim=0)  # (N, C, H, W)
    Y = torch.from_numpy(np.stack(masks)).unsqueeze(1).float()  # (N, 1, H, W)
    return TensorDataset(X, Y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-dir', default='checkpoints')
    args = parser.parse_args()

    cfg = load_config(args.config)
    trainer_cfg = cfg.get('trainer', {})
    model_cfg = cfg.get('model', {})

    lr = args.lr if args.lr is not None else trainer_cfg.get('lr', 1e-4)

    device = torch.device(args.device)

    print('Using device:', device)

    # 데이터: 합성 데이터 생성 (빠른 데모용)
    print('Generating synthetic dataset (demo)...')
    frames, masks, _ = create_synthetic_frame_sequence(num_frames=100, image_size=480)
    ds = frames_to_dataset(frames, masks)

    # train/val split
    val_frac = 0.2
    val_size = int(len(ds) * val_frac)
    train_size = len(ds) - val_size
    if val_size > 0:
        train_ds, val_ds = random_split(ds, [train_size, val_size])
    else:
        train_ds, val_ds = ds, None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0) if val_ds is not None else None

    # 모델
    unet_config = {
        'in_channels': model_cfg.get('in_channels', 3),
        'start_out_channels': model_cfg.get('start_out_channels', 32),
        'num_class': model_cfg.get('num_class', 1),
        'size': model_cfg.get('size', 4),
        'padding': model_cfg.get('padding', 1)
    }

    model = VanillaUNet(**unet_config).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, 'demo_unet.pth')
    info_path = os.path.join(args.save_dir, 'training_info.json')

    # 학습 루프 with validation and best-model saving
    print('Starting training...')
    best_val_loss = float('inf')
    best_epoch = -1
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)  # (B, 1, H, W)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        epoch_train_loss = running_loss / len(train_ds)
        history['train_loss'].append(epoch_train_loss)

        # validation
        epoch_val_loss = None
        if val_loader is not None:
            model.eval()
            val_running = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    val_running += loss.item() * xb.size(0)

            epoch_val_loss = val_running / len(val_ds)
            history['val_loss'].append(epoch_val_loss)

            print(f'Epoch {epoch}/{args.epochs} - train_loss: {epoch_train_loss:.6f} - val_loss: {epoch_val_loss:.6f}')

            # save best
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = epoch
                torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'val_loss': best_val_loss}, save_path)
                print(f'  -> New best model saved (val_loss={best_val_loss:.6f})')
        else:
            print(f'Epoch {epoch}/{args.epochs} - train_loss: {epoch_train_loss:.6f}')

    # 최종 정보 저장
    info = {
        'save_path': save_path,
        'best_epoch': best_epoch,
        'best_val_loss': None if best_epoch == -1 else best_val_loss,
        'epochs': args.epochs,
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss']
    }

    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)

    if best_epoch != -1:
        print('Training finished. Best checkpoint saved to', save_path)
        print('Training info saved to', info_path)
    else:
        print('Training finished. No validation performed; final checkpoint not saved as best.')


if __name__ == '__main__':
    main()

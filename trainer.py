import argparse
import yaml
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset, Dataset
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF
from model.Vanilla_UNet import VanillaUNet
from dataset.uav_dataset import UAVTrackingDataset
import os
import time
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 서버 환경 (GUI 없이)
import matplotlib.pyplot as plt


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class SegmentationPairDataset(Dataset):
    def __init__(self, base_dataset, image_size=None):
        self.base_dataset = base_dataset
        self.image_size = image_size

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, targets = self.base_dataset[idx]
        mask = targets['mask'].float().unsqueeze(0) / 255.0

        if self.image_size is not None:
            mask = TF.resize(mask, self.image_size, interpolation=TF.InterpolationMode.NEAREST)

        return image, mask


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
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-dir', default='checkpoints')
    parser.add_argument('--num-sequences', type=int, default=None,
                        help='Use only the first N sequences from the dataset')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit training to this many image-mask pairs')
    args = parser.parse_args()

    cfg = load_config(args.config)
    trainer_cfg = cfg.get('trainer', {})
    model_cfg = cfg.get('model', {})
    dataset_cfg = cfg.get('dataset', {})

    lr = args.lr if args.lr is not None else trainer_cfg.get('lr', 1e-4)
    batch_size = args.batch_size if args.batch_size is not None else dataset_cfg.get('batch_size', 4)
    num_sequences = args.num_sequences if args.num_sequences is not None else dataset_cfg.get('train_sequences', None)
    max_samples = args.max_samples

    data_root = dataset_cfg.get('root')
    mask_root = dataset_cfg.get('root_mask')
    image_size = tuple(dataset_cfg.get('image_size', [480, 480]))

    if data_root is None or mask_root is None:
        raise ValueError('dataset.root and dataset.root_mask must be defined in config.yaml')

    device = torch.device(args.device)

    print('Using device:', device)

    # 실제 데이터셋 로드
    print('Loading dataset from:', data_root)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    raw_dataset = UAVTrackingDataset(
        data_root,
        mask_root,
        transforms=transform,
        num_sequences=num_sequences
    )

    if max_samples is not None and max_samples < len(raw_dataset):
        raw_dataset = Subset(raw_dataset, list(range(max_samples)))
        print(f'Using subset of dataset: {len(raw_dataset)} samples')
    else:
        print(f'Loaded dataset size: {len(raw_dataset)} samples')

    train_ds = SegmentationPairDataset(raw_dataset, image_size=image_size)

    # train/val split
    val_frac = 0.2
    val_size = int(len(train_ds) * val_frac)
    train_size = len(train_ds) - val_size
    if val_size > 0:
        train_ds, val_ds = random_split(train_ds, [train_size, val_size])
    else:
        train_ds, val_ds = train_ds, None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dataset_cfg.get('num_workers', 0)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=dataset_cfg.get('num_workers', 0)
    ) if val_ds is not None else None

    # 모델
    unet_config = {
        'in_channels': model_cfg.get('in_channels', 3),
        'start_out_channels': model_cfg.get('start_out_channels', 32),
        'num_class': model_cfg.get('num_class', 1),
        'size': model_cfg.get('size', 4),
        'padding': model_cfg.get('padding', 1)
    }

    model_name = model_cfg.get('name', 'VanillaUNet')
    if model_name == 'VanillaUNet':
        model = VanillaUNet(**unet_config).to(device)
    elif model_name == 'UNetKalmanCombined':
        from combined_model.unet_kalman_combined import UNetKalmanCombined

        kalman_cfg = cfg.get('kalman', {})
        x0 = np.array(kalman_cfg.get('x0', [0., 0., 0., 0.]), dtype=np.float32)
        kalman_config = {
            'dt': kalman_cfg.get('dt', 1.0),
            'x0': x0,
            'Q': np.eye(4, dtype=np.float32) * kalman_cfg.get('process_noise', 0.01),
            'R': np.eye(2, dtype=np.float32) * kalman_cfg.get('measurement_noise', 0.5),
            'P': np.eye(4, dtype=np.float32) * kalman_cfg.get('initial_covariance', 100.0)
        }
        model = UNetKalmanCombined(
            unet_config,
            kalman_config=kalman_config,
            kalman_name=kalman_cfg.get('type', 'linear')
        ).to(device)
    else:
        raise ValueError(f"Unknown model name '{model_name}' in config.yaml")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, 'demo_unet.pth')
    info_path = os.path.join(args.save_dir, 'training_info.json')
    plot_path = os.path.join(args.save_dir, 'training_curve.png')

    # 학습 루프 with validation and best-model saving
    print('Starting training...')
    print(f'  Epochs: {args.epochs}, Batch size: {batch_size}, LR: {lr}')
    print(f'  Train samples: {train_size}, Val samples: {val_size}')
    print(f'  Save path: {save_path}')
    print('-' * 60)

    best_val_loss = float('inf')
    best_epoch = -1
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        # tqdm 프로그레스 바
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs} [Train]',
                    leave=False, ncols=100)
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            if model_cfg.get('name', 'VanillaUNet') == 'UNetKalmanCombined':
                preds = model(xb, use_kalman=False, return_logits=True)
            else:
                preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_train_loss = running_loss / train_size
        history['train_loss'].append(epoch_train_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # validation
        epoch_val_loss = None
        if val_loader is not None:
            model.eval()
            val_running = 0.0
            with torch.no_grad():
                for xb, yb in tqdm(val_loader, desc=f'Epoch {epoch}/{args.epochs} [Val]',
                                    leave=False, ncols=100):
                    xb = xb.to(device)
                    yb = yb.to(device)
                    if model_cfg.get('name', 'VanillaUNet') == 'UNetKalmanCombined':
                        preds = model(xb, use_kalman=False, return_logits=True)
                    else:
                        preds = model(xb)
                    loss = criterion(preds, yb)
                    val_running += loss.item() * xb.size(0)

            epoch_val_loss = val_running / val_size
            history['val_loss'].append(epoch_val_loss)

            elapsed = time.time() - epoch_start
            total_elapsed = time.time() - start_time
            eta = total_elapsed / epoch * (args.epochs - epoch)

            print(f'Epoch {epoch}/{args.epochs} - '
                  f'train_loss: {epoch_train_loss:.6f} - '
                  f'val_loss: {epoch_val_loss:.6f} - '
                  f'time: {elapsed:.1f}s - '
                  f'ETA: {eta/60:.1f}min')

            # save best
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'val_loss': best_val_loss
                }, save_path)
                print(f'  ★ New best model saved (val_loss={best_val_loss:.6f})')
        else:
            elapsed = time.time() - epoch_start
            print(f'Epoch {epoch}/{args.epochs} - '
                  f'train_loss: {epoch_train_loss:.6f} - '
                  f'time: {elapsed:.1f}s')

        # === 실시간 학습 커브 시각화 (매 epoch 업데이트) ===
        _plot_training_curve(history, plot_path, best_epoch)

    # 최종 정보 저장
    total_time = time.time() - start_time
    info = {
        'save_path': save_path,
        'best_epoch': best_epoch,
        'best_val_loss': None if best_epoch == -1 else best_val_loss,
        'epochs': args.epochs,
        'total_time_sec': total_time,
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'lr': history['lr'],
    }

    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)

    if best_epoch != -1:
        print(f'\n{"="*60}')
        print(f'Training finished in {total_time/60:.1f} minutes')
        print(f'Best checkpoint: epoch {best_epoch}, val_loss={best_val_loss:.6f}')
        print(f'Saved to: {save_path}')
        print(f'Training curve: {plot_path}')
        print(f'{"="*60}')
    else:
        print('Training finished. No validation performed; final checkpoint not saved as best.')


def _plot_training_curve(history, save_path, best_epoch=-1):
    """매 epoch마다 학습 커브를 PNG로 저장 (실시간 모니터링용)"""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    epochs = range(1, len(history['train_loss']) + 1)

    # Train/Val Loss
    ax1.plot(epochs, history['train_loss'], 'b-o', markersize=4, label='Train Loss')
    if history['val_loss']:
        ax1.plot(epochs, history['val_loss'], 'r-o', markersize=4, label='Val Loss')
        if best_epoch > 0:
            ax1.axvline(best_epoch, color='green', linestyle='--', alpha=0.7,
                       label=f'Best (epoch {best_epoch})')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (BCE)')
    ax1.set_title('Training Progress')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 현재 상태 텍스트
    current_epoch = len(history['train_loss'])
    info_text = f"Epoch: {current_epoch}"
    if history['train_loss']:
        info_text += f" | Train: {history['train_loss'][-1]:.6f}"
    if history['val_loss']:
        info_text += f" | Val: {history['val_loss'][-1]:.6f}"
    if best_epoch > 0 and history['val_loss']:
        info_text += f" | Best: {min(history['val_loss']):.6f} (ep{best_epoch})"

    ax1.set_title(f'Training Progress — {info_text}')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


if __name__ == '__main__':
    main()

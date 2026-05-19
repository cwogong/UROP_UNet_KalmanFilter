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

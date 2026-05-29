"""
Evaluate trained checkpoint on test dataset.

Usage examples:
    python experiments/eval_checkpoint.py --config config.yaml --checkpoint checkpoints/demo_unet.pth --batch-size 8

Outputs: prints mean BCE, IoU, Dice and saves results to experiments/eval_results.json
"""

import argparse
import json
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torchvision.transforms.functional as TF
from dataset.uav_dataset import UAVTrackingDataset
from model.Vanilla_UNet import VanillaUNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-samples', type=int, default=None)
    return parser.parse_args()


class EvalPairDataset(torch.utils.data.Dataset):
    """Wrap UAVTrackingDataset to return (image_tensor, mask_tensor_float)"""
    def __init__(self, base_dataset, image_size=(480, 480)):
        self.base = base_dataset
        self.image_size = image_size

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, targets = self.base[idx]
        # image: already transformed by UAVTrackingDataset transforms if provided
        mask = targets['mask'].float().unsqueeze(0) / 255.0
        # ensure mask resized if image was resized as tensor
        if isinstance(image, torch.Tensor):
            # try to ensure mask same H,W as image
            c,h,w = image.shape[-3:]
            if (h,w) != mask.shape[-2:]:
                mask = TF.resize(mask, (h,w), interpolation=TF.InterpolationMode.NEAREST)
        return image, mask


def iou_and_dice(pred_mask, true_mask, eps=1e-6):
    # pred_mask, true_mask: binary tensors (B,1,H,W)
    inter = (pred_mask & true_mask).float().sum(dim=[1,2,3])
    union = (pred_mask | true_mask).float().sum(dim=[1,2,3])
    sum_ = (pred_mask.float().sum(dim=[1,2,3]) + true_mask.float().sum(dim=[1,2,3]))
    iou = (inter + eps) / (union + eps)
    dice = (2 * inter + eps) / (sum_ + eps)
    return iou.cpu().numpy(), dice.cpu().numpy()


def load_model_from_checkpoint(checkpoint_path, model_name, unet_config, device):
    ck = torch.load(checkpoint_path, map_location=device)
    # instantiate model
    if model_name == 'VanillaUNet':
        model = VanillaUNet(**unet_config)
        state = ck.get('model_state', ck)
        model.load_state_dict(state)
    else:
        raise NotImplementedError('Only VanillaUNet checkpoint loading implemented in eval script')
    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    dataset_cfg = cfg.get('dataset', {})
    model_cfg = cfg.get('model', {})

    device = torch.device(args.device)

    # dataset
    root = dataset_cfg.get('root')
    root_mask = dataset_cfg.get('root_mask')
    image_size = tuple(dataset_cfg.get('image_size', [480,480]))
    batch_size = args.batch_size

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    test_sequences = dataset_cfg.get('test_sequences', None)
    base_ds = UAVTrackingDataset(root, root_mask, transforms=transform, num_sequences=test_sequences)

    if args.max_samples is not None and args.max_samples < len(base_ds):
        base_ds = Subset(base_ds, list(range(args.max_samples)))

    ds = EvalPairDataset(base_ds, image_size=image_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # model
    model_name = model_cfg.get('name', 'VanillaUNet')
    unet_config = {
        'in_channels': model_cfg.get('in_channels', 3),
        'start_out_channels': model_cfg.get('start_out_channels', 32),
        'num_class': model_cfg.get('num_class', 1),
        'size': model_cfg.get('size', 4),
        'padding': model_cfg.get('padding', 1)
    }

    model = load_model_from_checkpoint(args.checkpoint, model_name, unet_config, device)

    criterion = nn.BCEWithLogitsLoss(reduction='sum')

    total_loss = 0.0
    total_pixels = 0
    iou_list = []
    dice_list = []
    samples = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += float(loss.item())
            total_pixels += xb.size(0) * xb.size(2) * xb.size(3)

            probs = torch.sigmoid(logits)
            preds_bin = (probs > 0.5)
            true_bin = (yb > 0.5)

            iou, dice = iou_and_dice(preds_bin, true_bin)
            iou_list.extend(iou.tolist())
            dice_list.extend(dice.tolist())

            samples += xb.size(0)

    mean_bce = total_loss / total_pixels
    mean_iou = float(np.mean(iou_list)) if iou_list else 0.0
    mean_dice = float(np.mean(dice_list)) if dice_list else 0.0

    results = {
        'checkpoint': args.checkpoint,
        'num_samples': samples,
        'mean_bce_per_pixel': mean_bce,
        'mean_iou': mean_iou,
        'mean_dice': mean_dice
    }

    out_path = Path('experiments') / 'eval_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print('\n=== Evaluation Summary ===')
    print(json.dumps(results, indent=2))
    print(f"Saved results to: {out_path}")


if __name__ == '__main__':
    main()

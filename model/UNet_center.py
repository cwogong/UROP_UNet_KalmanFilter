import torch
import torch.nn as nn
import numpy as np
from model.Vanilla_UNet import VanillaUNet


class UNetCenter(nn.Module):
	"""
	UNet 기반 마스크 생성기 + 중심점 추출기

	forward(x) -> dict {
		'logits': Tensor(B,1,H,W),
		'probs': Tensor(B,1,H,W),
		'masks': Tensor(B,1,H,W),
		'centers': list of (Tensor(2,) or None)
	}
	"""

	def __init__(self, unet_config):
		super().__init__()
		self.unet = VanillaUNet(
			in_channels=unet_config.get('in_channels', 3),
			start_out_channels=unet_config.get('start_out_channels', 32),
			num_class=unet_config.get('num_class', 1),
			size=unet_config.get('size', 4),
			padding=unet_config.get('padding', 0)
		)

	def forward(self, x, threshold=0.5):
		# x: (B, C, H, W)
		logits = self.unet(x)
		probs = torch.sigmoid(logits)

		B, C, H, W = probs.shape
		masks = (probs > threshold).float()

		centers = []
		# create coordinate grid once on device
		device = probs.device
		xs = torch.arange(0, W, device=device, dtype=torch.float32)
		ys = torch.arange(0, H, device=device, dtype=torch.float32)
		grid_x = xs.view(1, 1, 1, W).expand(B, 1, H, W)
		grid_y = ys.view(1, 1, H, 1).expand(B, 1, H, W)

		for b in range(B):
			m = masks[b, 0]
			s = m.sum()
			if s.item() == 0:
				centers.append(None)
			else:
				cx = (m * grid_x[b, 0]).sum() / s
				cy = (m * grid_y[b, 0]).sum() / s
				centers.append(torch.tensor([cx, cy], device=device))

		return {
			'logits': logits,
			'probs': probs,
			'masks': masks,
			'centers': centers
		}


if __name__ == '__main__':
	# 간단한 동작 확인
	x = torch.randn(2, 3, 128, 128)
	unet_config = {'in_channels': 3, 'start_out_channels': 16, 'num_class': 1, 'size': 3, 'padding': 1}
	model = UNetCenter(unet_config)
	out = model(x)
	print('logits', out['logits'].shape)
	print('masks', out['masks'].shape)
	print('centers', [None if c is None else c.tolist() for c in out['centers']])

import torch
import matplotlib.pyplot as plt
import matplotlib
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

matplotlib.use('Agg')  # 无显示器环境下保存图片

# ── 两套 transform，对比用 ──────────────────
no_aug = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

with_aug = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=1.0),   # 强制翻转，方便对比
    transforms.RandomRotation(30),             # 随机旋转 ±30度
    transforms.ColorJitter(
        brightness=0.5,
        contrast=0.5,
        saturation=0.3
    ),
    transforms.ToTensor(),
])

# ── 加载原始图片（不经过 transform）──────────
raw_dataset = datasets.ImageFolder('dataset/train')

# 取4张图，分别展示原图和增强后的效果
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
fig.suptitle('上排：原图    下排：数据增强后', fontsize=13)

for col in range(4):
    raw_img, label = raw_dataset[col * 5]  # 每隔5张取一张

    # 上排：原图
    axes[0][col].imshow(raw_img)
    axes[0][col].set_title(raw_dataset.classes[label])
    axes[0][col].axis('off')

    # 下排：增强后
    aug_img = with_aug(raw_img)
    # tensor 转回可显示的格式：[C,H,W] -> [H,W,C]
    aug_np = aug_img.permute(1, 2, 0).numpy()
    aug_np = np.clip(aug_np, 0, 1)
    axes[1][col].imshow(aug_np)
    axes[1][col].set_title('augmented')
    axes[1][col].axis('off')

plt.tight_layout()
plt.savefig('augmentation_preview.png', dpi=100)
print("已保存 augmentation_preview.png")
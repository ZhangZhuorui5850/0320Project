import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import os

# ── 1. 自定义 Dataset ───────────────────────
class DefectDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.samples = []
        self.transform = transform
        for label_name in ['defect', 'normal']:
            label = 0 if label_name == 'defect' else 1
            folder = os.path.join(img_dir, label_name)
            for fname in sorted(os.listdir(folder)):
                if fname.endswith('.jpg'):
                    self.samples.append((os.path.join(folder, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# ── 2. 预处理（训练集做增强，验证集不做）──────
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── 3. 加载数据集 ───────────────────────────
train_dataset = DefectDataset('dataset/train', transform=train_transform)
val_dataset   = DefectDataset('dataset/val',   transform=val_transform)

# ── 4. 处理类别不平衡：WeightedRandomSampler ─
labels = [s[1] for s in train_dataset.samples]
class_counts = [labels.count(0), labels.count(1)]  # [defect数, normal数]
print(f"训练集类别分布：defect={class_counts[0]}  normal={class_counts[1]}")

# 类别越少权重越大
class_weights = [1.0 / c for c in class_counts]
sample_weights = [class_weights[l] for l in labels]
sample_weights = torch.tensor(sample_weights)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# 用 sampler 就不能用 shuffle=True
train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False)

# ── 5. 验证 sampler 效果 ─────────────────────
# 取3个batch，看类别是否均衡
defect_count = 0
normal_count = 0
for i, (imgs, labels_batch) in enumerate(train_loader):
    if i >= 3:
        break
    defect_count += (labels_batch == 0).sum().item()
    normal_count += (labels_batch == 1).sum().item()

print(f"\n前3个batch的类别分布（sampler生效后）：")
print(f"  defect: {defect_count}")
print(f"  normal: {normal_count}")
print(f"  比例接近 1:1 说明 sampler 生效了")

print(f"\ntrain_loader batch数：{len(train_loader)}")
print(f"val_loader   batch数：{len(val_loader)}")
print("\nDay 2 完成！")







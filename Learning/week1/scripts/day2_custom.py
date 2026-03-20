import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class DefectDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        # 收集所有图片路径和对应标签
        self.samples = []
        self.transform = transform

        for label_name in ['defect', 'normal']:
            label = 0 if label_name == 'defect' else 1
            folder = os.path.join(img_dir, label_name)
            for fname in os.listdir(folder):
                if fname.endswith('.jpg'):
                    self.samples.append((
                        os.path.join(folder, fname),
                        label
                    ))

    def __len__(self):
        # 返回数据集总大小，DataLoader 靠这个知道要循环多少次
        return len(self.samples)

    def __getitem__(self, idx):
        # 根据下标取出一张图和对应标签
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# ── 使用自定义 Dataset ──────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = DefectDataset('dataset/train', transform=transform)
loader  = DataLoader(dataset, batch_size=16, shuffle=True)

print("数据集大小：", len(dataset))

# 验证第一张
img, label = dataset[0]
print("第一张图形状：", img.shape)
print("第一张图标签：", label)

# 统计类别分布
labels_all = [s[1] for s in dataset.samples]
print("defect 数量：", labels_all.count(0))
print("normal 数量：", labels_all.count(1))
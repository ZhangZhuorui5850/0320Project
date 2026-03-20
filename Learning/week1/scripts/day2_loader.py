import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --1. 定义预处理-------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
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

# ----2. 加载数据集----------------
train_dataset = datasets.ImageFolder('dataset/train', transform=train_transform)
val_dataset = datasets.ImageFolder('dataset/val', transform=val_transform)

# ---3.看看加载了什么————————————
print("类别：", train_dataset.classes)
print("类别对应数字：", train_dataset.class_to_idx)
print("训练集大小：", len(train_dataset))
print("验证集大小：", len(val_dataset))


# ── 4. 取一张图看看 ────────────────────────
img, label = train_dataset[0]
print("\n第一张图：")
print("  形状：", img.shape)
print("  标签：", label, "->", train_dataset.classes[label])
print("  像素值范围：", img.min().item(), "~", img.max().item())

# ----5.dataLoader-----------------
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dloader = DataLoader(val_dataset, batch_size=16, shuffle=False)


#  取一个batch看看
img, label = next(iter(train_loader))
print("\n一个 batch：")
print("  imgs 形状：", img.shape)
print("  labels：", label)






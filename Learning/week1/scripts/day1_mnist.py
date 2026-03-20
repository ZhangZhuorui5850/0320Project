import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ----数据准备 -----------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))
])

train_data = datasets.MNIST(
    root = '../data',
    train = True,
    download = True,
    transform  = transform
)

val_data = datasets.MNIST(
    root = '../data',
    train = False,
    download = True,
    transform = transform
)

train_loader = DataLoader(train_data, batch_size = 64, shuffle = True)
val_loader = DataLoader(val_data, batch_size = 64, shuffle = False)

print(f"训练集大小： {len(train_data)}")
print(f"验证集大小： {len(val_data)}")

# --模型----------------------------
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64, 10)
).cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# 训练————————————————————————————————
for epoch in range(5):
    #训练阶段
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.cuda(), labels.cuda()

        output = model(imgs)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # 验证阶段
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.cuda(), labels.cuda()
            output  = model(imgs)
            pred = output.argmax(dim=1)
            correct += (pred == labels).sum().item()
    
    acc = correct / len(val_data)
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1} /5 Loss: {avg_loss:.4f} Val Acc: {acc:.4f}")

print("\n训练完成！")


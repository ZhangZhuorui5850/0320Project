import os
from PIL import Image
import random

# 造假图片，模拟质检场景：正常品 vs 缺陷品
folders = [
    'dataset/train/normal',
    'dataset/train/defect',
    'dataset/val/normal',
    'dataset/val/defect',
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

# 训练集：normal 80张，defect 20张（故意不平衡）
for i in range(80):
    img = Image.new('RGB', (224, 224), color=(
        random.randint(200, 255),
        random.randint(200, 255),
        random.randint(200, 255)
    ))
    img.save(f'dataset/train/normal/{i:03d}.jpg')

for i in range(20):
    img = Image.new('RGB', (224, 224), color=(
        random.randint(0, 80),
        random.randint(0, 80),
        random.randint(0, 80)
    ))
    img.save(f'dataset/train/defect/{i:03d}.jpg')

# 验证集各10张
for i in range(10):
    img = Image.new('RGB', (224, 224), color=(
        random.randint(200, 255),
        random.randint(200, 255),
        random.randint(200, 255)
    ))
    img.save(f'dataset/val/normal/{i:03d}.jpg')

for i in range(10):
    img = Image.new('RGB', (224, 224), color=(
        random.randint(0, 80),
        random.randint(0, 80),
        random.randint(0, 80)
    ))
    img.save(f'dataset/val/defect/{i:03d}.jpg')

print("数据集创建完成")
print("train/normal: 80张  train/defect: 20张")
print("val/normal:   10张  val/defect:   10张")
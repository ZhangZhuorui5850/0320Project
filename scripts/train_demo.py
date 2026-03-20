from ultralytics import YOLO

# 加载预训练模型
model = YOLO("yolo11n.pt")

# 开始训练
results = model.train(
    data="coco128.yaml",
    epochs=3,
    imgsz=640,
    batch=8,
    project="../outputs",
    name="demo_run",
)

print("训练完成！")
print(f"结果保存在: {results.save_dir}")
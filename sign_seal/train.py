from ultralytics import YOLO

# 加载模型（如 yolov8n.pt 或自定义预训练权重）
model = YOLO("yolo11l.pt")

# 训练
model.train(
    data="sign_seal/data.yaml",  # 数据集配置
    epochs=100,                      # 训练轮数
    imgsz=640,                       # 输入图片大小
    batch=16,                        # 批次大小
    workers=0                        # 数据加载线程
)

# 验证
metrics = model.val()
print(metrics)


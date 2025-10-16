import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# onnx onnxsim onnxruntime onnxruntime-gpu

# 导出参数官方详解链接：https://docs.ultralytics.com/modes/export/#usage-examples


if __name__ == '__main__':
    model = YOLO('runs/train/sdu/weights/best.pt')  # 加载你的模型

    # 运行验证，评估模型在指定验证集上的性能
    metrics = model.val(
        data='your_dataset.yaml',  # 指定数据集配置文件
        split='val',               # 使用验证集（默认就是 val）
        imgsz=1280,                # 输入图像尺寸
        conf=0.01,                 # 置信度阈值
        iou=0.7,                   # IoU 阈值
        half=False,                # 是否使用 FP16
        device=0                   # 使用 GPU 0
    )

    # 输出指标
    print(metrics)
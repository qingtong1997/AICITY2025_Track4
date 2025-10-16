# from ultralytics import YOLO

# if __name__ == '__main__':
#     model = YOLO('runs/train/sdu/weights/best.pt')  # 加载你的模型

#     # 运行验证，评估模型在指定验证集上的性能
#     metrics = model.val(
#         data='../../dataset/visdrone_fisheye8knew.yaml',  # 指定数据集配置文件
#         split='val',               # 使用验证集（默认就是 val）
#         imgsz=1280,                # 输入图像尺寸
#         conf=0.01,                 # 置信度阈值
#         iou=0.7,                   # IoU 阈值
#         half=False,                # 是否使用 FP16
#         device=0                   # 使用 GPU 0
#     )

#     # 输出指标
#     print(metrics)

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/yolo11n+huamn/weights/best.pt')  # 加载模型

    # 执行模型验证
    metrics = model.val(
        data='../../dataset/fisheye8k_human.yaml',  # 指定数据集配置文件
        imgsz=1280,
        conf=0.01,
    )

    # 提取 precision 和 recall
    precision = metrics.results_dict['metrics/precision(B)']
    recall = metrics.results_dict['metrics/recall(B)']

    # 计算 F1 分数
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    # 输出所有指标
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"mAP@0.5: {metrics.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"mAP@0.5:0.95: {metrics.results_dict['metrics/mAP50-95(B)']:.4f}")
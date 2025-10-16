import os
import subprocess
import time
import argparse
import cv2
import json
import numpy as np
from ultralytics import YOLO

from utils import get_model, preprocess_image, postprocess_result
# from utils import f1_score, changeId  # 如有需要可取消注释

def run_all_yolo_models(base_dir):
    """
    自动运行多个YOLO模型推理和格式转换。
    base_dir 是 infer 文件夹的路径，例如 AICITY2025_Track4-main/infer
    """
    print("\n[INFO] Starting YOLO model inference and conversion...\n")

    scripts = [
        {
            "name": "YOLOR",
            "cwd": os.path.join(base_dir, "YOLOR"),
            "commands": [
                "conda activate codetr",
                "python detect.py --source ../../dataset/fisheye_test/images "
                "--weights ../../checkpoints/yolor_w6_best_checkpoint.pt "
                "--conf 0.01 --iou 0.65 --img-size 1280 --device 0 --save-txt --save-conf",
                "python ../../dataprocessing/format_conversion/yolo2coco.py "
                "--images_dir ../../dataset/fisheye_test/images "
                "--labels_dir ./runs/detect/exp/labels "
                "--output ./yolor_w6.json --conf 1 --submission 1 --is_fisheye8k 1"
            ]
        },
        {
            "name": "YOLOv11",
            "cwd": os.path.join(base_dir, "yolo11"),
            "commands": [
                "python detect_v11.py",
                "python ../../dataprocessing/format_conversion/yolo2coco.py "
                "--images_dir ../../dataset/fisheye_test/images "
                "--labels_dir runs/detect/11/labels "
                "--output ./yolov11.json --conf 1 --submission 1 --is_fisheye8k 1"
            ]
        },
        {
            "name": "YOLOv112",
            "cwd": os.path.join(base_dir, "yolo11"),
            "commands": [
                "python detect112.py",
                "python ../../dataprocessing/format_conversion/yolo2coco.py "
                "--images_dir ../../dataset/fisheye_test/images "
                "--labels_dir runs/detect/112/labels "
                "--output ./yolov112.json --conf 1 --submission 1 --is_fisheye8k 1"
            ]
        },
        {
            "name": "YOLOv9",
            "cwd": os.path.join(base_dir, "YoloV9"),
            "commands": [
                "conda activate yolov9",
                "python detect_dual.py --source '../../dataset/fisheye_test/images' "
                "--img 1280 --device 0 --weights '../../checkpoints/yolov9_e_best_checkpoint.pt' "
                "--name yolov9_e --iou 0.75 --conf 0.01 --save-txt --save-conf",
                "python ../../dataprocessing/format_conversion/yolo2coco.py "
                "--images_dir ../../dataset/fisheye_test/images "
                "--labels_dir ./runs/detect/yolov9_e/labels "
                "--output ./yolov9.json --conf 1 --submission 1 --is_fisheye8k 1"
            ]
        },
        {
            "name": "YOLOv8d",
            "cwd": os.path.join(base_dir, "yolo8"),
            "commands": [
                "python detectv8.py",
                "python ../../dataprocessing/format_conversion/yolo2coco.py "
                "--images_dir ../../dataset/fisheye_test/images "
                "--labels_dir ./runs/detect/8d/labels "
                "--output ./8d.json --conf 1 --submission 1 --is_fisheye8k 1"
            ]
        }
    ]

    for script in scripts:
        print(f"[INFO] Running: {script['name']}")
        for cmd in script["commands"]:
            print(f"→ {cmd}")
            subprocess.run(cmd, cwd=script["cwd"], shell=True, check=True)
        print(f"[INFO] Done: {script['name']}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, default='/data/Fisheye1K_eval/images', help='Path to image folder')
    parser.add_argument('--model_path', type=str, default='models/yolo11n_fisheye8k.pt', help='Path to the model')
    parser.add_argument('--max_fps', type=float, default=25.0, help='Maximum FPS for evaluation')
    parser.add_argument('--output_json', type=str, default='/data/Fisheye1K_eval/predictions.json', help='Output JSON file for predictions')
    parser.add_argument('--run_all_models', action='store_true', help='Run all YOLO models before evaluation')
    args = parser.parse_args()

    if args.run_all_models:
        infer_dir = os.path.dirname(os.path.abspath(__file__))
        run_all_yolo_models(infer_dir)

    image_folder = args.image_folder
    model_path = args.model_path

    model = get_model(model_path)

    image_files = sorted([
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    print(f"Found {len(image_files)} images.")

    predictions = []
    print('Prediction started')
    total_time = 0
    start_time = time.time()

    for image_path in image_files:
        img = cv2.imread(image_path)
        if img is None:
            continue
        t0 = time.time()
        img = preprocess_image(img)
        results = model(img, verbose=False)
        results = postprocess_result(results)
        predictions.append((image_path, results))
        t1 = time.time()
        total_time += (t1 - t0)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processed {len(image_files)} images in {elapsed_time:.2f} seconds.")
    print(f"Avg Processing Time: {total_time / len(image_files) * 1000:.2f} ms")

    predictions_json = []
    # 你可以在这里将 predictions 转为 COCO 格式或你自定义格式
    with open(args.output_json, 'w') as f:
        json.dump(predictions_json, f, indent=2)

    fps = len(image_files) / total_time
    normfps = min(fps, args.max_fps) / args.max_fps

    print(f"\n--- Evaluation Complete ---")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"FPS: {fps:.2f}")
    print(f"Normalized FPS: {normfps:.4f}")
    # print(f"F1-score: {f1:.4f}")
    # print(f"Metric (harmonic mean of F1 and FPS): {harmonic_mean:.4f}")


if __name__ == "__main__":
    main()
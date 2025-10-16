
CONDA_PATH=$(which conda)
if [[ -z "$CONDA_PATH" ]]; then
    echo "[❌ ERROR] conda 未安装或未加入 PATH"
    exit 1
fi

# 获取初始化脚本路径
CONDA_BASE=$(dirname $(dirname $CONDA_PATH))
eval "$($CONDA_PATH shell.bash hook)"

set -e  # 一旦出错就退出

echo "[INFO] Starting multi-model inference and conversion..."

conda activate codetr
echo "[INFO] Running YOLOR..."
cd YoloR

python detect.py --source ../../dataset/fisheye_test/images \
    --weights ../../checkpoints/yolor_w6_best_checkpoint.pt \
    --conf 0.01 --iou 0.65 --img-size 1280 --device 0 --save-txt --save-conf

python ../../dataprocessing/format_conversion/yolo2coco.py \
    --images_dir ../../dataset/fisheye_test/images \
    --labels_dir ./runs/detect/exp/labels \
    --output ./yolor_w6.json --conf 1 --submission 1 --is_fisheye8k 1

rm -rf ./runs/detect/exp
cd ..

# === YOLOv11 ===
echo "[INFO] Running YOLOv11..."
cd yolo11
# 如果需要 conda activate 可在此添加
python detect_v11.py

python ../../dataprocessing/format_conversion/yolo2coco.py \
    --images_dir ../../dataset/fisheye_test/images \
    --labels_dir runs/detect/11/labels \
    --output ./yolov11.json --conf 1 --submission 1 --is_fisheye8k 1


rm -rf ./runs/detect/11

# === YOLOv112 ===
echo "[INFO] Running YOLOv112..."
python detect112.py

python ../../dataprocessing/format_conversion/yolo2coco.py \
    --images_dir ../../dataset/fisheye_test/images \
    --labels_dir runs/detect/112/labels \
    --output ./yolov112.json --conf 1 --submission 1 --is_fisheye8k 1

rm -rf ./runs/detect/112
cd ..

=== YOLOv9 ===
echo "[INFO] Running YOLOv9..."
cd YoloV9
conda activate yolor
python detect_dual.py --source '../../dataset/fisheye_test/images' \
    --img 1280 --device 0 --weights '../../checkpoints/yolov9_e_best_checkpoint.pt' \
    --name yolov9_e --iou 0.75 --conf 0.01 --save-txt --save-conf

python ../../dataprocessing/format_conversion/yolo2coco.py \
    --images_dir ../../dataset/fisheye_test/images \
    --labels_dir ./runs/detect/yolov9_e/labels \
    --output ./yolov9.json --conf 1 --submission 1 --is_fisheye8k 1

rm -rf ./runs/detect/yolov9_e
cd ..
conda deactivate
conda activate codetr 
# === YOLOv8 ===
echo "[INFO] Running YOLOv8..."
cd yolo8
# 如果需要 conda activate 可在此添加
python detectv8.py

python ../../dataprocessing/format_conversion/yolo2coco.py \
    --images_dir ../../dataset/fisheye_test/images \
    --labels_dir ./runs/detect/8d/labels \
    --output ./8d.json --conf 1 --submission 1 --is_fisheye8k 1
rm -rf ./runs/detect/8d
cd ..

echo "[✅] All models done successfully." 


conda activate codetr2
python fuse_results2.py
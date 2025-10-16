# import os
# import json
# from tqdm import tqdm
# from PIL import Image

# # ==== 配置区域 ====
# json_files = [
#     './CO-DETR/work_dirs/infer_pseudo.bbox.json',
#     './CO-DETR/work_dirs/infer_all.bbox.json',
#     './CO-DETR/work_dirs/infer_fold0.bbox.json',
#     './YoloR/yolor_w6.json',
#     './YoloV9/yolov9.json',
#     './CO-DETR/work_dirs/infer_syn_vis_fis.bbox.json',
#     'yolov112.json',
#     'yolov11sdu.json'
# ]

# # ✅ 路径配置
# image_dir = '/home/anonymous/AICITY2024_Track4/dataset/fisheye_test/images'
# output_label_dir = '/home/anonymous/AICITY2024_Track4/dataset/fisheye_test/labels'
# output_train_json = '/home/anonymous/AICITY2024_Track4/dataset/fisheye_test/train.json'
# coco_annotation_path = '/home/anonymous/AICITY2024_Track4/dataset/json_labels/vis_fish_all.json'
import json
import os
import cv2
from collections import defaultdict

def coco_submission_to_yolo(json_path, images_dir, output_dir, use_conf=True, conf_threshold=None):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        annotations = json.load(f)

    from collections import defaultdict
    annotations_per_image = defaultdict(list)
    for ann in annotations:
        annotations_per_image[ann['image_id']].append(ann)

    for image_file in os.listdir(images_dir):
        if not image_file.endswith('.png'):
            continue
        img_path = os.path.join(images_dir, image_file)
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        img_id = get_image_Id(image_file)
        if img_id not in annotations_per_image:
            continue

        label_lines = []
        for ann in annotations_per_image[img_id]:
            # ✅ 过滤低置信度框（无论是否写入 score）
            if conf_threshold is not None and 'score' in ann and ann['score'] < conf_threshold:
                continue

            bbox = ann['bbox']
            x, y, bw, bh = bbox
            cx = x + bw / 2
            cy = y + bh / 2

            cx /= w
            cy /= h
            bw /= w
            bh /= h

            line = f"{ann['category_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            if use_conf and 'score' in ann:
                line += f" {ann['score']:.6f}"
            label_lines.append(line)

        label_filename = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(output_dir, label_filename)
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))

def get_image_Id(img_name):
    img_name = img_name.split('.png')[0]
    sceneList = ['M', 'A', 'E', 'N']
    cameraIndx = int(img_name.split('_')[0].split('camera')[1])
    sceneIndx = sceneList.index(img_name.split('_')[1])
    frameIndx = int(img_name.split('_')[2])
    imageId = int(str(cameraIndx)+str(sceneIndx)+str(frameIndx))
    return imageId


coco_submission_to_yolo(
    json_path="./YoloR/yolor_w6.json",
    images_dir="/home/anonymous/AICITY2024_Track4/dataset/fisheye_test/images",
    output_dir="/home/anonymous/AICITY2024_Track4/dataset/fisheye_test/pseudo_yolo_labels",
    use_conf=False,
    conf_threshold=0.3  # <-- 添加这个

)
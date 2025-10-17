# AICITY2025 Track4

This repository is a modified version of the original codebase for AICITY2025 Track4. It provides instructions to reproduce leaderboard results using a pre-built Docker image, as well as guidance for training YOLOv8 and YOLOv11 models from scratch. For training other models, please refer to the original repositories.

## Quick Start with Docker

To quickly reproduce the leaderboard results, use the pre-built Docker image which includes all necessary weights and environments.

1. Pull the Docker image:
```
docker pull qingtong1997/codetr-image:latest
```
2. Run the Docker container (mount your local directories if needed):
```
docker run -it -v /path/to/local/dir:/AICITY2024_Track4 qingtong1997/codetr-image:latest /bin/bash
```
3. Inside the container, navigate to the `infer` directory:
```
cd infer
```
4. Add execute permissions to the script:
```
chmod +x run_all_models.sh
```
5. Run the script to reproduce the leaderboard results:
```
./run_all_models.sh
```
This will replicate the leaderboard results using the pre-included weights and environment.

## Downloading Weights and Datasets (If Needed)

If the Docker image does not include everything or you need to set up manually:

- Download weights from: [Google Drive](https://drive.google.com/drive/my-drive?dmr=1&ec=wgc-drive-hero-goto).
- Create a `checkpoints` folder and place all weight files inside it.

- Optionally, create a `dataset` directory:
- Download the dataset zip from Google Drive and unzip it into the `dataset` directory.
- Ensure you also download the following datasets: `fisheye8k`, `fisheye_test`, and `visdrone`.

- Example directory structure after setup:

```
AICITY2025_Track4/dataset/
├── fisheye8k/
│   ├── test/
│   └── train/
├── fisheye_test/
└── visdrone/
```

- Copy `image.json` from Google Drive to the `infer` folder.

Note: Parts of the leaderboard results are based on weights from [vnptai/AICITY2024_Track4](https://github.com/vnptai/AICITY2024_Track4).

## Reproducing Leaderboard Results

Follow the Quick Start steps above. The `run_all_models.sh` script will handle inference and reproduce the results.

## Training from Scratch

This repository provides training scripts only for YOLOv8 and YOLOv11. For Co-DETR, YOLOR-W6, and YOLOv9-e, refer to the original repositories.

### Training YOLOv8

1. Navigate to the YOLOv8 directory inside `infer`:

```
cd infer/yolov8
```

2. Modify the paths in `visdrone_fisheye8k_fake_label&fake_E.yaml` to point to your dataset directories.

3. Run the training script:
```

python train.py
```

### Training YOLOv11

1. Navigate to the YOLOv11 directory inside `infer`:
```

cd infer/yolo11
```

2. Modify the paths in `.visdrone_fisheye8knew.yaml` for both `train_v11.py` and `train_v112.py` to point to your dataset directories.

3. Run the training scripts:
```
python train_v11.py or python train_v112.py
```
## References

- Original weights for some models: [vnptai/AICITY2024_Track4](https://github.com/vnptai/AICITY2024_Track4)
- Docker image: `qingtong1997/codetr-image:latest`
- Datasets and additional files: Google Drive link provided above.
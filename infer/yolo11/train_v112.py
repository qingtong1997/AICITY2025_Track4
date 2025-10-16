import warnings, os

warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':

    model = YOLO('ultralytics/cfg/models/11/yolo11-CGRFPN.yaml')
    model.load('yolo11n.pt')
    model.train(data='../../dataset/visdrone_fisheye8knew.yaml',
            cache=False,
            imgsz=1280,
            epochs=300,
            batch=4, 
            close_mosaic=0,
            workers=4, 
            optimizer='SGD',
            project='runs/train',
            name='exp',
            )
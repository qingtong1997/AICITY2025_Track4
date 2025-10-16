import warnings, os

warnings.filterwarnings('ignore')
from ultralytics import YOLO



if __name__ == '__main__':
    
    model = YOLO('ultralytics/cfg/models/11/yolo11n.yaml')
    model.load('yolo11n.pt')

    model.train(data='../../dataset/visdrone_fisheye8knew.yaml',
            cache=False,
            imgsz=1280,
            epochs=300,
            batch=4, 
            close_mosaic=0, #
            workers=4, # 
            # device='0,1', # 
            optimizer='SGD', # using SGD
            project='runs/train',
            name='exp',
            )
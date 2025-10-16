import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('runs/train/exp28/weights/best.pt') 
    model.predict(source='../../dataset/fisheye_test/images',
                imgsz=1280,
                project='runs/detect',
                name='11',
                save=True,
                conf=0.01,

                show_conf=False, # do not show prediction confidence

                save_txt=True, # save results as .txt file

                save_conf=True,
              )

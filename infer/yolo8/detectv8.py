import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('runs/train/8D/weights/best.pt') # select your model.pt path
    model.predict(source='../../AICITY2024_Track4/dataset/fisheye_test/images',
                  imgsz=1280,
                  project='runs/detect',
                  name='8d',
                  save=True,
                  conf=0.01, 
                  # iou=0.7,
                  # agnostic_nms=True,
                  # visualize=True, # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  show_conf= True, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                  save_conf=True,
                )
    

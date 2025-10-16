import warnings, os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    # 代表用cpu训练 不推荐！没意义！ 而且有些模块不能在cpu上跑
# os.environ["CUDA_VISIBLE_DEVICES"]="0"     # 代表用第一张卡进行训练  0：第一张卡 1：第二张卡
# 多卡训练参考<YOLOV11配置文件.md>下方常见错误和解决方案
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# BILIBILI UP 魔傀面具
# 训练参数官方详解链接：https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings

# 训练过程中loss出现nan，可以尝试关闭AMP，就是把下方amp=False的注释去掉。
# 训练时候输出的AMP Check使用的YOLO11n的权重不是代表载入了预训练权重的意思，只是用于测试AMP，正常的不需要理会。

# 使用项目前必看<项目视频百度云链接.txt>的第一行有一个必看的视频!!


# 在20250502更新中，修改保存权重的逻辑，训练结束(注意是正常训练结束后，手动停止的没有)后统一会保存4个模型，
# 分别是best.pt、last.pt、best_fp32.pt、last_fp32.pt，其中不带fp32后缀的是fp16格式保存的，
# 但由于有些模块对fp16非常敏感，会出现后续使用val.py的时候精度为0的情况，这种情况下可以用后缀带fp32去测试。

if __name__ == '__main__':
     # 或者你的自定义路径
    
    model = YOLO('ultralytics/cfg/models/11/yolo11n.yaml')
    model.load('yolo11n.pt')
    #model = YOLO('ultralytics/cfg/models/11/yolo11-C3k2-MutilScaleEdgeInformationSelect.yaml')   这个不知道是多少
    # 
    #  6.14shi1model = YOLO('ultralytics/cfg/models/11/yolo11-CGRFPN.yaml')
    # sdu是yolo11-FDPN.yaml
    #28是original
    #30是yolo11-CGRFPN.yaml
    #可能的11优化方向yolo11-ContextGuideFPN.yaml，yolo11-FDPN-TADDH.yaml 
    # model = YOLO('ultralytics/cfg/models/11/yolo11n.yaml')
    # model.load('yolo11n.pt') # loading pretrain weights
    # model.train(data='dataset/data.yaml',
                
    #             cache=False,
    #             imgsz=640,
    #             epochs=300,
    #             batch=32, 
    #             close_mosaic=0, # 最后多少个epoch关闭mosaic数据增强，设置0代表全程开启mosaic训练
    #             workers=4, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
    #             # device='0,1', # 指定显卡和多卡训练参考<YOLOV11配置文件.md>下方常见错误和解决方案
    #             optimizer='SGD', # using SGD
    #             # patience=0, # set 0 to close earlystop.
    #             # resume=True, # 断点续训,YOLO初始化时选择last.pt
    #             # amp=False, # close amp | loss出现nan可以关闭amp
    #             # fraction=0.2,
    #             project='runs/train',
    #             name='exp',
    #             )
    model.train(data='../../dataset/fisheye8k_fake_label&fake_E&fake_fish.yaml',
            cache=False,
            imgsz=1280,
            epochs=300,
            batch=4, 
            close_mosaic=0, # 最后多少个epoch关闭mosaic数据增强，设置0代表全程开启mosaic训练
            workers=4, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
            # device='0,1', # 指定显卡和多卡训练参考<YOLOV11配置文件.md>下方常见错误和解决方案
            optimizer='SGD', # using SGD

            # lr0=1e-4,  # 默认是1e-2（SGD）或1e-3（Adam），微调建议降低
            # freeze=10,       # 冻结前10层，减少过拟合风险
            # patience=0, # set 0 to close earlystop.
            # resume=True, # 断点续训,YOLO初始化时选择last.pt
            # amp=False, # close amp | loss出现nan可以关闭amp
            # fraction=0.2,
            project='runs/train',
            name='exp',
            )
#!/bin/bash

# 运行测试脚本
echo "开始执行第一个测试命令..."
tools/dist_test.sh projects/CO-DETR/configs/codino/infer_all.py ../../checkpoints/best_vis_fish_all.pth 1

echo "开始执行第二个测试命令..."
tools/dist_test.sh projects/CO-DETR/configs/codino/infer_pseudo.py ../../checkpoints/best_vis_fish_pseudo.pth 1

echo "开始执行第三个测试命令..."
tools/dist_test.sh projects/CO-DETR/configs/codino/infer_syn_vis_fis.py ../../checkpoints/codetr_syn_vis_fish.pth 1

echo "所有测试命令执行完毕！"
import cv2
import os
import sys
from matplotlib.pyplot import annotate
import numpy as np
import time

import torch
import torch.backends.cudnn as cudnn

from models.common import DetectMultiBackend
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox


# cap = cv2.VideoCapture('rtmp://10.42.0.1:1935/rtmp/', cv2.CAP_FFMPEG)
cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 1280
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 720



def drawBox(annotator:Annotator, wh, prediction, names):

    # wh = (416, 736, 3)
    x_ratio = width / wh[1] # 1.739
    y_ratio = height / wh[0] # 1.731

    for *xyxy, conf, cls in reversed(prediction): # type(*xyxy) = list 
        c = int(cls)  # integer class
        label = (f'{names[c]} {conf:.2f}')

        ### resize box size ###
        for i in range(len(xyxy)): # 0, 1, 2, 3
            if(i%2):
                # x
                xyxy[i] = int(xyxy[i])*x_ratio
            else:
                # y
                xyxy[i] = int(xyxy[i])*y_ratio
        annotator.box_label(xyxy, label, color=colors(c, True))

@torch.no_grad()
def main():

    weights = 'yolov5s.pt'
    data = 'data/coco128.yaml'
    conf_thres = 0.25  # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    max_det = 1000  # maximum detections per image
    device = select_device('')

    model = DetectMultiBackend(weights, device=device, data=data)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = [width, height]
    imgsz = check_img_size(imgsz, s=stride)
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup

    FPS = 30
    yolo_fps = 1/3
    cycle_time = int(FPS/yolo_fps)
    count = 0

    last_pred = None
    
    while cap.isOpened():

        _, frame = cap.read()
        
        start_time = time.time()
        

        # frame.shape = (480, 640, 3) (720, 1280, 3)
        img = [letterbox(frame.copy(), imgsz, stride=stride, auto=pt)[0]] # size change (416, 736, 3)
        annotator = Annotator(frame.copy(), line_width=3, example=str(names))
        show_img = img[0].copy() # (384, 480, 3) (416, 736, 3)
        
        
        
        # fps count
        if (count%cycle_time) == 0 :
            # Stack
            img = np.stack(img, 0)
            # Convert
            img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            img = np.ascontiguousarray(img)
            im = torch.from_numpy(img).to(device)
            
            im = im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
        
            pred = model(im, augment=False, visualize=False)
            pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)
            
            if len(pred) != 1:
                continue

            pred = pred[0] # box 
            count = 0

        else:
            pred = last_pred


        drawBox(annotator, show_img.shape, pred, names)
        result = annotator.result() # (720, 1280, 3)
        cv2.imshow("Yolov5", result)

        last_pred = pred
        count += 1

        os.system('clear')
        # print(f"LEngth of prediction : {len(pred)}")
        print(f"Time cost : {time.time()-start_time}")
        print(count)

        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    main()
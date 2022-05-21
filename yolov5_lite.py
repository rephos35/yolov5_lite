import cv2
import os
import sys
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox

# cap = cv2.VideoCapture('rtmp://10.42.0.1:1935/rtmp/', cv2.CAP_FFMPEG)
cap = cv2.VideoCapture(0)


@torch.no_grad()
def main():

    weights = 'yolov5s.pt'
    data = 'data/coco128.yaml'
    conf_thres = 0.25  # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    max_det = 1000  # maximum detections per image
    imgsz = [640, 640]
    device = select_device('')

    model = DetectMultiBackend(weights, device=device, data=data)
    stride, names, pt = model.stride, model.names, model.pt
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup

    while cap.isOpened():
        _, frame = cap.read()

        img0 = frame.copy()

        img = [letterbox(img0, imgsz, stride=stride, auto=pt)[0]]
        # Stack
        img = np.stack(img, 0)

        print(img.shape)
        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        im = torch.from_numpy(img).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # im = im.view(1, 3, 480, 640)

        pred = model(im, augment=False, visualize=False)

        pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)
        print(pred)

        s = ''

        for i, det in enumerate(pred):  # per image

            im0 = frame.copy()

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=3, example=str(names))
            # Stream results

            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = (f'{names[c]} {conf:.2f}')
                annotator.box_label(xyxy, label, color=colors(c, True))

            im0 = annotator.result()
            cv2.imshow("Yolov5", im0)

        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    main()

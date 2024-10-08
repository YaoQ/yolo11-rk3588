import os
import numpy as np
import cv2
from rknnlite.api import RKNNLite
from math import exp
import argparse

# 定义类别名称
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush']

# 定义一些常量
class_num = len(CLASSES)
headNum = 3
strides = [8, 16, 32]
mapSize = [[80, 80], [40, 40], [20, 20]]
input_imgH = 640
input_imgW = 640

# 定义检测框类
class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

# Yolo11推理类
class Yolo11Inference:
    def __init__(self, model_path, conf_thresh=0.5, nms_thresh=0.5):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.meshgrid = []
        self.rknn = RKNNLite()
        self.load_model()
        self.generate_meshgrid()

    def load_model(self):
        """加载RKNN模型"""
        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            print('Load RKNN model failed!')
            exit(ret)

        ret = self.rknn.init_runtime()
        if ret != 0:
            print('Init runtime environment failed!')
            exit(ret)
        print('Model loaded and initialized.')

    def generate_meshgrid(self):
        """生成网格坐标"""
        for index in range(headNum):
            for i in range(mapSize[index][0]):
                for j in range(mapSize[index][1]):
                    self.meshgrid.append(j + 0.5)
                    self.meshgrid.append(i + 0.5)

    def iou(self, box1, box2):
        """计算两个框的IoU"""
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2

        xmin = max(xmin1, xmin2)
        ymin = max(ymin1, ymin2)
        xmax = min(xmax1, xmax2)
        ymax = min(ymax1, ymax2)

        inner_width = max(0, xmax - xmin)
        inner_height = max(0, ymax - ymin)

        inner_area = inner_width * inner_height
        area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
        area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

        total = area1 + area2 - inner_area
        return inner_area / total

    def nms(self, detect_result):
        """非极大值抑制"""
        pred_boxes = []
        sorted_detect_boxes = sorted(detect_result, key=lambda x: x.score, reverse=True)

        for i in range(len(sorted_detect_boxes)):
            xmin1 = sorted_detect_boxes[i].xmin
            ymin1 = sorted_detect_boxes[i].ymin
            xmax1 = sorted_detect_boxes[i].xmax
            ymax1 = sorted_detect_boxes[i].ymax
            classId = sorted_detect_boxes[i].classId

            if sorted_detect_boxes[i].classId != -1:
                pred_boxes.append(sorted_detect_boxes[i])
                for j in range(i + 1, len(sorted_detect_boxes)):
                    if classId == sorted_detect_boxes[j].classId:
                        xmin2 = sorted_detect_boxes[j].xmin
                        ymin2 = sorted_detect_boxes[j].ymin
                        xmax2 = sorted_detect_boxes[j].xmax
                        ymax2 = sorted_detect_boxes[j].ymax
                        iou = self.iou((xmin1, ymin1, xmax1, ymax1), (xmin2, ymin2, xmax2, ymax2))
                        if iou > self.nms_thresh:
                            sorted_detect_boxes[j].classId = -1
        return pred_boxes

    def sigmoid(self, x):
        """sigmoid函数"""
        return 1 / (1 + exp(-x))

    def postprocess(self, outputs, img_h, img_w):
        """后处理函数"""
        print('postprocess ... ')

        detect_result = []
        output = [out.reshape((-1)) for out in outputs]

        scale_h = img_h / input_imgH
        scale_w = img_w / input_imgW

        grid_index = -2
        cls_index = 0
        cls_max = 0

        for index in range(headNum):
            cls = output[index * 2 + 1]
            reg = output[index * 2 + 0]

            for h in range(mapSize[index][0]):
                for w in range(mapSize[index][1]):
                    grid_index += 2

                    if 1 == class_num:
                        cls_max = self.sigmoid(cls[0 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w])
                        cls_index = 0
                    else:
                        for cl in range(class_num):
                            cls_val = cls[cl * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]
                            if 0 == cl:
                                cls_max = cls_val
                                cls_index = cl
                            else:
                                if cls_val > cls_max:
                                    cls_max = cls_val
                                    cls_index = cl
                        cls_max = self.sigmoid(cls_max)

                    if cls_max > self.conf_thresh:
                        regdfl = []
                        for lc in range(4):
                            sfsum = 0
                            locval = 0
                            for df in range(16):
                                temp = exp(reg[((lc * 16) + df) * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w])
                                reg[((lc * 16) + df) * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w] = temp
                                sfsum += temp

                            for df in range(16):
                                sfval = reg[((lc * 16) + df) * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w] / sfsum
                                locval += sfval * df
                            regdfl.append(locval)

                        x1 = (self.meshgrid[grid_index + 0] - regdfl[0]) * strides[index]
                        y1 = (self.meshgrid[grid_index + 1] - regdfl[1]) * strides[index]
                        x2 = (self.meshgrid[grid_index + 0] + regdfl[2]) * strides[index]
                        y2 = (self.meshgrid[grid_index + 1] + regdfl[3]) * strides[index]

                        xmin = x1 * scale_w
                        ymin = y1 * scale_h
                        xmax = x2 * scale_w
                        ymax = y2 * scale_h

                        xmin = xmin if xmin > 0 else 0
                        ymin = ymin if ymin > 0 else 0
                        xmax = xmax if xmax < img_w else img_w
                        ymax = ymax if ymax < img_h else img_h

                        box = DetectBox(cls_index, cls_max, xmin, ymin, xmax, ymax)
                        detect_result.append(box)
        # NMS
        pred_box = self.nms(detect_result)

        return pred_box

    def inference(self, img):
        """推理函数"""
        print('--> Running model')
        outputs = self.rknn.inference(inputs=[img])
        print('done')
        return outputs

    def release(self):
        """释放RKNN资源"""
        self.rknn.release()

if __name__ == '__main__':
    print('This is main ...')
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='./img/bus.jpg', help='path of input')
    parser.add_argument('--model', type=str, default='./weights/yolo11s_sim-640-640_rm_dfl_rk3588.rknn', help='path of bmodel')
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.5, help='nms threshold')
    args = parser.parse_args()

    yolo11_inference = Yolo11Inference(args.model, args.conf_thresh, args.nms_thresh)

    orig_img = cv2.imread(args.input)
    img_h, img_w = orig_img.shape[:2]

    origimg = cv2.resize(orig_img, (input_imgW, input_imgH), interpolation=cv2.INTER_LINEAR)
    origimg = cv2.cvtColor(origimg, cv2.COLOR_BGR2RGB)

    img = np.expand_dims(origimg, 0)

    outputs = yolo11_inference.inference(img)
    predbox = yolo11_inference.postprocess(outputs, img_h, img_w)

    print(len(predbox))

    for i in range(len(predbox)):
        xmin = int(predbox[i].xmin)
        ymin = int(predbox[i].ymin)
        xmax = int(predbox[i].xmax)
        ymax = int(predbox[i].ymax)
        classId = predbox[i].classId
        score = predbox[i].score

        cv2.rectangle(orig_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        ptext = (xmin, ymin)
        title = CLASSES[classId] + ":%.2f" % (score)
        cv2.putText(orig_img, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite('./test_rknn_result.jpg', orig_img)
    yolo11_inference.release()
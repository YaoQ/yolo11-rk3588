import os
import numpy as np
import cv2
from rknnlite.api import RKNNLite
from math import exp
import argparse
import time

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

# 定义常量
class_num = len(CLASSES)
headNum = 3
strides = [8, 16, 32]
mapSize = [[80, 80], [40, 40], [20, 20]]
input_imgH = 640
input_imgW = 640

# 检测框类
class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

# 优化后的Yolo11推理类
class Yolo11Inference:
    def __init__(self, model_path, conf_thresh=0.5, nms_thresh=0.5):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.meshgrid = self.generate_meshgrid()  # 直接生成网格
        self.rknn = RKNNLite()
        self.load_model()

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
        meshgrid = []
        for index in range(headNum):
            for i in range(mapSize[index][0]):
                for j in range(mapSize[index][1]):
                    meshgrid.append((j + 0.5, i + 0.5))
        return np.array(meshgrid)

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
        return inner_area / total if total > 0 else 0

    def nms(self, detect_result):
        """非极大值抑制"""
        pred_boxes = []
        sorted_detect_boxes = sorted(detect_result, key=lambda x: x.score, reverse=True)

        for i in range(len(sorted_detect_boxes)):
            if sorted_detect_boxes[i].classId == -1:
                continue  # 跳过已经处理过的框

            box1 = sorted_detect_boxes[i]
            pred_boxes.append(box1)

            for j in range(i + 1, len(sorted_detect_boxes)):
                if sorted_detect_boxes[j].classId == -1:
                    continue
                if box1.classId == sorted_detect_boxes[j].classId:
                    box2 = sorted_detect_boxes[j]
                    if self.iou((box1.xmin, box1.ymin, box1.xmax, box1.ymax),
                                (box2.xmin, box2.ymin, box2.xmax, box2.ymax)) > self.nms_thresh:
                        sorted_detect_boxes[j].classId = -1  # 标记为已处理
        return pred_boxes

    def sigmoid(self, x):
        """sigmoid函数，使用NumPy向量化"""
        return 1 / (1 + np.exp(-x))

    def postprocess(self, outputs, img_h, img_w):
        """后处理"""
        print('postprocess ...')

        detect_result = []
        output = [out.flatten() for out in outputs]

        scale_h = img_h / input_imgH
        scale_w = img_w / input_imgW

        for index in range(headNum):
            cls = self.sigmoid(output[index * 2 + 1])  # 向量化sigmoid
            reg = np.exp(output[index * 2])  # 向量化exp
            grid = self.meshgrid[:mapSize[index][0] * mapSize[index][1]]

            for idx, (grid_x, grid_y) in enumerate(grid):
                cls_vals = cls[idx * class_num:(idx + 1) * class_num]
                cls_max = np.max(cls_vals)
                cls_index = np.argmax(cls_vals)

                if cls_max > self.conf_thresh:
                    reg_vals = reg[idx * 64:(idx + 1) * 64].reshape(4, 16)
                    regdfl = [np.sum((np.arange(16) * (vals / np.sum(vals)))) for vals in reg_vals]

                    x1 = (grid_x - regdfl[0]) * strides[index] * scale_w
                    y1 = (grid_y - regdfl[1]) * strides[index] * scale_h
                    x2 = (grid_x + regdfl[2]) * strides[index] * scale_w
                    y2 = (grid_y + regdfl[3]) * strides[index] * scale_h

                    xmin = max(0, x1)
                    ymin = max(0, y1)
                    xmax = min(img_w, x2)
                    ymax = min(img_h, y2)

                    detect_result.append(DetectBox(cls_index, cls_max, xmin, ymin, xmax, ymax))

        return self.nms(detect_result)

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
    print('yolo11 rknnlite demo on RK3588 ...')
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='./img/bus.jpg', help='path of input')
    parser.add_argument('--model', type=str, default='./weights/yolo11s_sim-640-640_rm_dfl_rk3588.rknn', help='path of bmodel')
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.5, help='nms threshold')
    args = parser.parse_args()

    yolo11_inference = Yolo11Inference(args.model, args.conf_thresh, args.nms_thresh)

    orig_img = cv2.imread(args.input)
    img_h, img_w = orig_img.shape[:2]

    origimg = cv2.resize(orig_img, (input_imgW, input_imgH), interpolation=cv2.INTER_NEAREST)
    origimg = cv2.cvtColor(origimg, cv2.COLOR_BGR2RGB)

    img = np.expand_dims(origimg, 0)

    # 记录推理开始时间
    start_time = time.time()

    outputs = yolo11_inference.inference(img)
    predbox = yolo11_inference.postprocess(outputs, img_h, img_w)

    # 记录推理结束时间
    end_time = time.time()
    # 计算推理耗时
    inference_time = end_time - start_time
    print(f"Inference time: {inference_time:.4f} seconds")

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

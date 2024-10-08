# 使用RKNNLite 在RK3588上实现yolo11的部署

## 环境准备
### 硬件环境
- RK3588开发板
### 软件环境
- Debian11
- RKNN-ToolKit 1.6.0
- Python 3.9.2

### 模型准备
- [yolo11s pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt)模型文件
- 导出onnx以及参考推理代码, 参考(yolov11s_onnx_rknn)[https://github.com/cqu20160901/yolov11_onnx_rknn]

## 运行

```bash
python3 yolov11s_onnx_rknn.py
```
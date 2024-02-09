# Object Detection YOLOv8

YOLOv8 is a variant of the YOLO (You Only Look Once) object detection model designed to be smaller and faster than its predecessors while maintaining good accuracy in object detection tasks. YOLOv8 is widely used in various computer vision applications, including surveillance, autonomous vehicles, and object tracking, due to its real-time performance and ability to detect multiple objects in a single pass.

[![Made withYOLOv8](https://img.shields.io/badge/Made%20with-YOLOv8-green)](https://docs.ultralytics.com/)

## Assignment

Develop and train a custom AI model that can accurately detect only two categories of objects (sheeps and owls) in an image using the YOLO8 framework.

Dataset: [Open Images v7](https://docs.ultralytics.com/datasets/detect/open-images-v7/)

There are two ways to install and run the code:
1. [Online using Google Colaboratory](https://github.com/mmravak/object-detection-yolov8/blob/main/README.md#installation-and-running-in-colab)
2. [Locally using Visual Studio Code]

## Installation and running in Colab

[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RtPTmGTpYKN7KiABjWfEaLc4l4EeCRxt?usp=sharing)


Install ultralytics library that contains YOLOv8 model using package manager command line:

```
pip install ultralytics
```

Install fiftyone library for downloading and analyzing the dataset:
```
pip install fiftyone
```

Import `fiftyone.zoo` to get the Open Images v7 dataset:
```
import fiftyone as fo
import fiftyone.zoo as foz
```

Import the `YOLO` class that represents an implementation of the YOLO object detection algorithm:
```
from ultralytics import YOLO
```
Load the Open Images v7 dataset:
```
dataset = foz.load_zoo_dataset(name,
                               split = split,
                               label_types = ['detections'],
                               dataset_dir = Path(SETTINGS['datasets_dir']) / 'fiftyone' / name,
                               classes = ['Sheep', 'Owl'],
                               max_samples = 1000)
```
Load the pretrained YOLOv8n model:
```
model = YOLO('yolov8n.pt')
```

Training process:
```
results = model.train(data='/content/datasets/open-images-v7/dataset.yaml', epochs=20, imgsz = 640)
```

Results:
[`/content/runs/detect/train`](https://github.com/mmravak/object-detection-yolov8/tree/main/runs/detect/train)

Here you can see evaluation on some samples of validation set:

![Image](https://github.com/mmravak/object-detection-yolov8/blob/main/runs/detect/train/val_batch0_pred.jpg)

The trained model is saved to file [`trained_yolov8.pt`](https://github.com/mmravak/object-detection-yolov8/blob/main/trained_yolov8n.pt)

## Installation and running locally: Visual Studio Code

[![Open in Visual Studio Code](https://img.shields.io/badge/Open%20in-VSCode-blue)](https://vscode.dev/github/mmravak/object-detection-yolov8)


Install virtual enviroment in Terminal using command line:
```
python -m venv venv
```
To activate venv go to _View_ tab and select _Command Palette_

Search _Python: Select Interpreter_ -> select _Edit interpreter path_

Choose _python.exe_ from _venv/scripts_

Install packages from _requirements.txt_:
```
pip install -r requirements.txt
```
Load the dataset:
```
python data.py
```
Train the YOLOv8 model:
```
python train.py
```

## Documentation

- [Python](https://docs.python.org/3/)
- [YOLOv8](https://docs.ultralytics.com/)
- [FiftyOne](https://docs.voxel51.com/)
- [Dataset YAML](https://docs.ultralytics.com/datasets/detect/open-images-v7/#dataset-yaml)

## License
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


import fiftyone as fo
import fiftyone.zoo as foz

from ultralytics.utils import SETTINGS, Path
import warnings

name = 'open-images-v7'

for split in 'train', 'validation':
    train = split == 'train'

    # Load Open Images dataset
    dataset = foz.load_zoo_dataset(name,
                                   split = split,
                                   label_types = ['detections'],
                                   dataset_dir = Path(SETTINGS['datasets_dir']) / 'fiftyone' / name,
                                   classes = ['Sheep', 'Owl'],
                                   max_samples = 1000)


    # Define classes (Sheep and Owl)
    if train:
        classes = ['Sheep', 'Owl']

    # Export to YOLO format
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="fiftyone.utils.yolo")
        dataset.export(export_dir = str(Path(SETTINGS['datasets_dir']) / name),
                       dataset_type = fo.types.YOLOv5Dataset,
                       label_field = 'ground_truth',
                       split = 'val' if split == 'validation' else split,
                       classes = classes,
                       overwrite = train)
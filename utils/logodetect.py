from unittest import result
import torch
from icevision.all import *
import pandas as pd
from icevision.models.checkpoint import model_from_checkpoint
from PIL import Image
import os
import gdown

class logodetection:
    def __init__(self) -> None:
        os.environ['TORCH_HOME'] = 'utils/models/torchvision'
        folder = 'utils/models'
        os.makedirs(folder, exist_ok=True)
        checkpoint_path_1 = 'utils/models/logo-retinanet-checkpoint-52k_384_50.pth'
        checkpoint_path_2 = 'utils/models/logo-retinanet-checkpoint-30000_30.pth'

        gdown.download('https://drive.google.com/u/0/uc?id=1J4hG6MRY-k_72wtz6DlYBTiHvAFLbHKP&export=download', checkpoint_path_1)
        gdown.download('https://drive.google.com/u/0/uc?id=11zaPWWCCUKF9yh2mIxWnpfCffUqDZUeu&export=download', checkpoint_path_2)

        self.iou_threshold = 0.6
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.class_map, self.model_1, self.model_type_1, self.image_size = self.__get_logo_model(checkpoint_path_1)
        _, self.model_2, self.model_type_2, _ = self.__get_logo_model(checkpoint_path_2)

        self.valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(self.image_size), tfms.A.Normalize()])

    @staticmethod
    def __get_xyxy(preds):
        boxes = []
        for box in preds:
            boxes.append([box.xmin, box.ymin, box.xmax, box.ymax])
        return boxes

    @staticmethod
    def __get_logo_model(checkpoint_path):
        checkpoint_and_model = model_from_checkpoint(checkpoint_path)
        model_type = checkpoint_and_model["model_type"]
        class_map = checkpoint_and_model["class_map"]
        img_size = checkpoint_and_model["img_size"]
        model = checkpoint_and_model["model"]
        return class_map, model, model_type, img_size

    @staticmethod
    def __iou(b1, b2):
      """
      Get the intersection over union of two boxes
      """
      xmin = max(b1[0], b2[0])
      ymin = max(b1[1], b2[1])
      xmax = min(b1[2], b2[2])
      ymax = min(b1[3], b2[3])

      intersection = max(xmax-xmin, 0)*max(ymax-ymin, 0)

      b1_area = abs((b1[2]-b1[0])*(b1[3]-b1[1]))
      b2_area = abs((b2[2]-b2[0])*(b2[3]-b2[1]))

      return intersection/(b1_area+b2_area-intersection+1e-6)

    @staticmethod
    def __merge_boxes(c1, c2):
      """
        Merges two boxes
        Takes and returns boxes in the format [xmin, ymin, xmax, ymax]

        Parameters:
          c1: coordinates of box 1 
          c2: coordinates of box 2

        Returns:
          merged box
      """
      xmin = min(c1[0], c2[0]) 
      ymin = min(c1[1], c2[1])
      xmax = max(c1[2], c2[2])
      ymax = max(c1[3], c2[3])
      return [xmin, ymin, xmax, ymax]

    def predict(self, image, score):
        img = Image.fromarray(image)
        pred_dict_1  = self.model_type_1.end2end_detect(img, self.valid_tfms, self.model_1, class_map=self.class_map, detection_threshold=score)
        result = self.__get_xyxy(pred_dict_1['detection']['bboxes'])
        pred_dict_2  = self.model_type_2.end2end_detect(img, self.valid_tfms, self.model_2, class_map=self.class_map, detection_threshold=score)
        output = self.__get_xyxy(pred_dict_2['detection']['bboxes'])
        for b1 in output:
            merged = False
            if not b1:
                continue
            for j, b2 in enumerate(result):
                if not b2:
                    continue
                if self.__iou(b1, b2)>= self.iou_threshold:
                    merged = True
                    result[j] = self.__merge_boxes(b1, b2)
                    break
            if merged==False:
                result.append(b1)

        return result

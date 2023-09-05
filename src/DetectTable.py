from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
import cv2
import numpy as np
import os
from google.colab.patches import cv2_imshow

class DetectTable():
  def __init__(self):
    self.processor = DetrImageProcessor.from_pretrained("TahaDouaji/detr-doc-table-detection")
    self.model = DetrForObjectDetection.from_pretrained("TahaDouaji/detr-doc-table-detection")


  def extract_table_bbox(self,image_path):
    tables=[]
    image = Image.open(image_path)
    self.inputs = self.processor(images=np.array(image), return_tensors="pt")
    self.outputs = self.model(**self.inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    self.target_sizes = torch.tensor([image.size[::-1]])
    self.results = self.processor.post_process_object_detection(self.outputs, target_sizes=self.target_sizes, threshold=0.9)[0]

    # Convert PIL image to OpenCV format
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for score, label, box in zip(self.results["scores"], self.results["labels"], self.results["boxes"]):
      box = [round(i, 2) for i in box.tolist()]
      class_label = self.model.config.id2label[label.item()]



      if round(score.item(), 3) > 0.95:
        # Draw rectangle around the detected object
        x_min, y_min, x_max, y_max = map(int, box)
        if x_min-10 < 0:
          table= [(x_min), (y_min-10), (x_max+20), (y_max+20)]
        elif y_min-10 < 0:
          table= [(x_min-10), (y_min), (x_max+20), (y_max+20)]
        else:
          table= [(x_min-10), (y_min-10), (x_max+20), (y_max+20)]
        # cv2.rectangle(image_cv2, (x_min-10, y_min-10), (x_max+20, y_max+20), (0, 0, 0), 2)
        tables.append(table)
        # cv2_imshow(image_cv2)
        # print(f"Detected {class_label} with confidence {round(score.item(), 3)} at location {box}")
    return tables, image_cv2

  def extract_bounding_box(self,bbox,image):
      # cv2_imshow(image)
      x1, y1, x2, y2 = bbox
      height=abs(y2-y1)
      width=abs(x2-x1)
      extracted_image = image[y1:y1+height, x1:x1+width]
      return extracted_image

  def execute(self,image_path,folder_name,page):

    if not os.path.exists(folder_name):
      os.mkdir(folder_name)

    tables,_  = self.extract_table_bbox(image_path)
    # print(tables)

    image=cv2.imread(image_path)
    for i,j in enumerate(tables):
      table_img=self.extract_bounding_box(j,image)
      cv2_imshow(table_img)
      cv2.imwrite(os.path.join(folder_name,f"table{i}_in_page{page}.png"),table_img)

    return tables

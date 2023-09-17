import fitz
import cv2
import numpy as np
import pandas as pd
import os

class ExtractDataBordered():

    def __init__(self, image, page_no,bbox,table_no,pdf_path):
        self.image = image
        self.page_no = page_no
        self.bbox = bbox
        self.table_no = table_no
        self.pdf_document= pdf_path

    def execute(self,page_no,table_no):
        self.image = self.extract_only_table(self.image,self.bbox)
        self.grayscale_image()
        # self.store_process_image( self.grey)
        self.threshold_image()
        # self.store_process_image(self.thresholded_image)
        self.invert_image()
        # self.store_process_image(self.inverted_image)
        self.erode_vertical_lines()
        # self.store_process_image(self.vertical_lines_eroded_image)
        self.erode_horizontal_lines()
        # self.store_process_image(self.horizontal_lines_eroded_image)
        # self.store_process_image(self.image)
        # print("No. of Rows: " , len(self.rows), "No. of Columns: " , len(self.columns))
        self.get_rows_and_columns()
        # print(self.row_range, self.column_range)
        self.get_data()
        # self.store_process_image(self.image)
        # self.generate_csv_file(page_no,table_no)
        # return(self.table)
        table_extracted=self.generate_json(self.table,self.page_no,self.bbox)
        return table_extracted

    def extract_only_table(self,image, bbox):

      x1, y1, x2, y2 = bbox
      height=abs(y2-y1)
      width=abs(x2-x1)

      img_height, img_width, _ = image.shape

      mask = np.zeros((img_height, img_width, 1), dtype=np.uint8)
      cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)

      inverted_mask = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
      cv2.rectangle(inverted_mask, (x1, y1), (x2, y2), (0, 0, 0), -1)

      # cv2_imshow(mask)
      # cv2_imshow(inverted_mask)
      # print(mask.shape, inverted_mask.shape, image.shape)

      extracted_image = cv2.bitwise_and(image, image, mask=mask)

      extracted_image_new = cv2.add( extracted_image, inverted_mask, mask = None)
      # cv2_imshow(extracted_image_new)

      return extracted_image_new

    def grayscale_image(self):
        self.grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def threshold_image(self):
        self.thresholded_image = cv2.threshold(self.grey, 253, 255, cv2.THRESH_BINARY)[1]

    def invert_image(self):
        self.inverted_image = cv2.bitwise_not(self.thresholded_image)

    def erode_vertical_lines(self):
        hor = np.array([[1,1,1,1,1,1]])
        self.vertical_lines_eroded_image = cv2.erode(self.inverted_image, hor, iterations=15)
        self.vertical_lines_eroded_image = cv2.dilate(self.vertical_lines_eroded_image, hor, iterations=15)
        contours, hierarchy = cv2.findContours(self.vertical_lines_eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # lines = cv2.HoughLines(edges,1,np.pi/180,200)
        # cv2.drawContours(self.image, contours, -1, (0, 255, 0), 2)
        self.rows = contours

    def erode_horizontal_lines(self):
        ver = np.array([[1],
               [1],
               [1],
               [1],
               [1],
               [1],
               [1]])
        self.horizontal_lines_eroded_image = cv2.erode(self.inverted_image, ver, iterations=15)
        self.horizontal_lines_eroded_image = cv2.dilate(self.horizontal_lines_eroded_image, ver, iterations=15)
        contours, hierarchy = cv2.findContours(self.horizontal_lines_eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        self.columns = contours

    def store_process_image(self, image):
        # cv2_imshow(image)
        pass

    def get_rows_and_columns(self):
      self.row_range = []
      for line in self.rows:
        sorted_data = sorted(line, key=lambda x: x[0][0])
        self.row_range.append(sorted_data[0][0][1])
      self.row_range.sort()

      self.column_range = []
      for line in self.columns:
        sorted_data = sorted(line, key=lambda x: x[0][1])
        self.column_range.append(sorted_data[0][0][0])
      self.column_range.sort()

    def extract_text(self, cell_bbox):

      selected_text = " "
      for inst in self.text_instances:
        bbox = inst[0:4]

        if cell_bbox[0]<=bbox[0] and cell_bbox[1]<=bbox[1] and cell_bbox[2]>=bbox[2] and cell_bbox[3]>=bbox[3]:
            selected_text += " " + inst[4]

      return selected_text

    def get_data(self):
      if "cells_folder" not in os.listdir():
        os.mkdir("cells_folder")

      row = len(self.row_range)
      col = len(self.column_range)

      self.table= [[" " for _ in range(col -1)] for _ in range(row -1)]


      doc = fitz.open(self.pdf_document)
      page = doc.load_page(self.page_no)

      self.text_instances = page.get_text("words")

      for r in range(row - 1):
        for c in range(col - 1):

          x0, y0, x1, y1 =  self.column_range[c], self.row_range[r] - 2, self.column_range[c + 1], self.row_range[r + 1]+2
          # cv2.rectangle(self.image, (x0, y0), (x1, y1), (0, 0, 255), 2)
          # cropped_image = self.image[y0:y1, x0:x1]
          # cv2_imshow(cropped_image)

          text = self.extract_text([x0, y0, x1, y1])
          # print(r, c)
          self.table[r][c] = text

    def generate_csv_file(self,page_no,table_no):
        # print(self.table)
        df = pd.DataFrame(self.table)
        df.to_csv(f"table{self.table_no}_in_page{self.page_no}.csv", index = False)
        # print(df)

    def generate_json(self,table,page_no,bbox):
     table_dict={"page_no":page_no,
                 "bbox":[bbox],
                 "data":table}
     return table_dict

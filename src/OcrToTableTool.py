import cv2
import numpy as np
import subprocess
import os
import easyocr
import numpy as np
from PIL import Image, ImageDraw



class OcrToTableTool():

    def __init__(self, image, original_image,page_no,table_no,bbox):
        self.thresholded_image = image
        self.original_image = original_image.copy()
        self.reader = easyocr.Reader(['en'])
        self.table_contents={}
        self.cell_area=650
        self.page_no=page_no
        self.table_no=table_no
        self.bbox=bbox

    def execute(self):
        self.dilate_image()
        # self.store_process_image('0_dilated_image.jpg', self.dilated_image)
        self.find_contours()
        # self.store_process_image('1_contours.jpg', self.image_with_contours_drawn)
        self.convert_contours_to_bounding_boxes()
        # self.store_process_image('2_bounding_boxes.jpg', self.image_with_all_bounding_boxes)

        self.mean_height = self.get_mean_height_of_bounding_boxes()
        self.mean_width =self.get_mean_width_of_bounding_boxes()

        self.sort_bounding_boxes_by_y_coordinate()
        self.club_all_bounding_boxes_by_similar_y_coordinates_into_rows()
        self.sort_all_rows_by_x_coordinate()

        self.sort_bounding_boxes_by_x_coordinate()
        self.club_all_bounding_boxes_by_similar_x_coordinates_into_columns()
        self.sort_all_columns_by_y_coordinate()

        self.create_ocr_dict(self.rows)

        self.crop_each_bounding_box_and_ocr_rows()

        self.crop_each_bounding_box_and_ocr_cols()
        # self.generate_csv_file(name = "columns")
        self.create_final_table()
        # self.generate_csv_file(self.page_no,self.table_no)

        table_extracted=self.generate_json(self.table,self.page_no,self.bbox)
        return table_extracted

    def threshold_image(self):
        return cv2.threshold(self.grey_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def convert_image_to_grayscale(self):
        return cv2.cvtColor(self.image, self.dilated_image)

    def dilate_image(self):
        kernel_to_remove_gaps_between_words = np.array([
                [1,1,1,1,1,1,1],
               [1,1,1,1,1,1,1]
        ])
        self.dilated_image = cv2.dilate(self.thresholded_image, kernel_to_remove_gaps_between_words, iterations=2)
        simple_kernel = np.ones((2,2), np.uint8)
        self.dilated_image = cv2.dilate(self.dilated_image, simple_kernel, iterations=3)

    def find_contours(self):
        result = cv2.findContours(self.dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = result[0]
        self.image_with_contours_drawn = self.original_image.copy()
        cv2.drawContours(self.image_with_contours_drawn, self.contours, -1, (0, 255, 0), 3)

    def approximate_contours(self):
        self.approximated_contours = []
        for contour in self.contours:
            approx = cv2.approxPolyDP(contour, 3, True)
            self.approximated_contours.append(approx)

    def draw_contours(self):
        self.image_with_contours = self.original_image.copy()
        cv2.drawContours(self.image_with_contours, self.approximated_contours, -1, (0, 255, 0), 5)

    def convert_contours_to_bounding_boxes(self):
        self.bounding_boxes = []
        self.image_with_all_bounding_boxes = self.original_image.copy()
        for contour in self.contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w*h > self.cell_area and h >10:
              self.bounding_boxes.append((x, y, w, h))
              self.image_with_all_bounding_boxes = cv2.rectangle(self.image_with_all_bounding_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2_imshow(self.image_with_all_bounding_boxes)
    def get_mean_height_of_bounding_boxes(self):
        heights = []
        for bounding_box in self.bounding_boxes:
            x, y, w, h = bounding_box
            heights.append(h)
        return np.mean(heights)

    def get_mean_width_of_bounding_boxes(self):
      widths = []
      for bounding_box in self.bounding_boxes:
          x, y, w, h = bounding_box
          widths.append(w)
      return np.mean(widths)

    def sort_bounding_boxes_by_y_coordinate(self):
        self.bounding_boxes = sorted(self.bounding_boxes, key=lambda x: x[1])

    def sort_bounding_boxes_by_x_coordinate(self):
        self.bounding_boxes = sorted(self.bounding_boxes, key=lambda x: x[0])

    def club_all_bounding_boxes_by_similar_y_coordinates_into_rows(self):
        self.rows = []
        half_of_mean_height = self.mean_height / 2
        current_row = [ self.bounding_boxes[0] ]
        for bounding_box in self.bounding_boxes[1:]:
            current_bounding_box_y = bounding_box[1]
            previous_bounding_box_y = current_row[-1][1]
            distance_between_bounding_boxes = abs(current_bounding_box_y - previous_bounding_box_y)
            if distance_between_bounding_boxes <= half_of_mean_height:
                current_row.append(bounding_box)
            else:
                self.rows.append(current_row)
                current_row = [ bounding_box ]
        self.rows.append(current_row)

    def club_all_bounding_boxes_by_similar_x_coordinates_into_columns(self):
      self.columns = []
      half_of_mean_width = self.mean_width / 3
      current_column = [self.bounding_boxes[0]]
      for bounding_box in self.bounding_boxes[1:]:
          current_bounding_box_x = bounding_box[0]
          previous_bounding_box_x = current_column[-1][0]
          distance_between_bounding_boxes = abs(current_bounding_box_x - previous_bounding_box_x)
          if distance_between_bounding_boxes <= half_of_mean_width:
              current_column.append(bounding_box)
          else:
              self.columns.append(current_column)
              current_column = [bounding_box]
      self.columns.append(current_column)


    def sort_all_rows_by_x_coordinate(self):
        for row in self.rows:
            row.sort(key=lambda x: x[0])


    def sort_all_columns_by_y_coordinate(self):
        for column in self.columns:
            column.sort(key=lambda x: x[1])


    def create_ocr_dict(self,table):
        if not os.path.exists('cells_folder'):
          os.mkdir("cells_folder")
        image_number = 0
        for row in table:
            for bounding_box in row:
                # print(bounding_box)
                x, y, w, h = bounding_box
                if h > 5:
                  if y > 5:
                    y = y - 5
                  else :
                    y=y
                  # cv2_imshow(self.original_image)
                  cropped_image = self.original_image[y:y+h+2, x:x+w]

                  image_slice_path = f"cells_folder/cell{image_number}.jpg"

                  grey_image= cv2.cvtColor(cropped_image,cv2.COLOR_BGR2GRAY)
                  # Create the sharpening kernel
                  kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                  # Apply the sharpening kernel to the image using filter2D
                  sharpened = cv2.filter2D(cropped_image, -1, kernel)
                  thresh = cv2.threshold(sharpened, 200, 255, cv2.THRESH_BINARY)[1]

                  cv2.imwrite(image_slice_path,sharpened)
                  # cv2_imshow(sharpened)
                  results_from_ocr = self.get_result_from_easyocr(image_slice_path)
                  # print(results_from_ocr)
                  image_number += 1

                  self.table_contents[str(bounding_box)]=results_from_ocr



    def crop_each_bounding_box_and_ocr_rows(self):
        if not os.path.exists('cells_folder'):
          os.mkdir("cells_folder")
        self.row_table = []
        current_row = []
        image_number = 0
        for row in self.rows:
            for bounding_box in row:
                x, y, w, h = bounding_box
                if h > 5:
                  y = y - 5
                  # cropped_image = self.original_image[y:y+h+2, x:x+w]
                  # image_slice_path = f"cells_folder/cell{image_number}.jpg"

                  # grey_image= cv2.cvtColor(cropped_image,cv2.COLOR_BGR2GRAY)
                  # # Create the sharpening kernel
                  # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                  # # Apply the sharpening kernel to the image using filter2D
                  # sharpened = cv2.filter2D(grey_image, -1, kernel)
                  # thresh = cv2.threshold(sharpened, 200, 255, cv2.THRESH_BINARY)[1]

                  # cv2_imshow(thresh)

                  # cv2.imwrite(image_slice_path,thresh)

                  # results_from_ocr = self.get_result_from_easyocr(image_slice_path)
                  # print("\""+self.table_contents[str(bounding_box)]+"\"")
                  current_row.append("\""+self.table_contents[str(bounding_box)]+"\"")
                  image_number += 1
            self.row_table.append(current_row)
            current_row = []
        # print(self.row_table)


    def crop_each_bounding_box_and_ocr_cols(self):
        self.column_table = []
        current_row = []
        image_number = 0
        for row in self.columns:
            for bounding_box in row:
                x, y, w, h = bounding_box
                if h > 5:
                  y = y - 5
                  # cropped_image = self.original_image[y:y+h+2, x:x+w]
                  # image_slice_path = f"cells_folder/cell{image_number}.jpg"

                  # grey_image= cv2.cvtColor(cropped_image,cv2.COLOR_BGR2GRAY)
                  # # Create the sharpening kernel
                  # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                  # # Apply the sharpening kernel to the image using filter2D
                  # sharpened = cv2.filter2D(grey_image, -1, kernel)
                  # thresh = cv2.threshold(sharpened, 200, 255, cv2.THRESH_BINARY)[1]

                  # cv2_imshow(thresh)

                  # cv2.imwrite(image_slice_path,thresh)

                  # results_from_ocr = self.get_result_from_easyocr(image_slice_path)
                  # print("\""+results_from_ocr+"\"")
                  current_row.append("\""+self.table_contents[str(bounding_box)]+"\"")
                  image_number += 1
            self.column_table.append(current_row)
            current_row = []
        # print(self.column_table)

    def get_result_from_tersseract(self, image_path):
        import pytesseract
        from pytesseract import Output

        output =  pytesseract.image_to_string(image_path,lang='eng')
        output = output.strip()
        return output

    def get_result_from_easyocr(self, image_path):

      text_coordinates = self.detect_text_blocks(image_path)
      recognition_results = self.reader.recognize(image_path,
                                 horizontal_list=text_coordinates,
                                 free_list=[]
                                 )

      return " ".join([t[1] for t in recognition_results]) if recognition_results else ""

    def generate_csv_file(self,page_no,table_no):
        # print(self.table)
        with open(f"table{table_no}_in_page{page_no}.csv", "w") as f:
          for row in self.table:
              f.write(",".join(row) + "\n")

    def store_process_image(self, file_name, image):
        path = "./process_images/ocr_table_tool/" + file_name
        # cv2_imshow(image)

    def detect_text_blocks(self,img_path):
      detection_result = self.reader.detect(img_path,
                                 width_ths=0.7,
                                 mag_ratio=1.5
                                 )
      text_coordinates = detection_result[0][0]
      return text_coordinates

    def find_element_position(self,matrix, target_element):
      for row_index, row in enumerate(matrix):
          for col_index, element in enumerate(row):
              if element == target_element:
                  matrix[row_index][col_index]=" "
                  return row_index, col_index
      return None

    def create_final_table(self):
      from itertools import chain
      self.text=list(chain.from_iterable(self.row_table))
      # print(self.text)

      row_len=len(self.row_table)
      col_len=len(self.column_table)

      self.table= [[" " for _ in range(col_len)] for _ in range(row_len)]
      # print(self.table)

      for i in self.text:

        row,_ = self.find_element_position(self.row_table,i)
        col,_ = self.find_element_position(self.column_table,i)
        self.table[row][col]=i


    def generate_json(self,table,page_no,bbox):
     table_dict={"page_no":page_no,
                 "bbox":[bbox],
                 "data":table}
     return table_dict


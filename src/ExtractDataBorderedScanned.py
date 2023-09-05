class ExtractDataBorderedScanned():

    def __init__(self, image, page_no,bbox,table_no,pdf_path):

        self.reader = easyocr.Reader(['en'])
        self.image = image
        self.page_no = page_no
        self.bbox = bbox
        self.table_no = table_no
        self.pdf_document= pdf_path

    def execute(self):
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
        print("No. of Rows: " , len(self.rows), "No. of Columns: " , len(self.columns))
        self.get_rows_and_columns()
        # print(self.row_range, self.column_range)
        self.get_data()
        # self.store_process_image(self.image)
        # self.generate_csv_file()
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
        cv2_imshow(extracted_image_new)

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
        cv2_imshow(image)

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

    def get_data(self):
      if "cells_folder" not in os.listdir():
        os.mkdir("cells_folder")

      row = len(self.row_range)
      col = len(self.column_range)

      self.table= [[" " for _ in range(col -1)] for _ in range(row -1)]

      for r in range(row - 1):
        for c in range(col - 1):

          x0, y0, x1, y1 =  self.column_range[c], self.row_range[r], self.column_range[c + 1], self.row_range[r + 1]
          # cv2.rectangle(self.image, (x0, y0), (x1, y1), (0, 0, 255), 2)
          cropped_image = self.image[y0:y1, x0:x1]
          # cv2_imshow(cropped_image)
          image_slice_path = f"cells_folder/cell{r}_{c}.jpg"

          grey_image= cv2.cvtColor(cropped_image,cv2.COLOR_BGR2GRAY)
          # Create the sharpening kernel
          kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
          # Apply the sharpening kernel to the image using filter2D
          sharpened = cv2.filter2D(cropped_image, -1, kernel)
          thresh = cv2.threshold(sharpened, 215, 255, cv2.THRESH_BINARY)[1]

          cv2.imwrite(image_slice_path,thresh)

          results_from_ocr = self.get_result_from_easyocr(image_slice_path)
          # print(r, c)
          self.table[r][c] = results_from_ocr
          # print(results_from_ocr)


    def get_result_from_easyocr(self, image_path):

      text_coordinates = self.detect_text_blocks(image_path)
      recognition_results = self.reader.recognize(image_path,
                                 horizontal_list=text_coordinates,
                                 free_list=[]
                                 )
      # print(recognition_results)

      return " ".join([t[1] for t in recognition_results]) if recognition_results else ""


    def detect_text_blocks(self,img_path):
      detection_result = self.reader.detect(img_path,
                                 width_ths=0.7,
                                 mag_ratio=1.5
                                 )
      # print(detection_result)
      text_coordinates = detection_result[0][0]
      return text_coordinates

    def generate_csv_file(self):
        # print(self.table)
        df = pd.DataFrame(self.table)
        df.to_csv("output.csv", index = False)

    def generate_json(self,table,page_no,bbox):
     table_dict={"page_no":page_no,
                 "bbox":[bbox],
                 "data":table}
     return table_dict

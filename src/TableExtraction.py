import fitz
import cv2
import os
import numpy as np
from src.GenerateTable import GenerateTable
from src.ExtractDataBordered import ExtractDataBordered
from src.TableLinesRemover import TableLinesRemover
from src.OcrToTableTool import OcrToTableTool
from src.ExtractDataBorderedScanned import ExtractDataBorderedScanned
from src.DetectTable import DetectTable

class TableExtraction():

  def __init__(self, doc_path):

    self.doc_path = doc_path

    self.table_detector = DetectTable()


  def execute(self):
    self.document = fitz.open(self.doc_path)
    is_scanned = not self.get_pdf_searchable_pages(self.doc_path)
    self.tables= {}
    self.tables=self.read_pdf(is_scanned=is_scanned)
    self.extracted_table_list=[]

    if is_scanned == False:
      for page_no,bbox_list in self.tables.items():

          doc = fitz.open(self.doc_path)
          page = doc.load_page(page_no)
          pix = page.get_pixmap()
          pix.save("page_test.png")
          is_borderless = not self.is_bodered("page_test.png")

          if is_borderless == True:

            for i,j in enumerate(bbox_list):
              table_gen = GenerateTable(j,page_no,self.doc_path)
              data= table_gen.execute(page_no,i)
              if data["data"] != [[" "]]:
                self.extracted_table_list.append(data)

          else :
            doc = fitz.open(self.doc_path)
            page = doc.load_page(page_no)
            pix = page.get_pixmap()
            pix.save("page.png")
            image = cv2.imread("page.png")
            for table_no,bbox in enumerate(bbox_list):
              extract_data = ExtractDataBordered(image, page_no,bbox,table_no,self.doc_path)
              data= table_gen.execute(page_no,table_no)
              if data["data"] != [[" "]]:
                self.extracted_table_list.append(data)

    elif is_scanned == True :
      for page_no,bbox_list in self.tables.items():

          doc = fitz.open(self.doc_path)
          page = doc.load_page(page_no)
          pix = page.get_pixmap()
          pix.save("page_test.png")
          is_borderless = not self.is_bodered("page_test.png")

          if is_borderless == True:
            zoom_x = 2.0
            zoom_y = 2.0
            doc = fitz.open(self.doc_path)
            page = doc.load_page(page_no)
            mat = fitz.Matrix(zoom_x, zoom_y)
            pix = page.get_pixmap(matrix=mat)
            pix.save("page.png")
            image = cv2.imread("page.png")
            # cv2_imshow(image)
            for table_no,j in enumerate(bbox_list):
              cropped_image=self.extract_bounding_box(j,image)
              # cv2_imshow(cropped_image)
              lines_remover = TableLinesRemover(cropped_image)
              image_without_lines = lines_remover.execute()
              ocr_tool = OcrToTableTool(image_without_lines,cropped_image,page_no,table_no)
              data= table_gen.execute(page_no,i)
              if data["data"] != [[" "]]:
                self.extracted_table_list.append(data)

          else :
            doc = fitz.open(self.doc_path)
            page = doc.load_page(page_no)
            pix = page.get_pixmap()
            pix.save("page.png")
            image = cv2.imread("page.png")
            for table_no,bbox in enumerate(bbox_list):
              extract_data = ExtractDataBorderedScanned(image, page_no,bbox,table_no,self.doc_path)
              data= table_gen.execute(page_no,table_no)
              if data["data"] != [[" "]]:
                self.extracted_table_list.append(data)

    elif isinstance(is_scanned, (list)):
      digital_list = is_scanned[0]
      scanned_list = is_scanned[1]
      for page_no in digital_list:

        bbox_list=self.tables[page_no]
        doc = fitz.open(self.doc_path)
        page = doc.load_page(page_no)
        pix = page.get_pixmap()
        pix.save("page_test.png")
        is_borderless = not self.is_bodered("page_test.png")

        if is_borderless == True:
            for i,j in enumerate(bbox_list):
              table_gen = GenerateTable(j,page_no,self.doc_path)
              data= table_gen.execute(page_no,i)
              if data["data"] != [[" "]]:
                self.extracted_table_list.append(data)
        else :
            doc = fitz.open(self.doc_path)
            page = doc.load_page(page_no)
            pix = page.get_pixmap()
            pix.save("page.png")
            image = cv2.imread("page.png")
            for table_no,bbox in enumerate(bbox_list):
              extract_data = ExtractDataBordered(image, page_no,bbox,table_no,self.doc_path)
              data= table_gen.execute(page_no,table_no)
              if data["data"] != [[" "]]:
                self.extracted_table_list.append(data)


      for page_no in scanned_list:

          bbox_list=self.tables[page_no]
          doc = fitz.open(self.doc_path)
          page = doc.load_page(page_no)
          pix = page.get_pixmap()
          pix.save("page_test.png")
          is_borderless = not self.is_bodered("page_test.png")

          if is_borderless == True:
            zoom_x = 2.0
            zoom_y = 2.0
            doc = fitz.open(self.doc_path)
            page = doc.load_page(page_no)
            mat = fitz.Matrix(zoom_x, zoom_y)
            pix = page.get_pixmap(matrix=mat)
            pix.save("page.png")
            image = cv2.imread("page.png")
            # cv2_imshow(image)
            for table_no,j in enumerate(bbox_list):
              cropped_image=self.extract_bounding_box(j,image)
              # cv2_imshow(cropped_image)
              lines_remover = TableLinesRemover(cropped_image)
              image_without_lines = lines_remover.execute()
              ocr_tool = OcrToTableTool(image_without_lines,cropped_image,page_no,table_no)
              data= table_gen.execute(page_no,i)
              if data["data"] != [[" "]]:
                self.extracted_table_list.append(data)

          else :
            doc = fitz.open(self.doc_path)
            page = doc.load_page(page_no)
            pix = page.get_pixmap()
            pix.save("page.png")
            image = cv2.imread("page.png")
            for table_no,bbox in enumerate(bbox_list):
              extract_data = ExtractDataBorderedScanned(image, page_no,bbox,table_no,self.doc_path)
              data= table_gen.execute(page_no,table_no)
              if data["data"] != [[" "]]:
                self.extracted_table_list.append(data)

    print(self.extracted_table_list)
    return self.extracted_table_list

  def extract_bounding_box(self,bbox,image):
      x1, y1, x2, y2 = bbox
      height=abs(y2-y1)
      width=abs(x2-x1)
      extracted_image = image[y1:y1+height, x1:x1+width]
      # cv2_imshow(extracted_image)
      return extracted_image

  def get_pdf_searchable_pages(self,fname):
    # pip install pdfminer
    from pdfminer.pdfpage import PDFPage
    searchable_pages = []
    non_searchable_pages = []
    page_num = 0
    with open(fname, 'rb') as infile:

        for page in PDFPage.get_pages(infile):
            page_num += 1
            if 'Font' in page.resources.keys():
                searchable_pages.append(page_num)
            else:
                non_searchable_pages.append(page_num)
    if page_num > 0:
        if len(searchable_pages) == 0:
            # print(f"Document '{fname}' has {page_num} page(s). "
            #       f"Complete document is non-searchable")
            return False
        elif len(non_searchable_pages) == 0:
            # print(f"Document '{fname}' has {page_num} page(s). "
            #       f"Complete document is searchable")
            return True
        else:
            # print(f"searchable_pages : {searchable_pages}")
            # print(f"non_searchable_pages : {non_searchable_pages}")
            return [searchable_pages, non_searchable_pages]
    else:
        print(f"Not a valid document")

  def extract_table_digital_borderless(self,page_no,bbox_list):
    doc_path=self.doc_path

    for i,j in enumerate(bbox_list):
      table_gen = GenerateTable(j,page_no,doc_path)
      table = table_gen.execute(page_no,i)

  def is_bodered(self,image_path):

    image = cv2.imread(image_path)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresholded_image = cv2.threshold(grey, 250, 255, cv2.THRESH_BINARY)[1]
    inverted_image = cv2.bitwise_not(thresholded_image)

    hor = np.array([[1,1,1,1,1,1, 1]])
    vertical_lines_eroded_image = cv2.erode(inverted_image, hor, iterations=15)
    vertical_lines_eroded_image = cv2.dilate(vertical_lines_eroded_image, hor, iterations=15)
    # cv2_imshow(vertical_lines_eroded_image)
    r_contours, hierarchy = cv2.findContours(vertical_lines_eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ver = np.array([[1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1]])
    horizontal_lines_eroded_image = cv2.erode(inverted_image, ver, iterations=15)
    horizontal_lines_eroded_image = cv2.dilate(horizontal_lines_eroded_image, ver, iterations=15)
    c_contours, hierarchy = cv2.findContours(horizontal_lines_eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2_imshow(horizontal_lines_eroded_image)

    combined_image = cv2.add(vertical_lines_eroded_image, horizontal_lines_eroded_image)
    # cv2_imshow(combined_image)

    rows, columns = self.get_rows_and_columns(r_contours, c_contours)
    # print(rows, columns)
    if len(rows) > 2 and len(columns) > 2:
      if self.intersects(rows, columns):
        return True
    return False

  def intersects(self,rows, columns):
    i = 0
    # print(len(rows), len(columns))
    for xr0, yr0, xr1, yr1 in rows:
      for xc0, yc0, xc1, yc1 in columns:
        # print(xr0, yr0, xr1, yr1, xc0, yc0, xc1, yc1)
        if xr0 <= xc0 and yr0 >= yc0 and xr1 >= xc1 and yr1 <= yc1:
          i+=1
    cal_i = (len(rows) * len(columns))*2/3
    # print(i, cal_i)
    return True if i >= cal_i else False

  def get_rows_and_columns(self,r_c, c_c):
    row_range = []
    for line in r_c:
      sorted_data = sorted(line, key=lambda x: x[0][0])
      row_range.append([sorted_data[0][0][0], sorted_data[0][0][1], sorted_data[-1][0][0], sorted_data[-1][0][1]])
    row_range.sort()

    column_range = []
    for line in c_c:
      sorted_data = sorted(line, key=lambda x: x[0][1])
      column_range.append([sorted_data[0][0][0], sorted_data[0][0][1], sorted_data[-1][0][0], sorted_data[-1][0][1]])
    column_range.sort()

    return row_range, column_range

  def read_pdf(self,is_scanned=False):



    zoom_x = 1.0 if not is_scanned else 2.0
    zoom_y = 1.0 if not is_scanned else 2.0

    if is_scanned == False or is_scanned == True :
      no_of_pages = len(self.document)
      for page_no in range(no_of_pages):
        page = self.document[page_no]
        mat = fitz.Matrix(zoom_x, zoom_y)
        pix = page.get_pixmap(matrix=mat)
        pix.save("page.png")

        table_list = self.table_detector.execute("page.png","tables_cropped",page_no)
        os.remove("page.png")
        if len(table_list)>=1:
          self.tables[page_no] = table_list


    elif isinstance(is_scanned, (list)):
      digital_list = is_scanned[0]
      scanned_list = is_scanned[1]
      for page_no in digital_list:
        zoom_x = 1.0
        zoom_y = 1.0
        page = self.document[page_no]
        mat = fitz.Matrix(zoom_x, zoom_y)
        pix = page.get_pixmap(matrix=mat)
        pix.save("page.png")
        table_list = self.table_detector.execute("page.png","tables_cropped_digital",page_no)
        if len(table_list)>=1:
          self.tables[page_no] = table_list
      for page_no in scanned_list:
        zoom_x = 2.0
        zoom_y = 2.0
        page = self.document[page_no]
        mat = fitz.Matrix(zoom_x, zoom_y)
        pix = page.get_pixmap(matrix=mat)
        pix.save("page.png")
        table_list = self.table_detector.execute("page.png","tables_cropped_scanned",page_no)
        if len(table_list)>=1:
          self.tables[page_no] = table_list

    return self.tables

import fitz
import cv2

class GenerateTable():
  def __init__(self,table_bbox,page_no,document):
    self.doc = fitz.open(document)
    self.page = self.doc.load_page(page_no)
    self.text_instances = self.page.get_text("words")

    self.table_rect=table_bbox
    self.bbox=table_bbox
    self.selected_text=[]

  def execute(self,page_no,table_no):
    self.extract_words()
    self.bounding_boxes = self.combine_words_along_x(distance_threshold=3)
    if len(self.bounding_boxes)< 1:self.bounding_boxes=[[0,0,1,1," "]]
    self.mean_height =self.get_mean_height_of_bounding_boxes()
    self.mean_width=self.get_mean_width_of_bounding_boxes()


    self.bounding_boxes = self.sort_bounding_boxes_by_y_coordinate(self.bounding_boxes)
    # print(bounding_boxes)
    self.rows=[]
    self.rows = self.club_all_bounding_boxes_by_similar_y_coordinates_into_rows(self.mean_height,self.bounding_boxes,self.rows)
    self.rows = self.sort_all_rows_by_x_coordinate(self.rows)
    # print(rows)
    # print(len(rows))

    self.bounding_boxes=self.sort_bounding_boxes_by_x_coordinate(self.bounding_boxes)
    # print(bounding_boxes)
    self.columns =[]
    self.columns = self.club_all_bounding_boxes_by_similar_x_coordinates_into_columns(self.mean_width,self.bounding_boxes,self.columns)
    self.columns = self.sort_all_columns_by_y_coordinate(self.columns)
    # print(columns)
    # print(len(columns))


    self.row_table = self.crop_each_bounding_box_and_ocr_rows(self.rows)
    # print(row_table)

    self.column_table = self.crop_each_bounding_box_and_ocr_cols(self.columns)
    # print(column_table)

    self.table = self.create_final_table(self.row_table,self.column_table)

    # self.generate_csv_file(self.table,page_no,table_no)

    table_extracted=self.generate_json(self.table,page_no,self.bbox)
    return table_extracted


  def intersects_rect(self,table_rect,bbox):
    if table_rect[0]<=bbox[0] and table_rect[1]<=bbox[1] and table_rect[2]>=bbox[2] and table_rect[3]>=bbox[3]:
      return True
    else:
      return False

  def extract_words(self):
    for inst in self.text_instances:
      bbox = inst[0:4]
      if self.intersects_rect(self.table_rect,bbox):
          self.selected_text.append(inst[0:5])

  def combine_words_along_x(self, distance_threshold=10):
    bbox_list = self.selected_text
    combined_words = []

    i = 0
    while i < len(bbox_list):
        curr_bbox = bbox_list[i]
        combined_text = curr_bbox[4]
        x1, y1, x2, y2 = curr_bbox[:4]

        j = i + 1
        while j < len(bbox_list):
            next_bbox = bbox_list[j]
            next_x1, next_y1, _, _ = next_bbox[:4]

            if next_x1 - x2 <= distance_threshold and next_y1 - y2 <= 0.2:
                combined_text += " " + next_bbox[4]
                x2 = next_bbox[2]
                j += 1
            else:
                break

        combined_words.append((x1, y1, x2, y2, combined_text))
        i = j

    return combined_words



  def get_mean_height_of_bounding_boxes(self):
    heights = []
    for bounding_box in self.bounding_boxes:
        # print(bounding_box)
        x1, y1,x2 , y2,_ = bounding_box
        heights.append(abs(y2-y1))
    return np.mean(heights)

  def get_mean_width_of_bounding_boxes(self):
    widths = []
    for bounding_box in self.bounding_boxes:
        x1, y1, x2, y2 ,_ = bounding_box
        widths.append(abs(x2-x1))
    return np.mean(widths)

  def sort_bounding_boxes_by_y_coordinate(self,bounding_boxes):
      self.bounding_boxes = sorted(bounding_boxes, key=lambda x: x[1])
      # print(bounding_boxes)
      return self.bounding_boxes
  def sort_bounding_boxes_by_x_coordinate(self,bounding_boxes):
      self.bounding_boxes = sorted(bounding_boxes, key=lambda x: x[0])
      # print(bounding_boxes)
      return self.bounding_boxes

  def club_all_bounding_boxes_by_similar_y_coordinates_into_rows(self,mean_height,bounding_boxes,rows):
      half_of_mean_height = mean_height/3
      current_row = [ self.bounding_boxes[0] ]
      for bounding_box in bounding_boxes[1:]:
          current_bounding_box_y = bounding_box[1]
          previous_bounding_box_y = current_row[-1][1]
          distance_between_bounding_boxes = abs(current_bounding_box_y - previous_bounding_box_y)
          if distance_between_bounding_boxes <= half_of_mean_height:
              current_row.append(bounding_box)
          else:
              rows.append(current_row)
              current_row = [ bounding_box ]
      rows.append(current_row)
      # print(rows)
      return rows

  def club_all_bounding_boxes_by_similar_x_coordinates_into_columns(self,mean_width,bounding_boxes,columns):
    half_of_mean_width = mean_width / 3.2
    current_column = [self.bounding_boxes[0]]
    for bounding_box in bounding_boxes[1:]:
        current_bounding_box_x = bounding_box[0]
        previous_bounding_box_x = current_column[-1][0]
        distance_between_bounding_boxes = abs(current_bounding_box_x - previous_bounding_box_x)
        if distance_between_bounding_boxes <= half_of_mean_width:
            current_column.append(bounding_box)
        else:
            columns.append(current_column)
            current_column = [bounding_box]
    columns.append(current_column)
    # print(columns)
    return columns

  def sort_all_rows_by_x_coordinate(self,rows):
      for row in rows:
          row.sort(key=lambda x: x[0])
      return rows


  def sort_all_columns_by_y_coordinate(self,columns):
      for column in columns:
          column.sort(key=lambda x: x[1])
      return columns


  def crop_each_bounding_box_and_ocr_rows(self,rows):
    row_table=[]
    current_row = []
    for row in rows:
        for bounding_box in row:
            x1, y1, x2, y2 ,text = bounding_box
            if abs(x2-x1)>7 :
              current_row.append("\""+bounding_box[-1]+"\"")

        row_table.append(current_row)
        current_row = []
    # print(row_table)
    return row_table

  def crop_each_bounding_box_and_ocr_cols(self,columns):
      column_table=[]
      current_row = []
      image_number = 0
      for row in columns:
          for bounding_box in row:
              x1, y1, x2, y2 ,text = bounding_box
              if abs(x2-x1)>7 :
                current_row.append("\""+bounding_box[-1]+"\"")
          column_table.append(current_row)
          current_row = []
      # print(column_table)
      return column_table

  def find_element_position(self,matrix, target_element):
    for row_index, row in enumerate(matrix):
        for col_index, element in enumerate(row):
            if element == target_element:
                matrix[row_index][col_index]=" "
                return row_index, col_index
    return None

  def create_final_table(self,row_table,column_table):
    from itertools import chain
    text=list(chain.from_iterable(row_table))
    # print(text)

    row_len=len(row_table)
    col_len=len(column_table)

    table= [[" " for _ in range(col_len)] for _ in range(row_len)]
    # print(table)

    for i in text:
      # print(i)
      row,_ = self.find_element_position(row_table,i)
      col,_ = self.find_element_position(column_table,i)
      table[row][col]=i
    # print(table)
    return table

  def generate_csv_file(self,table,page_no,table_no):
      # print(table)
      with open(f"table{table_no}_in_page{page_no}.csv", "w") as f:
          for row in table:
              f.write(",".join(row) + "\n")

  def generate_json(self,table,page_no,bbox):
     table_dict={"page_no":page_no,
                 "bbox":[bbox],
                 "data":table}
     return table_dict


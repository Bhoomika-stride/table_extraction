import fitz
import cv2
import pandas as pd

def get_doc_dimensions(doc_path):

  doc = fitz.open(doc_path)
  page = doc.load_page(0)
  pix = page.get_pixmap()
  pix.save("page.png")
  image = cv2.imread("page.png")
  # cv2_imshow(image)

  return image.shape[0:2]

def add_table_no(extracted_tables):

    for i, data in enumerate(extracted_tables):
      data["table_no"] = i

    return extracted_tables

def append_columns(list_0, list_1):
  result = []

  for sublist1, sublist2 in zip(list_0, list_1):
      combined_sublist = sublist1 + sublist2
      result.append(combined_sublist)
  return result

def detect_multipage_table(new_extracted_tables, doc_path):

  multi_page_tables = {}
  height, width = get_doc_dimensions(doc_path)
  i = 0

  for table_no in range(len(new_extracted_tables)-1):
    table_0 = new_extracted_tables[table_no]
    table_1 = new_extracted_tables[table_no + 1]
    if table_0["page_no"] == table_1["page_no"] - 1:#check if they are in consecutivee pages


        #checking if the first table ends at end of page and second table begins at beginning of page for row extended page
        if (table_0["bbox"][0][3] >= (height - height*1/5)  and table_1["bbox"][0][1] <= height*1/5):
          if ((len(table_0['data'][0]) == len(table_1['data'][0])) or (abs(abs(table_0["bbox"][0][2] - table_0["bbox"][0][0]) - abs(table_1["bbox"][0][2] - table_1["bbox"][0][0])) <= 5)): # if same no.of columns exist
        # if ((table_0["bbox"][0][3] >= (height - height*1/5)  and table_1["bbox"][0][1] <= height*1/5)  or (abs(abs(table_0["bbox"][0][2] - table_0["bbox"][0][0]) - abs(table_1["bbox"][0][2] - table_1["bbox"][0][0])) <= 5)): # if same no.of columns exist
          #Same no of colmns i.e append new data as rows
            if multi_page_tables.keys(): #if no entires are present
                existing_multi_page_tables = [v["table_no"] for k, v in multi_page_tables.items()]
                existing_multi_page_tables = [x for y in existing_multi_page_tables for x in y]
                if table_0['table_no'] not in existing_multi_page_tables: #if a no entires of the first table is present
                  multi_page_tables[i] = {'table_no' : [table_0['table_no'], table_1["table_no"]], 'page_no' : [table_0["page_no"], table_1["page_no"]], 'bbox' : [table_0["bbox"][0], table_1["bbox"][0]], 'data' :  table_0['data'] + table_0['data']}
                  i+=1
                else:
                  for key, values in multi_page_tables.items(): #if a entires of the first table is present
                    multi_page_tables[key]['page_no'] = multi_page_tables[key]['page_no'] + [table_1['page_no']]
                    multi_page_tables[key]['table_no'] = multi_page_tables[key]['table_no'] + [table_1['table_no']]
                    multi_page_tables[key]['bbox'] = multi_page_tables[key]['bbox'] + table_1["bbox"]
                    multi_page_tables[key]['data'] = multi_page_tables[key]['data'] + table_1["data"]
            else:
              multi_page_tables[i] = {'table_no' : [table_0['table_no'], table_1["table_no"]], 'page_no' : [table_0["page_no"], table_1["page_no"]], 'bbox' : [table_0["bbox"][0], table_1["bbox"][0]], 'data' :  table_0['data'] + table_0['data']}
              i+=1
              # print(multi_page_tables)

        elif (len(table_0['data']) == len(table_1['data'])) or (abs(abs(table_0["bbox"][0][3] - table_0["bbox"][0][1]) -( abs(table_1["bbox"][0][3] - table_1["bbox"][0][1]))) <=5): # if same no.of rows exist or height of table is almost same
          #Same no of rows i.e append new data as a new column
          if multi_page_tables.keys(): #if no entires are present
              existing_multi_page_tables = [v["table_no"] for k, v in multi_page_tables.items()]
              existing_multi_page_tables = [x for y in existing_multi_page_tables for x in y]
              if table_0['table_no'] not in existing_multi_page_tables: #if a no entires of the first table is present
                data = append_columns(table_0['data'], table_1['data'])
                multi_page_tables[i] = {'table_no' : [table_0['table_no'], table_1["table_no"]], 'page_no' : [table_0["page_no"], table_1["page_no"]], 'bbox' : [table_0["bbox"][0], table_1["bbox"][0]], 'data' :  data}
                i+=1

              else: #if a entires of the first table is present

                for key, values in multi_page_tables.items():
                  multi_page_tables[key]['page_no'] = multi_page_tables[key]['page_no'] + [table_1['page_no']]
                  multi_page_tables[key]['table_no'] = multi_page_tables[key]['table_no'] + [table_1['table_no']]
                  multi_page_tables[key]['bbox'] = multi_page_tables[key]['bbox'] + table_1["bbox"]
                  multi_page_tables[key]['data'] = append_columns(multi_page_tables[key]['data'], table_1['data'])
          else:
            data = append_columns(table_0['data'], table_1['data'])
            multi_page_tables[i] = {'table_no' : [table_0['table_no'], table_1["table_no"]], 'page_no' : [table_0["page_no"], table_1["page_no"]], 'bbox' : [table_0["bbox"][0], table_1["bbox"][0]], 'data' :  data}
            i+=1

  return multi_page_tables

def postprocessing(extracted_tables, doc_path):
  tables_with_table_no = add_table_no(extracted_tables)
  mulitpage_tables = detect_multipage_table(tables_with_table_no, doc_path)
  final_extracted_tables = []
  # print("detect_multipage_table", mulitpage_tables)
  tables_to_be_removed = [i["table_no"]  for i in mulitpage_tables.values()]
  # print("tables_to_be_removed", tables_to_be_removed)
  tables_to_be_removed = [x for y in tables_to_be_removed for x in y]
  tables_to_be_removed = sorted(set(tables_to_be_removed))


  for i, table in enumerate(tables_with_table_no):
    if table['table_no'] not in tables_to_be_removed:
      table["page_no"] = [table["page_no"]]
      final_extracted_tables.append(table)

  final_extracted_tables = final_extracted_tables + [table for table in mulitpage_tables.values()]

  for x in final_extracted_tables:
    del x["table_no"]

  final_extracted_tables = sorted(final_extracted_tables, key=lambda x: x['page_no'][0])

  for table in final_extracted_tables:
    df = pd.DataFrame(table["data"])
    df = df.loc[:, (df != ' ').any()]
    table["data"] = df.values.tolist()

  return final_extracted_tables
  # print(len(final_extracted_tables), final_extracted_tables)

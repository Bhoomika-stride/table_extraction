from src.TableExtraction import TableExtraction
from src.Utils import postprocessing
import time
import os

if __name__ == "__main__":

  start_time = time.time()
  doc_path = "test/tables_sample_pdf.pdf"
  table_extractor = TableExtraction(doc_path)
  extracted_tables = table_extractor.execute()

  final_extracted_tables = postprocessing(extracted_tables, doc_path)
  end_time = time.time()
  del table_extractor
  
  if os.path.isfile("page.png"):
    os.remove("page.png")

  if os.path.isfile("page_test.png"):
    os.remove("page_test.png")

  print(final_extracted_tables)
  print("Time Taken: ", (end_time - start_time))
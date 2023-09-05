from src.TableExtraction import TableExtraction
from src.Utils import postprocessing

if __name__ == "__main__":

  doc_path = "test/tables_sample_pdf.pdf"
  table_extractor = TableExtraction(doc_path)
  extracted_tables = table_extractor.execute()

  final_extracted_tables = postprocessing(extracted_tables, doc_path)

  del table_extractor

  print(final_extracted_tables)
import os
from PyPDF2 import PdfReader


def extract_text_from_pdfs(pdf_folder):
    texts = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            reader = PdfReader(pdf_path)
            pdf_text = ""
            for page in reader.pages:
                pdf_text += page.extract_text()
            texts.append(pdf_text)
    return texts

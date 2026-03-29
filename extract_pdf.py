import pypdf
import sys

def extract_text(pdf_path, txt_path):
    with open(pdf_path, 'rb') as f:
        reader = pypdf.PdfReader(f)
        text = ''
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + '\n'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)

extract_text('main.pdf', 'main_extracted.txt')

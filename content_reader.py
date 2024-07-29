import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path, txt_path):
    """
    Extracts text from a PDF file and writes it to a .txt file.
    
    Args:
        pdf_path (str): The path to the PDF file.
        txt_path (str): The path to the output .txt file.
    """
    # Ensure paths are valid
    if not pdf_path or not txt_path:
        raise ValueError("Both pdf_path and txt_path must be provided and cannot be empty.")
    
    # Ensure the directory path exists
    directory = os.path.dirname(txt_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Open the PDF file
    try:
        pdf_document = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF file: {e}")
        return
    
    # Initialize an empty string to collect all text
    full_text = ""

    # Loop through each page in the PDF
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        full_text += page.get_text()
    
    # Close the PDF document
    pdf_document.close()

    # Write the extracted text to the .txt file
    try:
        with open(txt_path, "w", encoding="utf-8") as text_file:
            text_file.write(full_text)
        print(f"Text successfully extracted and saved to {txt_path}")
    except Exception as e:
        print(f"Error writing to text file: {e}")

# Example usage
pdf_path = 'MoU Sample(1).pdf'
txt_path = 'txt_path/Output.txt'

# Extract text from PDF and save to .txt file
extract_text_from_pdf(pdf_path, txt_path)

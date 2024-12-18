import pytesseract
from PIL import Image

# Set the path to tesseract executable if not in PATH
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load an image file
image = Image.open('sample_image.png')  # Replace with your image path

# Perform OCR
text = pytesseract.image_to_string(image)
print("Extracted Text:", text)

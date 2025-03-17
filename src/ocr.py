import pytesseract
from PIL import Image

def extract_text(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(
        image, config="--psm 4 --oem 3"
    )
    return text

if __name__ == "__main__":
    print(extract_text("/Users/adityadamerla/Documents/MIE/model_test/data/sample_doc.png"))

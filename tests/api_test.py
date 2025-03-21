#!/usr/bin/env python3
import easyocr
import requests
import json
import sys

# BlueHive Completion API endpoint (text-based)
API_ENDPOINT = "https://ai.bluehive.com/api/v1/completion"

# Your API secret key (keep it secure)
SECRET_KEY = "BHSK-sandbox-d6TDZyX2PAVq6qL3IdMX8n8sA7bXe8DM_RWOq-8j"

# Headers for the completion API request.
headers = {
    "Authorization": f"Bearer {SECRET_KEY}",
    "Content-Type": "application/json"
}

def perform_ocr_easyocr(image_path: str) -> str:
    """
    Uses EasyOCR to extract text from the given image file.
    """
    try:
        # Initialize the reader for English (set gpu=True if you have GPU support)
        reader = easyocr.Reader(['en'], gpu=False)
        # Read the image; detail=0 returns only the text strings
        results = reader.readtext(image_path, detail=0)
        # Combine results into one string (each result on a new line)
        extracted_text = "\n".join(results)
        return extracted_text
    except Exception as e:
        print(f"Error during OCR on '{image_path}': {e}")
        return None

def get_document_details(ocr_text: str, user_question: str):
    """
    Combines the OCR text with a user question into a prompt and sends it to
    the BlueHive completion API to analyze the document.
    """
    prompt = (
        "Below is the OCR text extracted from a medical document:\n\n"
        f"{ocr_text}\n\n"
        "Based on the above text, please identify the type of document and extract details "
        "such as patient information, prescribed medications, prescription date, and any other relevant details.\n"
        f"User question: {user_question}"
    )
    
    payload = {
        "prompt": prompt,
        "systemMessage": "You are a helpful AI that analyzes medical documents and extracts key details."
    }
    
    response = requests.post(API_ENDPOINT, headers=headers, json=payload)
    if response.status_code == 200:
        try:
            return response.json()
        except Exception as e:
            print("Error parsing JSON response:", e)
            return response.text
    else:
        print(f"Request failed with status code {response.status_code}")
        print("Response:", response.text)
        return None

def main():
    # Path to your image file
    image_path = "data/sample_doc.png"
    # User question to be appended to the prompt.
    user_question = (
        "Can you provide the document type and extract details such as the patient's name, "
        "medications prescribed, and the prescription date?"
    )
    
    print("Performing OCR using EasyOCR...")
    ocr_text = perform_ocr_easyocr(image_path)
    if not ocr_text:
        print("OCR failed. Exiting.")
        sys.exit(1)
    
    print("\nExtracted OCR text:")
    print(ocr_text)
    
    print("\nSending combined prompt to BlueHive API for document analysis...")
    result = get_document_details(ocr_text, user_question)
    if result:
        print("\n--- AI API Response ---")
        print(json.dumps(result, indent=2))
    else:
        print("No result returned.")

if __name__ == "__main__":
    main()

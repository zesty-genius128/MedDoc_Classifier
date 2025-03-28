#!/usr/bin/env python3
import easyocr
import requests
import json
import sys
import cv2  # Still needed for preprocessing if using EasyOCR, or image loading
import numpy as np # Still needed for preprocessing if using EasyOCR
from thefuzz import process, fuzz 
import base64 # Added for OpenAI image encoding
import openai # Added for OpenAI API access - install using: pip install openai
import os # Recommended for API key handling
from typing import Union
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
# Select the OCR method to use: "easyocr" or "openai"
OCR_METHOD = "openai" # <--- CHANGE THIS TO SWITCH BETWEEN OCR METHODS

# BlueHive Completion API endpoint (text-based)
API_ENDPOINT = "https://ai.bluehive.com/api/v1/completion"

SECRET_KEY = os.environ.get("BLUEHIVE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Headers for the BlueHive completion API request.
headers = {
    "Authorization": f"Bearer {SECRET_KEY}",
    "Content-Type": "application/json"
}

if OCR_METHOD == "openai" and OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
    print("Warning: OpenAI API Key not set. Please replace 'YOUR_OPENAI_API_KEY_HERE'.")
    sys.exit("Exiting: OpenAI API Key is required.")
    openai_client = None # Set client to None if key is missing
else:
     try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
     except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        openai_client = None


DOMAIN_SPECIFIC_DICTIONARY = [
    "Patient Name", "Date of Birth", "DOB", "Address", "Medical Record Number", "MRN",
    "Prescription", "Medication", "Dosage", "Frequency", "Refills", "Signature",
    "Doctor", "Physician", "Clinic", "Hospital", "Diagnosis", "Symptoms",
    "Take", "tablet", "capsule", "daily", "twice", "three times", "as needed", "PRN",
    "mg", "ml", # Common units
    "Aspirin", "Lisinopril", "Metformin", "Simvastatin", "Amoxicillin", 
]
FUZZY_MATCH_THRESHOLD = 88 


# === START: EasyOCR Specific Code (Commented out if OCR_METHOD != 'easyocr') ===
# def preprocess_image(image_path: str):
#     """
#     Loads an image and applies preprocessing steps to improve OCR accuracy.
#     Steps: Grayscale, Contrast Enhancement (CLAHE), Sharpening, Thresholding.
#     """
#     try:
#         img = cv2.imread(image_path)
#         if img is None:
#             print(f"Error: Unable to load image at '{image_path}'")
#             return None
#             
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         alpha = 1.3 # Contrast control
#         beta = 10    # Brightness control
#         contrast_enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
#         processed_img = contrast_enhanced 
#         _, thresh = cv2.threshold(processed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         
#         print(f"Applied preprocessing: Grayscale, Contrast Adj, Thresholding to {image_path}")
#         return thresh # Return the final preprocessed image data (NumPy array)
# 
#     except Exception as e:
#         print(f"Error during image preprocessing for '{image_path}': {e}")
#         return None
# 
# 
# def perform_ocr_easyocr(image_data) -> str: 
#     """
#     Uses EasyOCR to extract text from the given preprocessed image data.
#     """
#     if image_data is None:
#         print("Error: No image data provided for EasyOCR.")
#         return None
#         
#     try:
#         # Initialize the reader each time for simplicity, consider reusing if performance is critical
#         reader = easyocr.Reader(['en'], gpu=False) 
#         results = reader.readtext(image_data, detail=0) 
#         extracted_text = "\n".join(results)
#         return extracted_text
#     except Exception as e:
#         print(f"Error during EasyOCR processing: {e}")
#         return None
# === END: EasyOCR Specific Code ===


# === START: OpenAI OCR Specific Code (Commented out if OCR_METHOD != 'openai') ===
def encode_image_to_base64(image_path: str) -> Union[str, None]:
    """Reads an image file and encodes it into base64."""
    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            # Determine MIME type (basic implementation)
            if image_path.lower().endswith(".png"):
                mime_type = "image/png"
            elif image_path.lower().endswith(".jpg") or image_path.lower().endswith(".jpeg"):
                mime_type = "image/jpeg"
            elif image_path.lower().endswith(".webp"):
                 mime_type = "image/webp"
            else:
                # Default or raise error if type is unsupported by OpenAI
                mime_type = "image/jpeg" 
                print(f"Warning: Unknown image type for {image_path}, assuming JPEG.")
                
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            return f"data:{mime_type};base64,{base64_image}"
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'")
        return None
    except Exception as e:
        print(f"Error encoding image '{image_path}': {e}")
        return None

def perform_ocr_openai(image_path: str) -> Union[str, None]:
    """
    Uses OpenAI's GPT-4 Vision model to extract text from the given image file.
    """
    if not openai_client:
        print("Error: OpenAI client not initialized (check API key). Cannot perform OpenAI OCR.")
        return None

    print(f"Encoding image '{image_path}' for OpenAI API...")
    base64_image_data = encode_image_to_base64(image_path)
    
    if not base64_image_data:
        return None # Error message already printed by encode function

    print("Sending request to OpenAI GPT-4 Vision for OCR...")
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Perform OCR on this image. Extract all text exactly as it appears, preserving line breaks and structure where possible."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base64_image_data,
                                "detail": "high" # Use high detail for potentially better OCR on dense text
                            },
                        },
                    ],
                }
            ],
            max_tokens=2000, # Adjust based on expected text length
            temperature=0.1 # Lower temperature for more deterministic OCR output
        )
        
        # print(response) # Uncomment for full response debugging
        
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            extracted_text = response.choices[0].message.content
            print("OpenAI OCR successful.")
            return extracted_text
        else:
            print("Error: OpenAI response did not contain the expected text content.")
            print("Response:", response)
            return None

    except openai.APIError as e:
        print(f"OpenAI API returned an API Error: {e}")
        return None
    except openai.APIConnectionError as e:
        print(f"Failed to connect to OpenAI API: {e}")
        return None
    except openai.RateLimitError as e:
        print(f"OpenAI API request exceeded rate limit: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during OpenAI OCR: {e}")
        return None
# === END: OpenAI OCR Specific Code ===


# === Shared Post-processing and API Call Logic ===
def apply_fuzzy_matching(text: str, dictionary: list, threshold: int) -> str:
    """
    Experimental: Attempts to correct common OCR errors using fuzzy matching
    against a domain-specific dictionary. (Works on output from any OCR method)
    """
    # (Implementation remains the same as previous version)
    if not text:
        return ""
        
    print(f"Applying fuzzy matching (threshold={threshold})...")
    corrected_lines = []
    lines = text.split('\n')
    
    corrections_made = 0
    for line in lines:
        words = line.split()
        corrected_words = []
        for word in words:
            if len(word) < 3 or word.isdigit():
                corrected_words.append(word)
                continue
                
            best_match, score = process.extractOne(word, dictionary, scorer=fuzz.ratio)
            
            if score >= threshold:
                if word != best_match:
                    corrected_words.append(best_match)
                    corrections_made += 1
                else:
                    corrected_words.append(word) 
            else:
                corrected_words.append(word) 
                
        corrected_lines.append(" ".join(corrected_words))
        
    print(f"Fuzzy matching complete. Made {corrections_made} potential corrections.")
    return "\n".join(corrected_lines)


def get_document_details(ocr_text: str, user_question: str):
    """
    Combines the (potentially corrected) OCR text with a user question into a prompt 
    and sends it to the BlueHive completion API to analyze the document. 
    (Works with text from any OCR method)
    """
    # (Implementation remains the same as previous version)
    prompt = (
        "Below is the OCR text extracted from a medical document (potentially with minor corrections applied):\n\n"
        f"{ocr_text}\n\n"
        "Based on the above text, please identify the type of document and extract details "
        "such as patient information, prescribed medications, prescription date, and any other relevant details.\n"
        f"User question: {user_question}"
    )
    
    payload = {
        "prompt": prompt,
        "systemMessage": "You are a helpful AI that analyzes medical documents and extracts key details."
    }
    
    try:
        response = requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=60) 
        response.raise_for_status() 
        
        try:
            return response.json()
        except json.JSONDecodeError: 
            print("Error parsing JSON response from BlueHive. Response text:")
            print(response.text)
            return {"error": "Failed to parse JSON response", "raw_response": response.text}

    except requests.exceptions.RequestException as e:
        print(f"BlueHive API Request failed: {e}")
        return None


def main():
    # Path to your image file
    image_path = "data/sample_doc.png" 
    # User question for BlueHive API
    user_question = (
        "Can you provide the document type and extract details such as the patient's name, "
        "medications prescribed, and the prescription date?"
    )
    
    print(f"--- Starting Analysis for {image_path} ---")
    print(f"--- Using OCR Method: {OCR_METHOD} ---")

    raw_ocr_text = None

    # --- OCR Step ---
    if OCR_METHOD == "easyocr":
        print("\nStep 1a: Preprocessing image for EasyOCR...")
        # # # UNCOMMENT THE FOLLOWING BLOCK TO USE EASYOCR # # #
        # preprocessed_image_data = preprocess_image(image_path)
        # if preprocessed_image_data is None:
        #     print("Image preprocessing failed. Exiting.")
        #     sys.exit(1)
            
        # # Optional: Save preprocessed image for debugging
        # # cv2.imwrite("data/preprocessed_output.png", preprocessed_image_data)
        # # print("Saved preprocessed image to data/preprocessed_output.png")
            
        # print("\nStep 1b: Performing OCR using EasyOCR on preprocessed image...")
        # raw_ocr_text = perform_ocr_easyocr(preprocessed_image_data)
        # # # END OF EASYOCR BLOCK # # #
        
        # --- Placeholder if EasyOCR code is commented out ---
        if raw_ocr_text is None:
             print("EasyOCR code is currently commented out. Set OCR_METHOD='easyocr' and uncomment the block above.")
             sys.exit(1)
             
    elif OCR_METHOD == "openai":
        print("\nStep 1: Performing OCR using OpenAI GPT-4 Vision...")
        # Make sure the function and client are available
        if 'perform_ocr_openai' in globals() and openai_client:
             raw_ocr_text = perform_ocr_openai(image_path)
        else:
            print("OpenAI OCR function not available or client not initialized. Check code and API key.")
            sys.exit(1)
            
    else:
        print(f"Error: Unknown OCR_METHOD configured: '{OCR_METHOD}'")
        sys.exit(1)

    # --- Post-OCR Steps (Common to all methods) ---
    if not raw_ocr_text:
        print("OCR failed or returned no text. Exiting.")
        sys.exit(1)
    
    print("\n--- Raw Extracted OCR text ---")
    print(raw_ocr_text)
    print("-" * 20)
    
    print("\nStep 2: Applying experimental fuzzy matching...")
    # Apply fuzzy matching regardless of the OCR source
    corrected_ocr_text = apply_fuzzy_matching(raw_ocr_text, DOMAIN_SPECIFIC_DICTIONARY, FUZZY_MATCH_THRESHOLD)
    
    print("\n--- Corrected OCR text (after fuzzy matching) ---")
    print(corrected_ocr_text)
    print("-" * 20)
    
    # Decide which text to send to the API - using corrected for now
    text_to_send = corrected_ocr_text 
    
    print("\nStep 3: Sending combined prompt to BlueHive API for document analysis...")
    result = get_document_details(text_to_send, user_question)
    
    if result:
        print("\n--- BlueHive AI API Response ---")
        if isinstance(result, dict) and "error" in result:
             print(f"API Error: {result['error']}")
             if "raw_response" in result:
                 print(f"Raw response snippet: {result['raw_response'][:500]}...") 
        else:
             print(json.dumps(result, indent=2))
    else:
        print("No valid result returned from BlueHive API.")
        
    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    # Check for necessary libraries based on chosen method
    try:
        from thefuzz import fuzz # Needed for fuzzy matching always
        if OCR_METHOD == "easyocr":
            import easyocr 
            import cv2
            import numpy
        elif OCR_METHOD == "openai":
            import openai
            import base64
    except ImportError as e:
        print(f"Error: Missing required library for selected OCR_METHOD ('{OCR_METHOD}'). {e}")
        print("Please install necessary libraries:")
        print("For 'easyocr': pip install easyocr opencv-python numpy python-levenshtein thefuzz")
        print("For 'openai': pip install openai python-levenshtein thefuzz")
        # No need to exit here if libraries for the *other* method are missing
        # sys.exit(1) 

    # Check for OpenAI API key if that method is selected
    if OCR_METHOD == "openai" and (not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE"):
         print("\nCRITICAL WARNING: OpenAI API Key is missing or not set.")
         print("Please set the OPENAI_API_KEY variable in the script or use environment variables.")
         # Decide if you want to exit if the key is missing for the selected method
         sys.exit("Exiting because required OpenAI API Key is not configured.")

    main()
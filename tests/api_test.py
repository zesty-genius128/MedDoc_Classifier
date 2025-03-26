#!/usr/bin/env python3
import easyocr
import requests
import json
import sys
import cv2  # added for image preprocessing
import numpy as np # added for sharpening kernel
# added for fuzzy matching - install using: pip install thefuzz python-levenshtein
from thefuzz import process, fuzz 

# BlueHive Completion API endpoint (text-based)
API_ENDPOINT = "https://ai.bluehive.com/api/v1/completion"

# Your API secret key (keep it secure)
SECRET_KEY = "BHSK-sandbox-d6TDZyX2PAVq6qL3IdMX8n8sA7bXe8DM_RWOq-8j" # Keep using the sandbox key for now

# Headers for the completion API request.
headers = {
    "Authorization": f"Bearer {SECRET_KEY}",
    "Content-Type": "application/json"
}


# --- Fuzzy Matching Setup (test) ---
# dictionary of common terms likely to appear or be misread in the documents should be 
# expanded post checks based on OCR errors
DOMAIN_SPECIFIC_DICTIONARY = [
    "Patient Name", "Date of Birth", "DOB", "Address", "Medical Record Number", "MRN",
    "Prescription", "Medication", "Dosage", "Frequency", "Refills", "Signature",
    "Doctor", "Physician", "Clinic", "Hospital", "Diagnosis", "Symptoms",
    "Take", "tablet", "capsule", "daily", "twice", "three times", "as needed", "PRN",
    "mg", "ml", # Common units
    # specific medication names add them here if errors occur often
    "Aspirin", "Lisinopril", "Metformin", "Simvastatin", "Amoxicillin", 
    # add common misread examples when possible like rn vs m, 0 vs o, etc.
    "rn", "m", "0", "o", "1", "l", "I", "i", "Z", "S", "5", "6", "8", "B", "D", "O", "Q", "T"
]
FUZZY_MATCH_THRESHOLD = 88 # Score out of 100 for replacement


def preprocess_image(image_path: str):
    """
    Loads an image and applies preprocessing steps to improve OCR accuracy.
    Steps: Grayscale, Contrast Enhancement (CLAHE), Sharpening, Thresholding.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Unable to load image at '{image_path}'")
            return None
            
        # 1. Convert to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Contrast Enhancement (using CLAHE)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # contrast_enhanced = clahe.apply(gray) 
        # trying a simple brightness/contrast adjustment for now
        alpha = 1.3 # contrast control (1.0-3.0)
        beta = 10    # brightness control (0-100)
        contrast_enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

        # 3. Sharpening
        # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) # Basic sharpening kernel
        # sharpened = cv2.filter2D(contrast_enhanced, -1, kernel)
        # sharpening can sometimes worsen noise more testing needed for this using contrast for now.
        processed_img = contrast_enhanced # using contrast-enhanced image directly for now

        # 4. Thresholding (Otsu's method automatically finds a good threshold)
        _, thresh = cv2.threshold(processed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        print(f"Applied preprocessing: Grayscale, Contrast Adj, Thresholding to {image_path}")
        return thresh # returning the final preprocessed image data (NumPy array)

    except Exception as e:
        print(f"Error during image preprocessing for '{image_path}': {e}")
        return None


def perform_ocr_easyocr(image_data) -> str: # Changed input to accept image data
    """
    Uses EasyOCR to extract text from the given preprocessed image data.
    """
    # old version took image_path:
    # reader = easyocr.Reader(['en'], gpu=False)
    # results = reader.readtext(image_path, detail=0)
    
    if image_data is None:
        print("Error: No image data provided for OCR.")
        return None
        
    try:
        # Initialize the reader for English (set gpu=True if you have GPU support)
        # Reuse reader if processing multiple images, but initialize here for simplicity now.
        reader = easyocr.Reader(['en'], gpu=True) 
        
        # Read the image data directly; detail=0 returns only the text strings
        results = reader.readtext(image_data, detail=0) 
        
        # Combine results into one string (each result on a new line)
        extracted_text = "\n".join(results)
        
        # --- Debug: Print raw results length
        # print(f"EasyOCR found {len(results)} text blocks.")
        
        return extracted_text
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        return None


def apply_fuzzy_matching(text: str, dictionary: list, threshold: int) -> str:
    """
    Experimental: Attempts to correct common OCR errors using fuzzy matching
    against a domain-specific dictionary.
    """
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
            # Skip very short words or numbers for now, adjust as needed
            if len(word) < 3 or word.isdigit():
                corrected_words.append(word)
                continue
                
            # Find the best match in the dictionary
            best_match, score = process.extractOne(word, dictionary, scorer=fuzz.ratio)
            
            # If score is above threshold, replace the word
            if score >= threshold:
                if word != best_match:
                    # print(f"Correction: '{word}' -> '{best_match}' (Score: {score})") # Debug output
                    corrected_words.append(best_match)
                    corrections_made += 1
                else:
                    corrected_words.append(word) # Match is the same word
            else:
                corrected_words.append(word) # No good match found
                
        corrected_lines.append(" ".join(corrected_words))
        
    print(f"Fuzzy matching complete. Made {corrections_made} potential corrections.")
    return "\n".join(corrected_lines)


def get_document_details(ocr_text: str, user_question: str):
    """
    Combines the (potentially corrected) OCR text with a user question into a prompt 
    and sends it to the BlueHive completion API to analyze the document.
    """
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
        # Consider adding parameters like 'max_tokens', 'temperature' if needed
    }
    
    try:
        response = requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=60) # Added timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        try:
            return response.json()
        except json.JSONDecodeError: # Changed from generic Exception
            print("Error parsing JSON response. Response text:")
            print(response.text)
            return {"error": "Failed to parse JSON response", "raw_response": response.text}

    except requests.exceptions.RequestException as e:
        print(f"API Request failed: {e}")
        # print("Response:", response.text if 'response' in locals() else "No response object") # Check if response exists
        return None


def main():
    # Path to your image file
    image_path = "data/sample_doc.png" 
    # User question to be appended to the prompt.
    user_question = (
        "Can you provide the document type and extract details such as the patient's name, "
        "medications prescribed, and the prescription date?"
    )
    
    print(f"--- Starting Analysis for {image_path} ---")
    
    print("\nStep 1: Preprocessing image...")
    preprocessed_image_data = preprocess_image(image_path)
    
    if preprocessed_image_data is None:
        print("Image preprocessing failed. Exiting.")
        sys.exit(1)
        
    # Optional: Save preprocessed image for debugging
    # cv2.imwrite("data/preprocessed_output.png", preprocessed_image_data)
    # print("Saved preprocessed image to data/preprocessed_output.png")
        
    print("\nStep 2: Performing OCR using EasyOCR on preprocessed image...")
    raw_ocr_text = perform_ocr_easyocr(preprocessed_image_data)
    
    if not raw_ocr_text:
        print("OCR failed. Exiting.")
        sys.exit(1)
    
    print("\n--- Raw Extracted OCR text ---")
    print(raw_ocr_text)
    print("-" * 20)
    
    print("\nStep 3: Applying experimental fuzzy matching...")
    corrected_ocr_text = apply_fuzzy_matching(raw_ocr_text, DOMAIN_SPECIFIC_DICTIONARY, FUZZY_MATCH_THRESHOLD)
    
    print("\n--- Corrected OCR text (after fuzzy matching) ---")
    print(corrected_ocr_text)
    print("-" * 20)
    
    # Decide which text to send to the API - using corrected for now
    text_to_send = corrected_ocr_text 
    
    print("\nStep 4: Sending combined prompt to BlueHive API for document analysis...")
    result = get_document_details(text_to_send, user_question)
    
    if result:
        print("\n--- AI API Response ---")
        # Check if the result contains an error key we added
        if isinstance(result, dict) and "error" in result:
             print(f"API Error: {result['error']}")
             if "raw_response" in result:
                 print(f"Raw response snippet: {result['raw_response'][:500]}...") # Print first 500 chars
        else:
             print(json.dumps(result, indent=2))
    else:
        print("No valid result returned from API.")
        
    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    # Ensure necessary libraries are installed
    try:
        import cv2
        from thefuzz import fuzz 
    except ImportError as e:
        print(f"Error: Missing required library. {e}")
        print("Please install using: pip install opencv-python python-levenshtein thefuzz")
        sys.exit(1)
        
    main()
#!/usr/bin/env python3
import easyocr
import requests
import json
import sys
import cv2
import numpy as np
from thefuzz import process, fuzz
import base64
import openai
import os
import io  # Moved io import to top
from typing import Union
from dotenv import load_dotenv
import argparse
import google.generativeai as genai
from PIL import Image

load_dotenv()


BLUEHIVE_API_ENDPOINT = "https://ai.bluehive.com/api/v1/completion"

# --- API KEYS - Loaded from Environment ---
SECRET_KEY = os.environ.get("BLUEHIVE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")


openai_client = None
gemini_model = None

DOMAIN_SPECIFIC_DICTIONARY = [
    "Patient Name",
    "Date of Birth",
    "DOB",
    "Address",
    "Medical Record Number",
    "MRN",
    "Prescription",
    "Medication",
    "Dosage",
    "Frequency",
    "Refills",
    "Signature",
    "Doctor",
    "Physician",
    "Clinic",
    "Hospital",
    "Diagnosis",
    "Symptoms",
    "Take",
    "tablet",
    "capsule",
    "daily",
    "twice",
    "three times",
    "as needed",
    "PRN",
    "mg",
    "ml",
    "Aspirin",
    "Lisinopril",
    "Metformin",
    "Simvastatin",
    "Amoxicillin",
]
FUZZY_MATCH_THRESHOLD = 88


# === START: EasyOCR Specific Code ===
def preprocess_image(image_path: str) -> Union[np.ndarray, None]:
    """
    Loads an image and applies preprocessing steps for EasyOCR.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Unable to load image at '{image_path}'")
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        alpha = 1.3  # Contrast control
        beta = 10  # Brightness control
        contrast_enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        processed_img = contrast_enhanced
        _, thresh = cv2.threshold(
            processed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Wrapped long print statement
        print(
            "Applied preprocessing for EasyOCR: Grayscale, Contrast Adj, "
            f"Thresholding to {image_path}"
        )
        return thresh  # Return the final preprocessed image data (NumPy array)
    except Exception as e:
        # Wrapped long print statement
        print(
            "Error during EasyOCR image preprocessing for "
            f"'{image_path}': {e}"
        )
        return None


def perform_ocr_easyocr(image_data: np.ndarray) -> Union[str, None]:
    """
    Uses EasyOCR to extract text from the given preprocessed image data.
    """
    if image_data is None:
        print("Error: No image data provided for EasyOCR.")
        return None

    try:
        print("Initializing EasyOCR Reader...")
        # Consider initializing the reader only once
        reader = easyocr.Reader(["en"], gpu=False)
        print("Performing EasyOCR...")
        results = reader.readtext(image_data, detail=0)
        extracted_text = "\n".join(results)
        print("EasyOCR successful.")
        return extracted_text
    except Exception as e:
        print(f"Error during EasyOCR processing: {e}")  # Relatively short line
        return None


# === END: EasyOCR Specific Code ===


# === START: OpenAI OCR Specific Code ===
def encode_image_to_base64(image_path: str) -> Union[str, None]:
    """Reads an image file and encodes it into base64 data URI."""
    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            if image_path.lower().endswith(".png"):
                mime_type = "image/png"
            elif image_path.lower().endswith((".jpg", ".jpeg")):
                mime_type = "image/jpeg"
            elif image_path.lower().endswith(".webp"):
                mime_type = "image/webp"
            else:
                mime_type = "image/jpeg"
                # Wrapped long print statement
                print(
                    f"Warning: Unknown image type for {image_path}, "
                    "assuming JPEG."
                )
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            return f"data:{mime_type};base64,{base64_image}"
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'")
        return None
    except Exception as e:
        print(f"Error encoding image '{image_path}': {e}")
        return None


def perform_ocr_openai(image_path: str, client) -> Union[str, None]:
    """
    Uses OpenAI's GPT-4 Vision model to extract text from the image file.
    Accepts the initialized OpenAI client.
    """
    if not client:
        print("Error: OpenAI client not initialized. Cannot perform OpenAI OCR.")
        return None

    print(f"Encoding image '{image_path}' for OpenAI API...")
    base64_image_data = encode_image_to_base64(image_path)
    if not base64_image_data:
        return None

    print("Sending request to OpenAI GPT-4 Vision for OCR...")
    # Wrapped long string in prompt
    ocr_prompt_text = (
        "Perform OCR on this image. Extract all text exactly as it appears, "
        "preserving line breaks and structure where possible."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ocr_prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": base64_image_data, "detail": "high"},
                        },
                    ],
                }
            ],
            max_tokens=2000,
            temperature=0.1,
        )
        if (
            response.choices
            and response.choices[0].message
            and response.choices[0].message.content
        ):
            extracted_text = response.choices[0].message.content
            print("OpenAI OCR successful.")
            if extracted_text.startswith("```") and extracted_text.endswith("```"):
                extracted_text = extracted_text[3:-3].strip()
            return extracted_text
        else:
            # Wrapped long print statement
            print("Error: OpenAI response did not contain expected text.")
            print(
                "Response Choice:",
                response.choices[0] if response.choices else "No choices",
            )
            return None
    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")
        return None
    except openai.APIConnectionError as e:
        print(f"OpenAI Connection Error: {e}")
        return None
    except openai.RateLimitError as e:
        print(f"OpenAI Rate Limit Error: {e}")
        return None
    except openai.AuthenticationError as e:
        print(f"OpenAI Authentication Error: {e}")
        return None
    except openai.InvalidRequestError as e:
        print(f"OpenAI Invalid Request Error: {e}")
        return None
    except Exception as e:
        # Wrapped long print statement
        print(f"An unexpected error occurred during OpenAI OCR: {e}")
        return None


# === END: OpenAI OCR Specific Code ===


# === START: Gemini OCR Specific Code ===
def get_image_parts_for_gemini(image_path: str) -> Union[dict, None]:
    """Reads an image file and prepares it for the Gemini API."""
    try:
        img = Image.open(image_path)
        mime_type = Image.MIME.get(img.format)
        if not mime_type:
            if img.format == "JPEG": mime_type = "image/jpeg"
            elif img.format == "PNG": mime_type = "image/png"
            elif img.format == "WEBP": mime_type = "image/webp"
            else:
                print(f"Error: Unsupported image format for Gemini: {img.format}")
                return None
        # Use io.BytesIO for getting bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=img.format)
        img_bytes = img_byte_arr.getvalue()
        print(f"Image read successfully for Gemini ({mime_type}).")
        return {"mime_type": mime_type, "data": img_bytes}

    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'")
        return None
    except Exception as e:
        # Wrapped long print statement
        print(f"Error reading image for Gemini '{image_path}': {e}")
        return None


def perform_ocr_gemini(image_path: str, model) -> Union[str, None]:
    """
    Uses Google's Gemini Vision model to extract text from the image file.
    Accepts the initialized Gemini model instance.
    """
    if not model:
        print("Error: Gemini model not initialized. Cannot perform Gemini OCR.")
        return None

    print(f"Preparing image '{image_path}' for Gemini API...")
    image_parts = get_image_parts_for_gemini(image_path)
    if not image_parts:
        return None

    # Wrapped long string in prompt
    ocr_prompt_text = (
        "Perform OCR on this image. Extract all text exactly as it appears, "
        "preserving line breaks and structure where possible."
    )
    prompt_parts = [
        ocr_prompt_text,
        image_parts,
    ]

    print("Sending request to Google Gemini Vision for OCR...")
    try:
        response = model.generate_content(prompt_parts)

        if not response.candidates:
            print("Error: Gemini response was blocked or empty.")
            # Wrapped potentially long print
            print("Prompt Feedback:", response.prompt_feedback)
            return None

        extracted_text = "".join(part.text for part in response.parts)

        if extracted_text:
            print("Gemini OCR successful.")
            return extracted_text.strip()
        else:
            print("Warning: Gemini response did not contain extractable text.")
            return ""

    except Exception as e:
        print(f"An error occurred during Gemini OCR: {e}")
        return None


# === END: Gemini OCR Specific Code ===


# === Shared Post-processing and API Call Logic ===
def apply_fuzzy_matching(
    text: str, dictionary: list, threshold: int
) -> Union[str, None]:
    """
    Experimental: Attempts to correct common OCR errors using fuzzy matching.
    """
    if not text:
        return None
    print(f"Applying fuzzy matching (threshold={threshold})...")
    corrected_lines = []
    lines = text.split("\n")
    corrections_made = 0
    for line in lines:
        words = line.split()
        corrected_words = []
        for word in words:
            if len(word) < 3 or word.isdigit():
                corrected_words.append(word)
                continue
            best_match, score = process.extractOne(
                word, dictionary, scorer=fuzz.ratio
            )
            if score >= threshold and word != best_match:
                corrected_words.append(best_match)
                corrections_made += 1
            else:
                corrected_words.append(word)
        corrected_lines.append(" ".join(corrected_words))
    print(f"Fuzzy matching complete. Made {corrections_made} potential corrections.")
    return "\n".join(corrected_lines)


def get_document_details(
    ocr_text: str, user_question: str, bluehive_key: str
) -> Union[dict, None]:
    """
    Sends OCR text and question to the BlueHive completion API.
    Requires the BlueHive API key.
    """
    if not ocr_text:
        print("Error: Cannot call BlueHive API with empty OCR text.")
        return None
    if not bluehive_key:
        print("Error: BlueHive API key is missing.")
        return None

    bh_headers = {
        "Authorization": f"Bearer {bluehive_key}",
        "Content-Type": "application/json",
    }

    # Wrapped long f-string in prompt construction
    prompt = (
        "Below is the OCR text extracted from a medical document "
        "(potentially with minor corrections applied):\n\n"
        f"{ocr_text}\n\n"
        "Based on the above text, please identify the type of document and "
        "extract details such as patient information, prescribed medications, "
        "prescription date, and any other relevant details.\n"
        f"User question: {user_question}"
    )
    payload = {
        "prompt": prompt,
        "systemMessage": "You are a helpful AI that analyzes medical documents.",
    }

    try:
        response = requests.post(
            BLUEHIVE_API_ENDPOINT, headers=bh_headers, json=payload, timeout=90
        )
        response.raise_for_status()
        try:
            return response.json()
        except json.JSONDecodeError:
            # Wrapped long print statement
            print(
                "Error parsing JSON response from BlueHive. Response text:",
                response.text,
            )
            return {
                "error": "Failed to parse JSON response",
                "raw_response": response.text,
            }
    except requests.exceptions.RequestException as e:
        print(f"BlueHive API Request failed: {e}")
        return None


# === END Shared Logic ===


def main(ocr_method: str, image_path: str):
    """
    Main pipeline execution function.
    """
    # Wrapped long user question string
    user_question = (
        "Can you provide the document type and extract details such as the "
        "patient's name, medications prescribed, and the prescription date?"
    )

    print(f"--- Starting Analysis for {image_path} ---")
    print(f"--- Using OCR Method: {ocr_method} ---")

    raw_ocr_text = None
    global openai_client
    global gemini_model

    # --- OCR Step ---
    if ocr_method == "easyocr":
        print("\nStep 1a: Preprocessing image for EasyOCR...")
        preprocessed_image_data = preprocess_image(image_path)
        if preprocessed_image_data is None:
            print("Image preprocessing failed. Exiting.")
            sys.exit(1)
        print("\nStep 1b: Performing OCR using EasyOCR...")
        raw_ocr_text = perform_ocr_easyocr(preprocessed_image_data)

    elif ocr_method == "openai":
        if not openai_client and OPENAI_API_KEY:
            try:
                print("Initializing OpenAI client...")
                openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
            except Exception as e:
                print(f"Failed to initialize OpenAI client: {e}")
                sys.exit(1)
        elif not OPENAI_API_KEY:
            print("Error: OpenAI API Key missing, cannot initialize client.")
            sys.exit(1)
        print("\nStep 1: Performing OCR using OpenAI GPT-4...")
        raw_ocr_text = perform_ocr_openai(image_path, openai_client)

    elif ocr_method == "gemini":
        if not gemini_model and GOOGLE_API_KEY:
            try:
                print("Initializing Google Gemini client...")
                genai.configure(api_key=GOOGLE_API_KEY)
                gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest")
                print(f"Using Gemini model: {gemini_model.model_name}")
            except Exception as e:
                print(f"Failed to initialize Google Gemini client: {e}")
                sys.exit(1)
        elif not GOOGLE_API_KEY:
            print("Error: Google API Key missing, cannot initialize client.")
            sys.exit(1)
        print("\nStep 1: Performing OCR using Google Gemini Vision...")
        raw_ocr_text = perform_ocr_gemini(image_path, gemini_model)

    else:
        print(f"Error: Invalid OCR_METHOD configured: '{ocr_method}'")
        sys.exit(1)

    # --- Post-OCR Steps (Common to all methods) ---
    if raw_ocr_text is None:
        print("OCR failed or returned no result. Exiting.")
        sys.exit(1)
    if not raw_ocr_text:
        print("Warning: OCR returned empty text.")

    print("\n--- Raw Extracted OCR text ---")
    print(raw_ocr_text)
    print("-" * 20)

    print("\nStep 2: Applying experimental fuzzy matching...")
    corrected_ocr_text = apply_fuzzy_matching(
        raw_ocr_text, DOMAIN_SPECIFIC_DICTIONARY, FUZZY_MATCH_THRESHOLD
    )

    if corrected_ocr_text is None:
        print("Fuzzy matching returned None, using raw OCR text for API call.")
        corrected_ocr_text = raw_ocr_text

    print("\n--- Corrected OCR text (after fuzzy matching) ---")
    print(corrected_ocr_text)
    print("-" * 20)

    text_to_send = corrected_ocr_text

    print("\nStep 3: Sending combined prompt to BlueHive API for document analysis...")
    result = get_document_details(text_to_send, user_question, SECRET_KEY)

    if result:
        print("\n--- BlueHive AI API Response ---")
        if isinstance(result, dict) and "error" in result:
            print(f"API Error: {result['error']}")
            if "raw_response" in result:
                # Wrapped potentially long print
                print(
                    f"Raw response snippet: {result['raw_response'][:500]}..."
                )
        else:
            print(json.dumps(result, indent=2))
    else:
        print("No valid result returned from BlueHive API.")

    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    # --- Argument Parsing ---
    # Wrapped long description string
    parser = argparse.ArgumentParser(
        description=(
            "Perform OCR on an image using a selected method and analyze "
            "with BlueHive API."
        )
    )
    parser.add_argument("image_path", help="Path to the input image file.")
    # Wrapped long add_argument call
    parser.add_argument(
        "-m",
        "--method",
        choices=["easyocr", "openai", "gemini"],
        required=True,
        help="Select the OCR method to use.",
    )
    args = parser.parse_args()
    selected_method = args.method
    input_image_path = args.image_path

    # --- Check Core Libraries ---
    try:
        import requests
        import json
        import sys
        import os
        from typing import Union
        from dotenv import load_dotenv
        from thefuzz import fuzz  # Needed for fuzzy matching always
        import argparse
    except ImportError as e:
        print(f"Error: Missing core library: {e}")
        # Wrapped long print statement
        print(
            "Please install requirements: pip install requests "
            "python-levenshtein thefuzz python-dotenv"
        )
        sys.exit(1)

    # --- Check Method-Specific Libraries & API Keys ---
    try:
        if selected_method == "easyocr":
            print("Checking EasyOCR libraries...")
            import easyocr
            import cv2
            import numpy
            print("EasyOCR libraries OK.")
        elif selected_method == "openai":
            print("Checking OpenAI libraries...")
            import openai
            import base64
            print("OpenAI libraries OK.")
            if not OPENAI_API_KEY:
                # Wrapped long print statement
                print(
                    "\nCRITICAL WARNING: OpenAI API Key (OPENAI_API_KEY) "
                    "not found in environment variables."
                )
                sys.exit("Exiting: Required key missing.")
            print("OpenAI API Key found.")
        elif selected_method == "gemini":
            print("Checking Google Gemini libraries...")
            import google.generativeai as genai
            from PIL import Image
            # io already imported at top
            print("Google Gemini libraries OK.")
            if not GOOGLE_API_KEY:
                # Wrapped long print statement
                print(
                    "\nCRITICAL WARNING: Google API Key (GOOGLE_API_KEY) "
                    "not found in environment variables."
                )
                sys.exit("Exiting: Required key missing.")
            print("Google API Key found.")

    except ImportError as e:
        # Wrapped long print statement
        print(
            f"\nError: Missing required library for selected "
            f"OCR_METHOD ('{selected_method}'). {e}"
        )
        print("Please install necessary libraries:")
        print("For 'easyocr': pip install easyocr opencv-python numpy")
        print("For 'openai': pip install openai")
        print("For 'gemini': pip install google-generativeai Pillow")
        sys.exit(1)

    # --- Check BlueHive API Key (always needed for final step) ---
    if not SECRET_KEY:
        # Wrapped long print statement
        print(
            "\nCRITICAL WARNING: BlueHive API Key (BLUEHIVE_API_KEY) "
            "not found in environment variables."
        )
        print("Using BlueHive sandbox key as fallback.")
        SECRET_KEY = "BHSK-sandbox-d6TDZyX2PAVq6qL3IdMX8n8sA7bXe8DM_RWOq-8j"
        # Or exit:
        # sys.exit("Exiting: BlueHive API Key missing.")
    else:
        print("BlueHive API Key found.")

    # --- Execute Main Pipeline ---
    main(ocr_method=selected_method, image_path=input_image_path)
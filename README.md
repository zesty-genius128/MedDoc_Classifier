# Medical Document Classifier

## Overview

This project is designed to build a robust medical document classifier that automatically extracts relevant information from medical documents and categorizes them into predefined types (e.g., prescription, insurance claim). Our long‑term vision is to integrate state‑of‑the‑art NLP and OCR technologies to not only extract text from scanned or digital documents but also to correct and understand the extracted text by leveraging domain‑specific models.

Originally, the plan was to experiment with multiple transformer models such as BiomedBERT, ClinicalBERT, and Longformer to generate embeddings and then use a FAISS index to find the closest match among pre‑labeled documents. The output is then fed into a text completion API (currently BlueHive’s completion endpoint) for further analysis and final classification. A web service is provided via a FastAPI app, which accepts an uploaded file, extracts text using OCR, and returns the document type.

## Architecture & Components

- **Models:**  
  The project uses several transformer-based models for generating document embeddings:
  - **BiomedBERT**: (Defined in `models/biomedbert.py`)  
    Uses `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` for generating embeddings from input text.
  - **ClinicalBERT & Longformer:**  
    These models (found in similar files inside the `models` directory) are implemented similarly, with model names adjusted to suit their specific tasks.
  
- **Document Classification:**  
  The classification logic is implemented in `src/classify.py`. It:
  - Loads the chosen model.
  - Computes embeddings for incoming documents.
  - Indexes these embeddings using FAISS.
  - Performs nearest neighbor search to classify a new document based on its embedding similarity to previously added (labeled) documents.

- **API:**  
  The FastAPI application (in `api/app.py`) provides an endpoint to upload a document file. It:
  - Extracts text from the file (via a custom OCR function, see `src/ocr.py` – assumed to be implemented).
  - Uses the document classifier to determine the document type.
  - Returns the result in JSON format.

## Installation

### Prerequisites

- Python 3.9 or later.
- A virtual environment to isolate dependencies (recommended).
- Install any system-level dependencies (e.g., for OCR libraries such as Tesseract, refer to their documentation).

### Setup Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/medical-document-classifier.git
   cd medical-document-classifier
   ```

2. **Create and Activate a Virtual Environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install the Required Python Packages:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set Up Environment Variables:** 

For example, if any API keys are required (e.g., for the BlueHive API), export them:

```bash
export BLUEHIVE_API_KEY="your_bluehive_api_key"
 ```

## Usage

### Running the Classifier Locally (CLI)

You can test the document classifier via the command line by running the Python scripts. For example:

```bash
python models/biomedbert.py
This file tests the BiomedBERT embedding by printing an embedding for a test sentence.
```

### Running the API Server

The FastAPI application is defined in api/app.py. To run it locally:

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

> Once running, you can test the endpoint by navigating to:

<http://localhost:8000/docs> for interactive API documentation.

### Testing on Another User’s System

To allow other users to test this repository on their system, follow these instructions:

1. Clone the Repository:
Ensure the user has cloned the repository as described above.

2. Create a Virtual Environment:
Instruct the user to create and activate a virtual environment.

3. Install Dependencies:
The user should run the provided installation commands. Ensure that all required packages are listed in your requirements.txt.

4. Set Environment Variables:
Ask the user to set the necessary environment variables (e.g., API keys) as described in the "Installation" section.

5. Run the API:
The user can run the API server with:

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

6. And then test the endpoint (for example, using curl):

```bash
curl -X POST "<http://localhost:8000/upload/>" -F "file=@data/sample_doc.png"
```

Project Structure
```graphql
medical-document-classifier/
├── api/
│   └── app.py           # FastAPI application for handling document uploads and classification.
├── models/
│   ├── biomedbert.py    # Defines the BiomedBERT model for generating document embeddings.
│   ├── clinicalbert.py  # Similar to biomedbert.py, for clinical data.
│   └── longformer.py    # Similar to biomedbert.py, for longer documents.
├── src/
│   ├── classify.py      # Contains the DocumentClassifier class using FAISS.
│   └── ocr.py           # OCR extraction module (assumed to be implemented).
├── tests/
│   └── api_test.py      # Testing script for the API endpoints.
├── requirements.txt     # List of required Python packages.
└── README.md            # This file.
```

## Future Improvements

1. OCR Enhancements:
Incorporate advanced OCR techniques and domain-specific corrections (such as fuzzy matching with a dictionary of known medical terms) to improve text extraction accuracy.

2. Model Optimization:
Enable GPU support for transformer models and OCR libraries when available.

3. API Robustness:
Expand the API to handle additional document types and integrate error handling, logging, and user authentication.

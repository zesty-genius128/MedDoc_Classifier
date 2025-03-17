from fastapi import FastAPI, UploadFile, File
from src.ocr import extract_text
from src.classify import DocumentClassifier

app = FastAPI()
classifier = DocumentClassifier(model="longformer")  # Change model here

# Pre-populate the classifier with reference documents
classifier.add_document("This is a prescription for medication.", "prescription")
classifier.add_document("This is an insurance claim form.", "insurance claim")

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    text = extract_text(file.file)
    category = classifier.classify(text)
    return {"document_type": category}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

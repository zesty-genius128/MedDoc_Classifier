from fastapi import FastAPI, UploadFile, File
from src.ocr import extract_text
from src.classify import DocumentClassifier

app = FastAPI()
classifier = DocumentClassifier(model="clinicalbert")  # gotta change model here

classifier.add_document(
    """Patient: John Doe
    DOB: 01/01/1980
    Prescription: Amoxicillin 500mg capsules
    Dosage: Take 1 capsule three times daily for 7 days
    Prescribing Physician: Dr. Jane Smith, ABC Medical Clinic
    Instructions: Take with food. No refills.""",
    "prescription",
)

classifier.add_document(
    """Patient: John Doe
    Policy Number: XYZ123456
    Service Date: 03/15/2025
    Service Provided: Outpatient Radiology – MRI of the lumbar spine
    Diagnosis: Lower back pain
    Billed Amount: $1,200.00
    Approved Amount: $1,100.00
    Notes: Claim submitted for review. Please process for reimbursement.""",
    "insurance claim",
)


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    text = extract_text(file.file)
    category = classifier.classify(text)
    return {"document_type": category}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

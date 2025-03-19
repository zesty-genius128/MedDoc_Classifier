import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import faiss
import numpy as np
from models.biomedbert import BiomedBERT
from models.clinicalbert import ClinicalBERT
from models.longformer import Longformer

class DocumentClassifier:
    def __init__(self, model="biomedbert"):
        if model == "biomedbert":
            self.embedder = BiomedBERT()
        elif model == "clinicalbert":
            self.embedder = ClinicalBERT()
        elif model == "longformer":
            self.embedder = Longformer()
        else:
            raise ValueError("Unsupported model")

        self.index = faiss.IndexFlatL2(768)  # 768 dimensions for BERT-based models
        self.doc_embeddings = []
        self.labels = []

    def add_document(self, text, label):
        embedding = self.embedder.get_embedding(text)
        print("Embedding shape:", embedding.shape)
        self.index.add(np.array([embedding]).astype("float32"))
        self.doc_embeddings.append(embedding)
        self.labels.append(label)

    def classify(self, text):
        embedding = self.embedder.get_embedding(text)
        _, I = self.index.search(np.array([embedding]).astype("float32"), 1)
        return self.labels[I[0][0]]

if __name__ == "__main__":
    classifier = DocumentClassifier(model="biomedbert")
    classifier.add_document(
     """Patient: John Doe
        DOB: 01/01/1980
        Prescription: Amoxicillin 500mg capsules
        Dosage: Take 1 capsule three times daily for 7 days
        Prescribing Physician: Dr. Jane Smith, ABC Medical Clinic
        Instructions: Take with food. No refills.""",
        "prescription"
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
        "insurance claim"
    )
    print(classifier.classify("A medical prescription."))

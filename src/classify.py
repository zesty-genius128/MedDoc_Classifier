import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import faiss
import numpy as np
from models.biomedbert import BiomedBERT
from models.clinicalbert import ClinicalBERT
from models.longformer import Longformer

class DocumentClassifier:
    def __init__(self, model="biobert"):
        if model == "biobert":
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
    classifier = DocumentClassifier(model="biobert")
    classifier.add_document("This is a prescription for medication.", "prescription")
    classifier.add_document("This is an insurance claim form.", "insurance claim")

    print(classifier.classify("A medical prescription."))

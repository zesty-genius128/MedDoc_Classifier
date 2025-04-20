import faiss
import numpy as np

index = faiss.IndexFlatL2(768)
dummy_embedding = np.random.rand(768).astype("float32")
index.add(np.expand_dims(dummy_embedding, axis=0))
print("FAISS index size:", index.ntotal)

from transformers import AutoModel, AutoTokenizer
import torch
device = torch.device("cpu")  # Force CPU execution

class ClinicalBERT:
    def __init__(self):
        self.model_name = "emilyalsentzer/Bio_ClinicalBERT"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(device)

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

if __name__ == "__main__":
    clinical = ClinicalBERT()
    print(clinical.get_embedding("This is a test medical document."))

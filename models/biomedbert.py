from transformers import AutoModel, AutoTokenizer
import torch
device = torch.device("cpu")  #force CPU execution

class BiomedBERT:
    def __init__(self):
        self.model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(device)

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

if __name__ == "__main__":
    biomedbert = BiomedBERT()
    print(biomedbert.get_embedding("This is a test medical document."))

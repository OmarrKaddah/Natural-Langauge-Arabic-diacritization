# embedding_extraction/bert_extractor.py

import torch
from transformers import AutoTokenizer, AutoModel

class BertFeatureExtractor:
    def __init__(self, model_name="aubmindlab/bert-base-arabertv2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer + model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, sentence: str):
        """
        Returns encoded token ids, attention mask, and BERT embeddings of shape:
        (seq_len, 768)
        """
        encoded = self.tokenizer(sentence,
                                 return_tensors='pt',
                                 add_special_tokens=True)

        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            output = self.model(**encoded)

        # last_hidden_state shape: (1, seq_len, 768)
        embeddings = output.last_hidden_state.squeeze(0)

        tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])

        return tokens, embeddings

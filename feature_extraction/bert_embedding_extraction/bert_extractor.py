import torch
from transformers import AutoTokenizer, AutoModel

class BertFeatureExtractor:
    def __init__(self, model_name="aubmindlab/bert-base-arabertv2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, sentence: str):
        """
        Returns tokens, embeddings, and offsets.
        Fix: remove offset_mapping before sending inputs to BERT.
        """
        encoded = self.tokenizer(
            sentence,
            return_tensors="pt",
            add_special_tokens=True,
            return_offsets_mapping=True
        )

        # Extract offsets before sending to BERT
        offsets = encoded.pop("offset_mapping").squeeze(0).cpu()

        # Move other inputs to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # Forward pass (now safe)
        with torch.no_grad():
            output = self.model(**encoded)

        embeddings = output.last_hidden_state.squeeze(0)
        tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])

        return tokens, embeddings, offsets

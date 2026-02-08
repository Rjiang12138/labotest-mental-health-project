import torch
import torch.nn as nn
from transformers import BertModel


class TransformerBERT(nn.Module):
    def __init__(self, model_name, num_labels):
        super(TransformerBERT, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.transformer = nn.Transformer(d_model=self.bert.config.hidden_size, nhead=8, num_encoder_layers=6)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, ids, attention_mask, token_type_ids):
        # Extract BERT embeddings
        bert_output = self.bert(ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = bert_output.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # Transformer expects input of shape (seq_len, batch_size, hidden_size)
        transformer_input = sequence_output.permute(1, 0, 2)
        transformer_output = self.transformer(transformer_input, transformer_input)

        # Use the output corresponding to the [CLS] token, which is the first token
        cls_output = transformer_output[0]

        logits = self.classifier(cls_output)
        return logits
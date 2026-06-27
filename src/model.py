from transformers import AutoModel
import torch

MODEL_NAME = "bert-base-uncased"


class MedicalBERT(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.bert = AutoModel.from_pretrained(MODEL_NAME)

        self.dropout = torch.nn.Dropout(0.3)

        self.linear = torch.nn.Linear(
            self.bert.config.hidden_size,
            2
        )

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = outputs.last_hidden_state[:, 0]

        cls = self.dropout(cls)

        logits = self.linear(cls)

        return logits


def main():

    model = MedicalBERT()

    print(model)


if __name__ == "__main__":

    main()
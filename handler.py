# handler.py
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

class TextSummarizationHandler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = "summarization_model"  # Directory where model is saved
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_dir)
        self.model.to(self.device)

    def preprocess(self, data):
        text = data[0]["body"]["text"]
        inputs = self.tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        return inputs.input_ids.to(self.device)

    def inference(self, inputs):
        summary_ids = self.model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        return summary_ids

    def postprocess(self, inference_output):
        summary = self.tokenizer.decode(inference_output[0], skip_special_tokens=True)
        return {"summary_text": summary}

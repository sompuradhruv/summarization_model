import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

class TextSummarizationProcessor:
    def __init__(self, model_dir):
        # Load the tokenizer and model from the specified directory
        self.tokenizer = T5Tokenizer.from_pretrained(model_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def preprocess(self, input_data):
        # Extract the text to summarize from the input data
        text = input_data.get('text', '')
        if not text:
            raise ValueError("Input data must contain 'text' field.")
        # Prepare the inputs for the model
        inputs = self.tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        return inputs.input_ids.to(self.device)

    def inference(self, inputs):
        # Generate the summary
        summary_ids = self.model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        return summary_ids

    def postprocess(self, summary_ids):
        # Decode the summary IDs to get the actual summary text
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return {"summary_text": summary}

    def handle_request(self, request_body):
        # This function handles a complete request cycle
        input_data = json.loads(request_body)
        inputs = self.preprocess(input_data)
        summary_ids = self.inference(inputs)
        output = self.postprocess(summary_ids)
        return json.dumps(output)


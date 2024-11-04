# model.py
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
import torch

# Load T5 tokenizer and model
model_name = "t5-small"  # Use "t5-base" or "t5-large" for larger models
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Function for summarization
def summarize(text, max_length=150):
    # Preprocess text for T5 model
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=max_length, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Save the model and tokenizer to the specified directory
model_dir = "./summarization_model"
os.makedirs(model_dir, exist_ok=True)
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

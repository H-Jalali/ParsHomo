import torch
from transformers import (
    ByT5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    MT5Tokenizer,
    MT5ForConditionalGeneration,
    T5Tokenizer
)
from datasets import Dataset
import pandas as pd

# Loading ByT5 Tokenizer and Model
model_path = "Homograph\Models\Homograph_ByT5_Hasan"
tokenizer = ByT5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to("cuda")
model.eval()

# Converting Raw Data into Input_text
def convert_to_seq2seq(example):
    example["input_text"] = f"{example['Sentence']} | {example['Homograph']}"

    return example, example["input_text"]

# Predicting the Output
def predict_g2p(text):
    inputs = tokenizer(
        text, return_tensors="pt", padding="max_length", truncation=True, max_length=128
    ).to("cuda")
    # print("inputs:", "\n", inputs)

    with torch.no_grad():
        output_tokens = model.generate(**inputs, max_length=16)
    # print("output_tokens:", "\n", output_tokens)
    phonemes = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return phonemes

# Test
text = {
    "Sentence": "به درخواستم اره گفت",
    "Homograph": "اره"
}

_ , model_input = convert_to_seq2seq(text)
phonemes = predict_g2p(model_input)
print(f"Persian Text: {model_input}")
print(f"Phonemes: {phonemes}")

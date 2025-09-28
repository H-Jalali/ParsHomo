import torch
from transformers import ByT5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
import numpy as np
import evaluate

# Load tokenizer and model
model_path = "D:/Homograph/Models/Homograph_Paper_ByT5_Separator_14040430"
tokenizer = ByT5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
model.eval()

# Loading Test Dataset
dataset = load_dataset(
    "json",
    data_files="D:/Homograph/Data/Data/Benchmark_Dataset.jsonl",
    split="train",
)

# Converting Raw Data into Input_text
def convert_to_seq2seq(example):
    # example["input_text"] = f"{example['Sentence']} | {example['Homograph']}"
    example["input_text"] = f"{example['Sentence']} | {example['Homograph']}"     # Separator
    # example["input_text"] = f"جمله: {example['Sentence']} هموگراف: {example['Homograph']}"     # Label
    # example["input_text"] = f"در جمله '{example['Sentence']}' تلفظ واژه '{example['Homograph']}' چیست؟"     # Question
    # example["input_text"] = f":جمله {example['Sentence']} هدف: تعیین تلفظ صحیح واژه '{example['Homograph']}'"     # Instruction
    return example
dataset = dataset.map(convert_to_seq2seq)

# Predicting the Outputs
def predict_g2p(input_text):
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128,
    )

    with torch.no_grad():
        output_tokens = model.generate(**inputs, max_length=32)
    phonemes = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return phonemes

# Sending Data into Prediction Function
Predictions = []
References = []
counter = 0
for sample in dataset:
    pred_phoneme = predict_g2p(sample["input_text"])
    Predictions.append(pred_phoneme)
    true_phoneme = sample["Homograph_Phoneme_eSpeak"]
    References.append(true_phoneme)
    counter = counter + 1
    # print(counter)
print(Predictions, '\n')
print(References)

# Calculating CER (Character Error Rate)
CER = evaluate.load("cer")
cer = CER.compute(predictions=Predictions, references=References)

# # Printing Predictations
# for pred, ref in zip(Predictions[:10], References[:10]):
#     print(f"PRED: {pred} | REF: {ref}")

# Calculating Accuracy
correct = sum(p == l for p, l in zip(Predictions, References))
accuracy = correct / len(Predictions)

print("CER (%): Instruction byt5", cer * 100)
print("Accuracy (%): Instruction byt5 ", accuracy * 100)
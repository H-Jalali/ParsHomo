from datasets import load_dataset, Dataset
from transformers import (
    ByT5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    MT5Tokenizer, 
    MT5ForConditionalGeneration,
)
import torch
import editdistance
import numpy as np
import jiwer
import evaluate

# Calculating Metrics for Validation
CER = evaluate.load("cer")
def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # Converting Logits to Token IDs if Needed
    if isinstance(preds, tuple):
        preds = preds[0]
    if preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decoding
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Cleaning
    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    # Calculating CER (Character Error Rate)
    cer = CER.compute(predictions=decoded_preds, references=decoded_labels)
    
    # Printing Predictations
    for pred, ref in zip(decoded_preds[:10], decoded_labels[:10]):
        print(f"PRED: {pred} | REF: {ref}")

    # Calculating Accuracy
    correct = sum(p == l for p, l in zip(decoded_preds, decoded_labels))
    accuracy = correct / len(decoded_preds)

    return {
        "cer": cer,
        "accuracy": accuracy
    }



train_dataset = load_dataset(
    "json",
    data_files="Homograph\\Data\\Data\\HomoRich_Human_Combined_Checked_Train_eSpeak_FIXED.jsonl",
    split="train"
)
val_dataset = load_dataset(
    "json",
    data_files="Homograph\\Data\\Data\\HomoRich_Human_Combined_Checked_Validation_eSpeak_FIXED.jsonl",
    split="train"
)

print(f"Train size: {len(train_dataset)}")
print(f"Validation size: {len(val_dataset)}")

# Loading mT5 Model and Tokenizer
model_name = "google/mt5-small"     # or   "google/mt5-base"   or   "google/mt5-large"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)


# Converting Raw Data into Input_text and Target_text
def convert_to_seq2seq(example):
    # example["input_text"] = f"{example['Sentence']} | {example['Homograph']}"     # Separator
    # example["input_text"] = f"جمله: {example['Sentence']} هموگراف: {example['Homograph']}"     # Label
    # example["input_text"] = f"در جمله '{example['Sentence']}' تلفظ واژه '{example['Homograph']}' چیست؟"     # Question
    example["input_text"] = f":جمله {example['Sentence']} هدف: تعیین تلفظ صحیح واژه '{example['Homograph']}'"     # Instruction
    # example["input_text"] = example["Sentence"]     # Tag

    example["target_text"] = example["Homograph_Phoneme"]
    return example

train_dataset = train_dataset.map(convert_to_seq2seq)
val_dataset = val_dataset.map(convert_to_seq2seq)


# Tokenizing Input and Target
def preprocess(example):
    inputs = tokenizer(
        example["input_text"], padding="max_length", truncation=True, max_length=128
    )
    targets = tokenizer(
        example["target_text"], padding="max_length", truncation=True, max_length=16
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

train_tokenized_dataset = train_dataset.map(
    preprocess,
    remove_columns=[
        "Sentence",
        "Homograph",
        "Homograph_Phoneme",
        "input_text",
        "target_text",
    ],
)

val_tokenized_dataset = val_dataset.map(
    preprocess,
    remove_columns=[
        "Sentence",
        "Homograph",
        "Homograph_Phoneme",
        "input_text",
        "target_text",
    ],
)
print("tokenized_dataset", val_tokenized_dataset[0])

# Setting Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="Homograph\Models\Homograph_Paper_mT5_Instruction_14040430_log",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=64,
    num_train_epochs=100,
    save_total_limit=3,
    fp16=False,
    logging_dir="./logs",
    logging_steps=10,
    predict_with_generate=True
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_dataset,
    eval_dataset=val_tokenized_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Fine-tunimg the Modela
trainer.train()

# Saving the Model
model.save_pretrained("Homograph\Models\Homograph_Paper_mT5_Instruction_14040430")
tokenizer.save_pretrained("Homograph\Models\Homograph_Paper_mT5_Instruction_14040430")

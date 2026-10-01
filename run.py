import os
import zipfile
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Configuration
CSV_PATH = "dataset.csv"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_DIR = "./my_multilabel_model"
MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 3e-5

# 1. Load Data
print(f"Reading {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["tags", "text"]).reset_index(drop=True)

# Parse comma-separated tags into lists
df["tag_list"] = df["tags"].apply(
    lambda x: [t.strip() for t in str(x).split(",") if t.strip()]
)

# 2. Multi-Label Binarization
mlb = MultiLabelBinarizer()
binary_matrix = mlb.fit_transform(df["tag_list"])

classes = list(mlb.classes_)
num_classes = len(classes)
id2label = {i: c for i, c in enumerate(classes)}
label2id = {c: i for i, c in enumerate(classes)}

print(f"Found {num_classes} classes: {classes}")

# BCEWithLogitsLoss requires target tensors to be floats
df["labels"] = [b.astype(float).tolist() for b in binary_matrix]

# 3. Train/Validation Split (85% / 15%)
train_df, val_df = train_test_split(df[["text", "labels"]], test_size=0.15, random_state=42)

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
    "validation": Dataset.from_pandas(val_df.reset_index(drop=True))
})

# 4. Tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

print("Tokenizing datasets...")
tokenized_dataset = dataset.map(tokenize_batch, batched=True)

# 5. Metrics & Model Initialization
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision": precision_score(labels, preds, average="micro", zero_division=0),
        "recall": recall_score(labels, preds, average="micro", zero_division=0),
    }

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_classes,
    problem_type="multi_label_classification",
    id2label=id2label,
    label2id=label2id
)

# 6. Training Setup
device_is_cuda = torch.cuda.is_available()
training_args = TrainingArguments(
    output_dir="./checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_micro",
    logging_steps=10,
    fp16=device_is_cuda,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
)

print(f"Beginning training (CUDA available: {device_is_cuda})...")
trainer.train()

# 7. Save Weights & Artifacts
print(f"Exporting model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)

# 8. Zip Artifacts for easy download
zip_path = "my_multilabel_model.zip"
print(f"Creating archive {zip_path}...")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(OUTPUT_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            zipf.write(full_path, os.path.relpath(full_path, OUTPUT_DIR))

print("Execution finished successfully.")


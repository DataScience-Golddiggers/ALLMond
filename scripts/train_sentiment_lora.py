import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, TaskType
import torch
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Config
DATA_PATH = "data/raw/Sentiment/Mobile Reviews Sentiment.csv"
MODEL_OUTPUT_DIR = "models/sentiment"
BASE_MODEL = "distilbert-base-uncased"

def train():
    print(f"Loading dataset from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    
    # Map sentiment
    label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
    df['label'] = df['sentiment'].map(label_map)
    
    # Filter out any rows where mapping failed (if any)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    print(f"Dataset loaded: {len(df)} samples")
    
    # Create HF Dataset
    dataset = Dataset.from_pandas(df[['review_text', 'label']])
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    print("Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    def preprocess_function(examples):
        return tokenizer(examples["review_text"], truncation=True, padding=True, max_length=512)
    
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    
    print("Setting up model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, 
        num_labels=3,
        id2label={0: "Negative", 1: "Neutral", 2: "Positive"},
        label2id={"Negative": 0, "Neutral": 1, "Positive": 2}
    )
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        inference_mode=False, 
        r=8, 
        lora_alpha=16, 
        lora_dropout=0.1
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir=f"{MODEL_OUTPUT_DIR}/checkpoints",
        learning_rate=2e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=100,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    
    print("Training...")
    trainer.train()
    
    print(f"Saving model to {MODEL_OUTPUT_DIR}...")
    model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    train()

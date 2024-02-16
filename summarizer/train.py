import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import default_data_collator
from torch.optim import AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Load config
with open("summarizer/config.json") as f:
    config = json.load(f)

model_name = config["model_name"]
batch_size = config["batch_size"]
num_epochs = config["num_epochs"]
lr = config["learning_rate"]
max_input_len = config["max_input_length"]
max_target_len = config["max_target_length"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# Load model and tokenizer
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# Load dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

def preprocess(example):
    inputs = ["summarize: " + doc for doc in example["article"]]
    targets = example["highlights"]

    model_inputs = tokenizer(
        inputs,
        max_length=max_input_len,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        targets,
        max_length=max_target_len,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


print("Preprocessing dataset...")
tokenized_datasets = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

train_data = tokenized_datasets["train"].select(range(500))
train_dataloader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=default_data_collator
)


# Optimizer
optimizer = AdamW(model.parameters(), lr=lr)

print("Starting training...")
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    loop = tqdm(train_dataloader, leave=True)
    for batch in loop:
        batch = {k: torch.tensor(v).to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loop.set_description(f"Epoch {epoch+1}/{num_epochs}")
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

# Save model
os.makedirs("outputs", exist_ok=True)
model.save_pretrained("outputs/t5_finetuned_xsum")
tokenizer.save_pretrained("outputs/t5_finetuned_xsum")

print("Model saved to outputs/t5_finetuned_xsum")
"# Adjusted training config" 

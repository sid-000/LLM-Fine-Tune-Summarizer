import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
import evaluate
import matplotlib.pyplot as plt


# Load model & tokenizer from local fine-tuned output
model = T5ForConditionalGeneration.from_pretrained("outputs/t5_finetuned_xsum")
tokenizer = T5Tokenizer.from_pretrained("outputs/t5_finetuned_xsum")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load test dataset
dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:10]")  # just 10 for demo

# Preprocess inputs
inputs = ["summarize: " + article for article in dataset["article"]]
inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)

input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# Generate predictions
print("Generating summaries...")
outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=64)
predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

# Show results
print("\n--- Sample Summaries ---\n")
for i in range(len(predictions)):
    print(f"\n Article #{i+1}")
    print(f"Ground Truth: {dataset[i]['highlights']}")
    print(f"Prediction  : {predictions[i]}")


rouge = evaluate.load("rouge")
results = rouge.compute(predictions=predictions, references=[x['highlights'] for x in dataset])


print("\n--- Aggregated ROUGE Scores ---\n")
for key, value in results.items():
    print(f"{key}: {value:.4f}")


# Plot ROUGE scores as bar chart
labels = list(results.keys())
scores = [results[k] for k in labels]

plt.figure(figsize=(8, 5))
plt.bar(labels, scores)
plt.ylim(0, 1)
plt.title("ROUGE Score Summary")
plt.xlabel("Metric")
plt.ylabel("F1 Score")
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# Save the figure
plt.savefig("outputs/rouge_scores.png")
print("\nROUGE score plot saved to outputs/rouge_scores.png")
# Added ROUGE and BLEU evaluation 

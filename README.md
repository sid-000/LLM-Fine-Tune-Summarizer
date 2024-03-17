# 🧠 LLM Fine-Tuning for Text Summarization

This project fine-tunes a pretrained T5 model on the XSum dataset for abstractive text summarization using the Hugging Face Transformers library. It demonstrates training, evaluation, and generating summaries with custom inputs.

## 🛠️ Tools & Libraries
- 🤗 Hugging Face Transformers
- PyTorch
- Datasets (XSum)
- ROUGE Evaluation
- Jupyter Notebooks

## 📁 Project Structure

summarizer/
├── train.py # Fine-tuning script using Trainer API
├── evaluate.py # Script to generate predictions & ROUGE scores
├── config.json # Training configuration parameters

outputs/
├── t5_finetuned_xsum
├── rouge_scores.png # ROUGE evaluation chart or metrics image
├── sample_preds.txt # Example model predictions on test data




## 🚀 Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt

2. Run training:
python summarizer/train.py

3. Evaluate model:
python summarizer/evaluate.py


📈 Sample ROUGE Scores

ROUGE-1: 43.2

ROUGE-2: 20.1

ROUGE-L: 40.7


📝 Sample Input & Output

Input: "The government is planning to announce new tax reforms..."
Generated Summary: "The government is preparing new tax reforms."


🗓️ Project Timeline

Originally Started: Nov 2023

Finalized: Jan 2024

Portfolio Update: July 2025


ECHO is on.
# Portfolio Update: March 2024 

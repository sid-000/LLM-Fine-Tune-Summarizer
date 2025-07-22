# 🧠 LLM Fine-Tuning for Text Summarization

This project fine-tunes a pretrained T5 model on the XSum dataset for abstractive text summarization using the Hugging Face Transformers library. It demonstrates training, evaluation, and generating summaries with custom inputs.

## 🛠️ Tools & Libraries
- 🤗 Hugging Face Transformers
- PyTorch
- Datasets (XSum)
- ROUGE Evaluation
- Jupyter Notebooks

## 📁 Project Structure

<pre>
📁 Project Structure

summarizer/
├── train.py            # Fine-tuning script using Hugging Face Trainer API
├── evaluate.py         # Script to evaluate model & generate ROUGE scores
├── config.json         # Training configuration

outputs/
└── t5_finetuned_xsum/
    ├── rouge_scores.png     # ROUGE evaluation metrics image
    ├── sample_preds.txt     # Sample model predictions on test data
</pre>


## 🚀 Getting Started

Follow these steps to set up and run the summarization pipeline:

---

### 📦 **Step 1: Install Dependencies**

```bash
pip install -r requirements.txt
```

---

### 🏋️‍♂️ **Step 2: Run Training**

```bash
python summarizer/train.py
```

---

### 📊 **Step 3: Evaluate the Model**

```bash
python summarizer/evaluate.py
```

---

### 🧪 **Sample ROUGE Scores**

- **ROUGE-1**: 43.2  
- **ROUGE-2**: 20.1  
- **ROUGE-L**: 40.7  

---

### 🔍 **Sample Input & Output**

**Input:**  
`"The government is planning to announce new tax reforms..."`  

**Generated Summary:**  
`"The government is preparing new tax reforms."`

---

### 🗓️ **Project Timeline**

- **Originally Started:** Jan 2024  
- **Finalized:** Mar 2024  
- **Portfolio Update:** July 2025
🗓️ **Project Timeline**

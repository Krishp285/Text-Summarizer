# AI-Powered Text Summarizer by [Your Name]
# Fine-tunes BART and provides a user interface

from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
import os

# Ensure TensorFlow backend
os.environ["TRANSFORMERS_BACKEND"] = "tensorflow"

# Load dataset (1,000 samples)
dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:1000]")

# Load tokenizer and model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Preprocess function
def preprocess(examples):
    inputs = tokenizer(examples['article'], max_length=512, truncation=True, padding='max_length')
    targets = tokenizer(examples['highlights'], max_length=128, truncation=True, padding='max_length')
    inputs["labels"] = targets["input_ids"]
    return inputs

# Tokenize dataset
tokenized_dataset = dataset.map(preprocess, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="C:/summarizer_project/model",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train (comment out after first run)
# trainer.train()

# Load fine-tuned model
model = BartForConditionalGeneration.from_pretrained("C:/summarizer_project/fine_tuned_model")
tokenizer = BartTokenizer.from_pretrained("C:/summarizer_project/fine_tuned_model")

# Summarize function
def summarize_text(text):
    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt")
    summary_ids = model.generate(inputs["input_ids"], max_length=50, min_length=25, num_beams=4)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Interface
print("AI Text Summarizer - Enter text (or 'quit' to exit):")
while True:
    text = input("> ")
    if text.lower() == 'quit':
        break
    summary = summarize_text(text)
    print("Summary:", summary)
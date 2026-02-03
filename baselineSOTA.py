import pandas as pd
from datasets import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch


df = pd.read_csv("processed/sentences_processed.csv")
df = df[df["use_for_training_inclusive"] == True].copy()
df = df[["text", "y_bias_bin_inclusive"]].dropna()


df = df.rename(columns={"text": "text", "y_bias_bin_inclusive": "label"})
df["label"] = df["label"].astype(int)  # make sure labels are int

train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)


tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)


def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True).remove_columns(["text"])
test_dataset = Dataset.from_pandas(test_df).map(tokenize, batched=True).remove_columns(["text"])

# metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

# config
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    eval_strategy="epoch",
    save_strategy="no",
    logging_dir="./logs",
    logging_steps=50,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)


trainer.train()

#  evaluation
results = trainer.evaluate()
print("\nFinal Evaluation Results:")
for k, v in results.items():
    print(f"{k}: {v:.4f}")
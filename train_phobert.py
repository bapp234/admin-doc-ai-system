"""
train_phobert.py

Colab-ready training script for PhoBERT (`vinai/phobert-base-v2`).

Top-of-file Colab pip installs (run these in a Colab cell before executing the script):

!pip install -q transformers datasets pyvi scikit-learn torch

Notes:
- This script expects a CSV named `collected_dataset.csv` with two columns: `text`, `label`.
- The script performs Vietnamese word segmentation with `pyvi.ViTokenizer.tokenize` BEFORE tokenization.
- Output (model + tokenizer + label maps) will be saved to `./phobert_admin_doc_model` by default.

Usage (example in Colab):

```bash
# install dependencies in Colab
!pip install -q transformers datasets pyvi scikit-learn torch

# run training (in a Colab code cell)
!python train_phobert.py --csv collected_dataset.csv --output_dir ./phobert_admin_doc_model --batch_size 8
```

"""

import os
import json
import argparse
import logging
import random

import numpy as np
import pandas as pd
import torch

from pyvi import ViTokenizer
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


# -----------------------------
# Utility helpers
# -----------------------------

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataframe(csv_path: str) -> pd.DataFrame:
    """Load `collected_dataset.csv` and drop missing rows.

    Expects columns: `text`, `label`.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # keep only required columns, coerce to string
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Input CSV must contain 'text' and 'label' columns")

    df = df[['text', 'label']].dropna()
    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(str)
    df = df.reset_index(drop=True)
    return df


def encode_labels(df: pd.DataFrame):
    """Map categorical label strings to integer ids and return mappings.

    Returns:
        df (modified): `label` column becomes integer ids
        label2id (dict): {label_str: id}
        id2label (dict): {id: label_str}
    """
    unique_labels = sorted(df['label'].unique().tolist())
    label2id = {lab: i for i, lab in enumerate(unique_labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    # replace textual labels with integer ids (column `label`)
    df['label'] = df['label'].map(label2id)
    return df, label2id, id2label


def save_label_maps(label2id: dict, id2label: dict, out_dir: str):
    """Save JSON mappings to the model output directory."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'label2id.json'), 'w', encoding='utf-8') as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, 'id2label.json'), 'w', encoding='utf-8') as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)


def compute_metrics(eval_pred):
    """Compute accuracy and macro F1 for the Trainer.

    `eval_pred` is a (logits, labels) tuple provided by the Trainer.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {
        'accuracy': acc,
        'f1_macro': f1,
    }


# -----------------------------
# Main training pipeline
# -----------------------------

def main(csv_path: str, output_dir: str, batch_size: int, max_length: int = 256):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    set_seed(42)

    logger.info("Loading dataset...")
    df = load_dataframe(csv_path)
    if df.shape[0] == 0:
        raise RuntimeError("No training data found after dropping missing rows.")

    logger.info(f"Loaded {len(df)} rows from {csv_path}")

    # Encode labels and save mappings
    df, label2id, id2label = encode_labels(df)
    logger.info(f"Found {len(label2id)} unique labels")
    save_label_maps(label2id, id2label, out_dir=output_dir)

    # Build a HuggingFace `Dataset` and split 80/20
    logger.info("Creating HuggingFace Dataset and train/validation split (80/20)...")
    ds = Dataset.from_pandas(df[['text', 'label']], preserve_index=False)
    split = ds.train_test_split(test_size=0.2, seed=42)
    train_ds = split['train']
    val_ds = split['test']

    # Initialize tokenizer (PhoBERT)
    logger.info("Loading tokenizer: vinai/phobert-base-v2")
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base-v2')

    # Preprocessing: Vietnamese segmentation with pyvi BEFORE tokenization
    def preprocess_batch(examples):
        # `examples['text']` is a list of strings
        texts = [str(t) for t in examples['text']]
        segmented = [ViTokenizer.tokenize(t) for t in texts]
        # Tokenize segmented text (padding + truncation)
        return tokenizer(segmented, padding='max_length', truncation=True, max_length=max_length)

    logger.info("Segmenting (pyvi) and tokenizing datasets... (this may take a few moments)")
    # Use num_proc=1 to avoid pickling tokenizer issues in multi-process map
    train_ds = train_ds.map(preprocess_batch, batched=True, remove_columns=['text'], num_proc=1)
    val_ds = val_ds.map(preprocess_batch, batched=True, remove_columns=['text'], num_proc=1)

    # Rename label -> labels for Trainer and set PyTorch format
    train_ds = train_ds.rename_column('label', 'labels')
    val_ds = val_ds.rename_column('label', 'labels')
    train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    num_labels = len(label2id)
    logger.info(f"Initializing model with num_labels={num_labels}")

    model = AutoModelForSequenceClassification.from_pretrained(
        'vinai/phobert-base-v2',
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_strategy='steps',
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='f1_macro',
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving model and tokenizer to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_label_maps(label2id, id2label, out_dir=output_dir)

    logger.info("Training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PhoBERT classifier (vinai/phobert-base-v2)')
    parser.add_argument('--csv', type=str, default='collected_dataset.csv', help='Path to collected_dataset.csv')
    parser.add_argument('--output_dir', type=str, default='./phobert_admin_doc_model', help='Directory to save model + tokenizer')
    parser.add_argument('--batch_size', type=int, default=8, help='Per-device batch size (use 8 for Colab T4; 16 if you have more memory)')
    parser.add_argument('--max_length', type=int, default=256, help='Max token length')
    args = parser.parse_args()

    main(args.csv, args.output_dir, args.batch_size, max_length=args.max_length)

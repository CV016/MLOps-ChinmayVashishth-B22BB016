"""
BERT fine-tuning: load DistilBERT, configure Trainer, train and save.
Task 5: Prepares dataset, configures training arguments, trains with Trainer API,
and logs training metrics to a JSON file.
"""

import json
import os
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from utils.config import MODEL_NAME, CACHED_MODEL_DIR, get_device


# Disable Weights & Biases logging when no API key is set.
os.environ.setdefault("WANDB_DISABLED", "true")


def compute_metrics(pred):
    """
    HuggingFace compute_metrics callback: accuracy and F1 (macro/weighted).
    Used for both training eval steps and final evaluation (Task 6).
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def build_trainer(
    model,
    training_args,
    train_dataset,
    eval_dataset,
    compute_metrics_fn=None,
):
    """
    Build HuggingFace Trainer with model, args, datasets and optional compute_metrics.
    """
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_fn or compute_metrics,
    )


def train_bert(
    train_dataset,
    test_dataset,
    id2label,
    model_name=MODEL_NAME,
    output_dir="./results",
    logging_dir="./logs",
    num_train_epochs=3,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=None,
    save_dir=None,
):
    """
    Load DistilBERT for sequence classification, train, and save.

    Args:
        train_dataset, test_dataset: ReviewDataset instances.
        id2label: Dict[int, str] for model num_labels and label names.
        model_name: Pretrained model name.
        output_dir, logging_dir: Trainer output and log dirs.
        num_train_epochs, per_device_*: Training hyperparameters.
        learning_rate, warmup_steps, weight_decay: Optimizer/scheduler.
        logging_steps, eval_strategy, eval_steps: Logging and evaluation.
        save_dir: Directory to save the fine-tuned model; defaults to CACHED_MODEL_DIR.

    Returns:
        Tuple (trainer, model).
    """
    device = get_device()
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id2label),
    ).to(device)

    if eval_steps is None:
        eval_steps = logging_steps

    training_args = TrainingArguments(
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        output_dir=output_dir,
        logging_dir=logging_dir,
        logging_steps=logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        report_to=[],
    )

    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics_fn=compute_metrics,
    )

    trainer.train()

    save_dir = save_dir or CACHED_MODEL_DIR
    trainer.save_model(save_dir)

    # Task 5: Log training metrics to JSON (log_history from Trainer state).
    _save_training_metrics(trainer, output_dir)

    return trainer, model


def _save_training_metrics(trainer, output_dir):
    """Write Trainer log_history to training_metrics.json for Task 5."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "training_metrics.json"
    # log_history contains dicts with loss, eval_*, etc.; may contain non-serializable types.
    log_history = []
    for entry in trainer.state.log_history:
        clean = {}
        for k, v in entry.items():
            try:
                clean[k] = float(v)
            except (TypeError, ValueError):
                clean[k] = v
        log_history.append(clean)
    with open(path, "w") as f:
        json.dump(log_history, f, indent=2)

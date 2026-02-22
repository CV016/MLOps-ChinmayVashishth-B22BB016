"""
Evaluate trained Trainer: run evaluation and extract predicted labels.
"""


def evaluate_trainer(trainer, eval_dataset=None):
    """
    Run HuggingFace Trainer evaluation.

    Args:
        trainer: HuggingFace Trainer (after training).
        eval_dataset: Optional dataset; uses trainer.eval_dataset if None.

    Returns:
        Dict of eval metrics (e.g. eval_loss, eval_accuracy).
    """
    dataset = eval_dataset if eval_dataset is not None else trainer.eval_dataset
    return trainer.evaluate(eval_dataset=dataset)


def get_predictions(trainer, test_dataset, id2label):
    """
    Get predicted class labels (strings) from the trainer on test_dataset.

    Returns:
        List of predicted genre strings.
    """
    pred = trainer.predict(test_dataset)
    pred_ids = pred.predictions.argmax(-1).flatten().tolist()
    return [id2label[i] for i in pred_ids]

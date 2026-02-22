"""
Task 4: Load tokenizer and model from Hugging Face.

Model selection: distilbert-base-cased (same as in the original notebook).

Why this model was selected:
- DistilBERT is a distilled version of BERT (Sanh et al., 2019), retaining ~97% of
  BERT's language understanding while being 40% smaller and 60% faster. This makes
  it suitable for resource-constrained environments and faster iteration during
  fine-tuning.
- The "cased" variant preserves case (e.g. "BERT" vs "bert"), which can help for
  book reviews where capitalization and punctuation carry meaning.
- It is well supported in Hugging Face Transformers, has a matching
  DistilBertTokenizerFast for efficient tokenization, and is a common choice for
  sequence classification tasks such as genre prediction from review text.
- For the Goodreads multi-class genre classification task (8 classes), the
  pre-trained representations provide a strong baseline before adding a
  classification head; fine-tuning then adapts the model to the target domain.
"""

from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from utils.config import MODEL_NAME, get_device


def load_tokenizer(model_name=MODEL_NAME):
    """Load the Hugging Face tokenizer for the selected model."""
    return DistilBertTokenizerFast.from_pretrained(model_name)


def load_model_for_classification(model_name=MODEL_NAME, num_labels=8, id2label=None):
    """
    Load the pre-trained classification model and move it to the active device.

    Args:
        model_name: Hugging Face model id (default: distilbert-base-cased).
        num_labels: Number of output classes (required for classification head).
        id2label: Optional dict mapping label id to label name for config.

    Returns:
        Model on the appropriate device (cuda/cpu).
    """
    kwargs = {"num_labels": num_labels}
    if id2label is not None:
        kwargs["id2label"] = id2label
    model = DistilBertForSequenceClassification.from_pretrained(model_name, **kwargs)
    device = get_device()
    return model.to(device)

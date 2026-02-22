# Goodreads Genre Classification Report

## Model Selection

* **Model Used:** `distilbert-base-cased`
* **Rationale:** Selected for efficiency (smaller and faster than standard BERT), cased tokenization which is highly beneficial for analyzing text like book reviews, and strong native support for sequence classification.

## Training Summary

* **Total Epochs:** 3
* **Final Training Loss:** 0.9859
* **Hyperparameters:**
  * Learning Rate: 5e-5
  * Train Batch Size: 10
  * Eval Batch Size: 16
  * Warmup Steps: 100
  * Weight Decay: 0.01
* **Runtime Metrics:** Training took approximately 1520.77 seconds (~25 minutes) over 1920 global steps.

## Evaluation Comparison

The model was evaluated against a test set of 1,600 sampled reviews (200 per genre).

* **Baseline Model (TF-IDF + Logistic Regression):**

  * Accuracy: 0.55
  * F1 Macro: 0.55
  * F1 Weighted: 0.55
* **Fine-Tuned DistilBERT Model (Overall):**

  * Accuracy: 0.5937
  * F1 Macro: 0.5964
  * F1 Weighted: 0.5964
  * Evaluation Loss: 1.2751
* **Per-Class Metrics (Fine-Tuned Model):**

  * **Children:** Precision: 0.6404, Recall: 0.6500, F1: 0.6452
  * **Comics Graphic:** Precision: 0.8541, Recall: 0.7900, F1: 0.8208
  * **Fantasy Paranormal:** Precision: 0.3818, Recall: 0.4200, F1: 0.4000
  * **History Biography:** Precision: 0.5930, Recall: 0.5900, F1: 0.5915
  * **Mystery Thriller Crime:** Precision: 0.5263, Recall: 0.5500, F1: 0.5379
  * **Poetry:** Precision: 0.7525, Recall: 0.7450, F1: 0.7487
  * **Romance:** Precision: 0.6250, Recall: 0.5750, F1: 0.5990
  * **Young Adult:** Precision: 0.4257, Recall: 0.4300, F1: 0.4279
* **Comparison Summary:** The fine-tuned DistilBERT model outperformed the logistic regression baseline, demonstrating a ~4.3% improvement in overall accuracy and a ~4.6% improvement in F1 macro/weighted scores.

## Challenges

* **Environment Rendering Issues:** During training in Google Colab, the default Hugging Face `Trainer` progress bar caused JavaScript execution errors. This was resolved by appending `disable_tqdm=True` to the `TrainingArguments`.
* **Class Confusion on Overlapping Genres:** The model experienced difficulty distinguishing between conceptually similar genres. While visually distinct genres like `comics_graphic` achieved high scores (F1: 0.8208), classification performance dropped significantly for overlapping categories such as `fantasy_paranormal` (F1: 0.4000) and `young_adult` (F1: 0.4279).

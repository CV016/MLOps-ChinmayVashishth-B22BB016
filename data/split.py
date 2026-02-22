"""
Split genre-based reviews into train and test lists.
"""

import random


def split_train_test(
    genre_reviews_dict,
    reviews_per_genre=1000,
    train_ratio=0.8,
):
    """
    Build train_texts, train_labels, test_texts, test_labels from genre_reviews_dict.

    For each genre, samples reviews_per_genre reviews, then splits by train_ratio
    into train (first fraction) and test (remainder).

    Args:
        genre_reviews_dict: Dict[str, list] of genre -> list of review strings.
        reviews_per_genre: Number of reviews to use per genre (sampled if more available).
        train_ratio: Fraction of each genre used for training (e.g. 0.8 = 80% train).

    Returns:
        Tuple of (train_texts, train_labels, test_texts, test_labels), each list.
    """
    train_texts = []
    train_labels = []
    test_texts = []
    test_labels = []

    for genre, reviews in genre_reviews_dict.items():
        sampled = random.sample(reviews, min(reviews_per_genre, len(reviews)))
        n_train = int(len(sampled) * train_ratio)
        for r in sampled[:n_train]:
            train_texts.append(r)
            train_labels.append(genre)
        for r in sampled[n_train:]:
            test_texts.append(r)
            test_labels.append(genre)

    return train_texts, train_labels, test_texts, test_labels

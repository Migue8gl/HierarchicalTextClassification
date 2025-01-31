from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    figsize: tuple = (16, 10),
) -> plt.Figure:
    """Generate confusion matrix plot with Seaborn"""
    cm = metrics.confusion_matrix(y_true, y_pred)
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])))

    fig = plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig


def evaluate(clf: BaseEstimator, X_test, y_test, label: Optional[str] = None):
    if label:
        pred = clf.predict(X_test, label)
    else:
        pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, pred)
    f1 = metrics.f1_score(y_test, pred, average="weighted", zero_division=0)
    rec = metrics.recall_score(y_test, pred, average="weighted", zero_division=0)
    prec = metrics.precision_score(y_test, pred, average="weighted", zero_division=0)
    print(f"Accuracy: {100 * acc:.1f}%")
    print(f"Precision: {100 * prec:.1f}%")
    print(f"Recall: {100 * rec:.1f}%")
    print(f"F1-score: {100 * f1:.1f}%")
    return pred


def create_pipeline_nb() -> Pipeline:
    """Create base ML pipeline"""
    return Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", MultinomialNB()),
        ]
    )

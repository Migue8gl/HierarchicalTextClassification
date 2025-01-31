import numpy as np
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics

from utils import create_pipeline_nb, evaluate

seed = np.random.randint(0, 1000)

def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, title: str, category: str
) -> None:
    cm = metrics.confusion_matrix(y_true, y_pred)
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])))

    plt.figure(figsize=(16, 10))
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
    plt.savefig(f"img/cm_local_level_classification_{category}.png")


def main():
    pipeline = create_pipeline_nb()

    df = pl.read_csv("data/train.csv")
    test_df = pl.read_csv("data/test.csv")

    for category in df.select(pl.exclude("text")).columns:
        X_test = test_df["text"].to_numpy()
        y_test = test_df[category].to_numpy()
        X_train = df["text"].to_numpy()
        y_train = df[category].to_numpy()
        _ = pipeline.fit(X_train, y_train)
        final_preds = evaluate(pipeline, X_test, y_test)

        plot_confusion_matrix(
            y_test, final_preds, f"{category} confusion matrix", category
        )


if __name__ == "__main__":
    main()

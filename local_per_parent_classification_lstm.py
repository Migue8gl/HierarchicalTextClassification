import warnings

warnings.filterwarnings("ignore")

from typing import Any, List, Optional, Tuple

import polars as pl

from lstm_utils import evaluate, predict, train_lstm
from utils import plot_confusion_matrix


class HierarchicalLSTMModel:
    def __init__(self, hierarchy_levels: List[Any], hierarchy_labels: List[str]):
        """
        Initialize with hierarchical classification levels.
        hierarchy_levels: List where the first element is a tuple (model, tokenizer, vocab, label_decoder)
                          for the root classifier, and subsequent elements are dictionaries mapping
                          parent category (in lowercase) to a tuple (model, tokenizer, vocab, label_decoder).
        """
        self.hierarchy = hierarchy_levels
        self.hierarchy_labels = hierarchy_labels

    def predict(self, X: List[str], level_label: Optional[str] = None) -> List[str]:
        """Make hierarchical predictions"""
        # Get the root model tuple and perform prediction.
        root_model, root_tokenizer, root_vocab, root_label_decoder = self.hierarchy[0]
        current_pred = predict(
            root_model, X, root_tokenizer, root_vocab, root_label_decoder
        )

        if level_label is None:
            level_label = self.hierarchy_labels[-1]

        # Iterate through the hierarchy levels.
        for index, level in enumerate(self.hierarchy[1:]):
            next_pred = []
            # If we reached the specified level, break.
            if self.hierarchy_labels[index].lower() == level_label.lower():
                break
            for i, pred in enumerate(current_pred):
                classifier_tuple = level.get(pred.lower())
                if classifier_tuple:
                    model_i, tokenizer_i, vocab_i, label_decoder_i = classifier_tuple
                    next_pred.append(
                        predict(model_i, [X[i]], tokenizer_i, vocab_i, label_decoder_i)[
                            0
                        ]
                    )
                else:
                    next_pred.append(
                        pred
                    )  # If no specific classifier is found, keep previous prediction.
            current_pred = next_pred

        return current_pred


def train_hierarchy(
    df: pl.DataFrame, hierarchy: List[Tuple[Optional[str], str]]
) -> HierarchicalLSTMModel:
    """
    Train hierarchical LSTM model.
    hierarchy: List of tuples specifying (filter_column, target_column).
               For the root level, filter_column should be None.
    """
    models = []
    current_df = df

    for level, (filter_col, target_col) in enumerate(hierarchy):
        if level == 0:  # Root level
            X_train = current_df["text"].to_list()
            y_train = current_df[target_col].to_list()
            # Get label_decoder from train_lstm (third returned value)
            model, _, label_decoder, vocab, tokenizer = train_lstm(
                X_train,
                y_train,
                hidden_size=128,
                num_layers=2,
                num_epochs=10,
                batch_size=32,
            )
            # Store as a tuple for the root.
            models.append((model, tokenizer, vocab, label_decoder))
        else:
            level_models = {}
            parent_categories = current_df[filter_col].unique().to_list()

            for category in parent_categories:
                filtered_df = current_df.filter(pl.col(filter_col) == category)
                X_train = filtered_df["text"].to_list()
                y_train = filtered_df[target_col].to_list()

                if len(set(y_train)) > 1:  # Only train if multiple classes exist.
                    model, _, label_decoder, vocab, tokenizer = train_lstm(
                        X_train,
                        y_train,
                        hidden_size=128,
                        num_layers=2,
                        num_epochs=10,
                        batch_size=32,
                    )
                    level_models[category.lower()] = (
                        model,
                        tokenizer,
                        vocab,
                        label_decoder,
                    )

            models.append(level_models)
            # Filter current_df to only include rows with the given parent categories.
            current_df = current_df.filter(pl.col(filter_col).is_in(parent_categories))

    # Extract the target columns as labels from the hierarchy definition.
    hierarchy_labels = [cat[1] for cat in hierarchy]
    return HierarchicalLSTMModel(models, hierarchy_labels)


def main():
    # Load data.
    df = pl.read_csv("data/train.csv")
    test_df = pl.read_csv("data/test.csv")

    # Define hierarchy: For the root level, use None as the filter column.
    categories = df.select(pl.exclude("text")).columns
    hierarchy = [(None, categories[0])] + [
        (categories[i], categories[i + 1]) for i in range(len(categories) - 1)
    ]

    # Train hierarchical LSTM model.
    model = train_hierarchy(df, hierarchy)

    # For each category, make predictions, evaluate, and plot the confusion matrix.
    for category in categories:
        X_test = test_df["text"].to_list()
        y_test = test_df[category].to_list()

        predictions = model.predict(X_test, category)

        # Evaluate.
        try:
            save_kwargs = {"name": "lstm_per_parent", "category": category}
            metrics_result = evaluate(model, X_test, y_test, **save_kwargs)
            print(f"Evaluation metrics for {category}:")
            print(metrics_result)
        except Exception as e:
            print(f"Could not run evaluate function: {str(e)}")

        # Plot confusion matrix.
        try:
            plot_confusion_matrix(
                y_test,
                predictions,
                f"{category} confusion matrix",
                category,
                "lstm_per_parent",
            )
            print(f"Confusion matrix plotted for {category}")
        except Exception as e:
            print(f"Could not plot confusion matrix: {str(e)}")


if __name__ == "__main__":
    main()

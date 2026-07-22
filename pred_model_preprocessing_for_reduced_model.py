import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss
)

# Loading data

survey_df = pd.read_csv('ACME-HappinessSurvey2020.csv')
X = survey_df[["X1", "X2", "X3", "X4", "X5", "X6"]]
y = survey_df["Y"]

# Testing the full and reduced models

candidate_subsets = {
    "Full": ["X1", "X2", "X3", "X4", "X5", "X6"],
    "Reduced": ["X1", "X2", "X5"]
}

# Search a reasonable range around 0.50 and the observed optimum
threshold_grid = np.arange(0.40, 0.751, 0.01)

# Test repeatedly

outer_cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=1900
)

outer_results = []

for outer_evaluation, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X, y), start = 1):
    X_outer_train = X.iloc[outer_train_idx]
    X_outer_test = X.iloc[outer_test_idx]

    y_outer_train = y.iloc[outer_train_idx]
    y_outer_test = y.iloc[outer_test_idx]

    inner_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=10_000 + outer_evaluation
    )

    best_choice = None
    best_tie_breaking_key = None

    # Select feature set and threshold using outer-training data only
    for subset_name, columns in candidate_subsets.items():

        # Maximizing accuracy

        # Accuracy values for every candidate threshold
        threshold_scores = {
            threshold: []
            for threshold in threshold_grid
        }

        for inner_train_idx, inner_validation_idx in inner_cv.split(
            X_outer_train,
            y_outer_train
        ):
            X_inner_train = X_outer_train.iloc[inner_train_idx][columns]
            X_inner_validation = (
                X_outer_train.iloc[inner_validation_idx][columns]
            )

            y_inner_train = y_outer_train.iloc[inner_train_idx]
            y_inner_validation = y_outer_train.iloc[inner_validation_idx]

            # Fitting inner model

            inner_model = LogisticRegression(
                max_iter=2000
            )
            inner_model.fit(X_inner_train, y_inner_train)
            validation_probabilities = inner_model.predict_proba(
                X_inner_validation
            )[:, 1]

            for threshold in threshold_grid:
                validation_predictions = (
                    validation_probabilities >= threshold
                ).astype(int)

                threshold_scores[threshold].append(
                    accuracy_score(
                        y_inner_validation,
                        validation_predictions
                    )
                )
        
        # Mean inner accuracy
        for threshold, fold_scores in threshold_scores.items():
            mean_inner_accuracy = np.mean(fold_scores)

            # Primary criterion: highest inner accuracy.
            # Ties: fewer predictors, then threshold nearest 0.50.
            tie_breaking_key = (
                mean_inner_accuracy,
                -len(columns),
                -abs(threshold - 0.50)
            )

            if (
                best_tie_breaking_key is None
                or tie_breaking_key > best_tie_breaking_key
            ):
                best_tie_breaking_key = tie_breaking_key

                best_choice = {
                    "subset_name": subset_name,
                    "columns": columns,
                    "threshold": threshold,
                    "inner_accuracy": mean_inner_accuracy
                }

    # Refit the selected model on the entire outer training set
    selected_columns = best_choice["columns"]
    selected_threshold = best_choice["threshold"]

    final_outer_model = LogisticRegression(
        max_iter=2000
    )

    final_outer_model.fit(
        X_outer_train[selected_columns],
        y_outer_train
    )

    outer_probabilities = final_outer_model.predict_proba(
        X_outer_test[selected_columns]
    )[:, 1]

    outer_predictions = (
        outer_probabilities >= selected_threshold
    ).astype(int)

    outer_results.append({
        "evaluation": outer_evaluation,
        "subset": best_choice["subset_name"],
        "features": tuple(selected_columns),
        "threshold": selected_threshold,
        "inner_accuracy": best_choice["inner_accuracy"],
        "accuracy": accuracy_score(
            y_outer_test,
            outer_predictions
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_outer_test,
            outer_predictions
        ),
        "precision": precision_score(
            y_outer_test,
            outer_predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_outer_test,
            outer_predictions,
            zero_division=0
        ),
        "f1": f1_score(
            y_outer_test,
            outer_predictions,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_outer_test,
            outer_probabilities
        ),
        "log_loss": log_loss(
            y_outer_test,
            outer_probabilities,
            labels=[0, 1]
        )
    })

outer_results = pd.DataFrame(outer_results)

metric_columns = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "log_loss"
]

print("\nOuter-CV performance:")
print(
    outer_results[metric_columns]
    .agg(["mean", "std", "min", "max"])
    .T
)

print("\nSelected feature sets:")
print(outer_results["subset"].value_counts())

print("\nSelected thresholds:")
print(outer_results["threshold"].describe())

print("\nComplete results:")
print(outer_results)

# Mean threshold is 0.55. Thus, this was chosen for the final model. 
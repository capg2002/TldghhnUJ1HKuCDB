import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import FixedThresholdClassifier

from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    confusion_matrix
)


# ============================================================
# Reporting functions
# ============================================================

def print_heading(title):
    """Print a consistent section heading."""

    line = "=" * 70

    print(f"\n{line}")
    print(title)
    print(line)


def print_metrics(y_true, y_pred, y_prob):
    """
    Print binary-classification performance metrics.

    Parameters
    ----------
    y_true:
        Observed outcome values.

    y_pred:
        Predicted class labels.

    y_prob:
        Predicted probabilities for Y = 1.
    """

    metrics = [
        ("Accuracy", accuracy_score(y_true, y_pred)),
        (
            "Balanced accuracy",
            balanced_accuracy_score(y_true, y_pred)
        ),
        (
            "Precision",
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),
        (
            "Recall",
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),
        (
            "F1 score",
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),
        ("ROC AUC", roc_auc_score(y_true, y_prob)),
        (
            "Average precision",
            average_precision_score(y_true, y_prob)
        ),
        ("Log loss", log_loss(y_true, y_prob)),
        (
            "Brier score",
            brier_score_loss(y_true, y_prob)
        )
    ]

    print("\nTest-set performance")
    print("-" * 38)
    print(f"{'Metric':<24}{'Value':>14}")
    print("-" * 38)

    for metric_name, metric_value in metrics:
        print(f"{metric_name:<24}{metric_value:>14.4f}")

    # Force the confusion matrix to use the order 0, 1.
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    )

    print("\nConfusion matrix")
    print("-" * 38)
    print(matrix_df.to_string())


def print_cv_results(cv_scores):
    """
    Summarize the cross-validation results returned by
    sklearn.model_selection.cross_validate.
    """

    # Scikit-learn reports negative log loss because larger scorer
    # values are always treated as better. Multiply it by -1 when
    # displaying the result.
    metrics = [
        (
            "Accuracy",
            cv_scores["test_accuracy"]
        ),
        (
            "ROC AUC",
            cv_scores["test_roc_auc"]
        ),
        (
            "Log loss",
            -cv_scores["test_neg_log_loss"]
        )
    ]

    print("\nCross-validation performance")
    print("-" * 48)
    print(
        f"{'Metric':<18}"
        f"{'Mean':>12}"
        f"{'Std. dev.':>14}"
    )
    print("-" * 48)

    for metric_name, values in metrics:
        print(
            f"{metric_name:<18}"
            f"{np.mean(values):>12.4f}"
            f"{np.std(values, ddof=1):>14.4f}"
        )


def print_logistic_equation(
    threshold_model,
    feature_names,
    threshold
):
    """
    Print the fitted equation from a FixedThresholdClassifier
    containing a LogisticRegression model.
    """

    # FixedThresholdClassifier stores the fitted logistic-regression
    # estimator in estimator_ after fit() has been called.
    logistic_model = threshold_model.estimator_

    intercept = logistic_model.intercept_[0]
    coefficients = logistic_model.coef_[0]

    # Construct a readable fitted linear predictor.
    linear_terms = [f"{intercept:.6f}"]

    for feature, coefficient in zip(
        feature_names,
        coefficients
    ):
        sign = "+" if coefficient >= 0 else "-"

        linear_terms.append(
            f" {sign} {abs(coefficient):.6f}({feature})"
        )

    linear_predictor = "".join(linear_terms)

    positive_class = logistic_model.classes_[1]

    print("\nFitted logistic-regression equation")
    print("-" * 70)

    print(
        f"Let p = P(Y = {positive_class} | X)."
    )

    print(
        "\nlog(p / (1 - p)) = "
        f"{linear_predictor}"
    )

    print(
        "\np = 1 / "
        f"(1 + exp(-({linear_predictor})))"
    )

    print("\nClassification rule")

    print(
        f"Predicted Y = 1 when p >= {threshold:.2f}; "
        "otherwise predicted Y = 0."
    )


def print_naive_bayes_equation(
    fitted_search,
    feature_names,
    original_categories
):
    """
    Print the fitted Categorical Naive Bayes prediction rule,
    class priors, and conditional probability tables.

    A Naive Bayes model does not have one linear equation like
    logistic regression. It is completely defined by:

        1. The class prior probabilities.
        2. The conditional probability of every response category
           for every feature and class.
    """

    best_pipeline = fitted_search.best_estimator_
    nb_model = best_pipeline.named_steps["nb"]

    classes = nb_model.classes_
    class_priors = np.exp(nb_model.class_log_prior_)

    best_alpha = fitted_search.best_params_["nb__alpha"]

    print("\nFitted Categorical Naive Bayes equation")
    print("-" * 70)

    print(
        "For each possible class c, calculate:\n"
    )

    for class_index, class_label in enumerate(classes):
        conditional_terms = " × ".join(
            [
                f"P({feature}=x_{feature} | Y={class_label})"
                for feature in feature_names
            ]
        )

        print(
            f"score_{class_label}(x) = "
            f"{class_priors[class_index]:.6f} × "
            f"{conditional_terms}"
        )

    print(
        "\nP(Y=c | x) = "
        "score_c(x) / sum(score_k(x) for all classes k)"
    )

    print(
        "\nPredicted Y = the class with the largest posterior "
        "probability."
    )

    print(
        "\nThe smoothed conditional probabilities are estimated as:"
    )

    print(
        "P(X_j=k | Y=c) = "
        "(N_jkc + alpha) / (N_c + alpha*K_j)"
    )

    print(f"\nSelected alpha = {best_alpha:g}")

    # Print the fitted class-prior probabilities.
    prior_table = pd.DataFrame(
        {
            "Class": classes,
            "Prior probability": class_priors
        }
    )

    print("\nClass prior probabilities")
    print("-" * 38)

    print(
        prior_table.to_string(
            index=False,
            formatters={
                "Prior probability": lambda value: f"{value:.6f}"
            }
        )
    )

    print(
        "\nConditional probability tables\n"
        "Each entry is P(feature = category | Y = class)."
    )

    # feature_log_prob_ contains one probability table per feature.
    for feature_index, feature in enumerate(feature_names):
        probabilities = np.exp(
            nb_model.feature_log_prob_[feature_index]
        )

        probability_table = pd.DataFrame(
            probabilities,
            index=[
                f"Y = {class_label}"
                for class_label in classes
            ],
            columns=[
                f"{feature} = {category}"
                for category in original_categories[feature_index]
            ]
        )

        print(f"\n{feature}")
        print("-" * 70)

        print(
            probability_table.to_string(
                float_format=lambda value: f"{value:.4f}"
            )
        )


# ============================================================
# Load and prepare the data
# ============================================================

survey_df = pd.read_csv(
    "ACME-HappinessSurvey2020.csv"
)

Y_var = survey_df["Y"]

X_full = survey_df[
    ["X1", "X2", "X3", "X4", "X5", "X6"]
]

# Reduced logistic-regression feature set.
X_reduced = survey_df[
    ["X1", "X2", "X5"]
]


# ============================================================
# Train/test split
# ============================================================

seed = 22

X_train, X_test, Y_train, Y_test = train_test_split(
    X_full,
    Y_var,
    test_size=0.20,
    random_state=seed
)

X_train_reduced = X_train[
    ["X1", "X2", "X5"]
]

X_test_reduced = X_test[
    ["X1", "X2", "X5"]
]


# ============================================================
# Cross-validation definitions
# ============================================================

# Used to evaluate the reduced logistic-regression model.
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=seed
)

# Used to select the Naive Bayes smoothing parameter.
inner_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=seed
)


# ============================================================
# Reduced logistic regression
# Features: X1, X2, and X5
# ============================================================

model_threshold = 0.62

base_reduced_model = LogisticRegression()

# The logistic probabilities are unchanged, but class 1 is predicted
# when its estimated probability is at least 0.62 instead of 0.50.
reduced_model = FixedThresholdClassifier(
    estimator=base_reduced_model,
    threshold=model_threshold,
    response_method="predict_proba"
)

reduced_model.fit(
    X_train_reduced,
    Y_train
)

Y_pred = reduced_model.predict(
    X_test_reduced
)

Y_prbs = reduced_model.predict_proba(
    X_test_reduced
)[:, 1]


# Evaluate the model using five-fold stratified cross-validation.
scores_reduced = cross_validate(
    reduced_model,
    X_reduced,
    Y_var,
    cv=cv,
    scoring=[
        "accuracy",
        "roc_auc",
        "neg_log_loss"
    ]
)


print_heading(
    "REDUCED LOGISTIC REGRESSION: X1, X2, X5"
)

print_logistic_equation(
    threshold_model=reduced_model,
    feature_names=["X1", "X2", "X5"],
    threshold=model_threshold
)

print_metrics(
    y_true=Y_test,
    y_pred=Y_pred,
    y_prob=Y_prbs
)

print_cv_results(
    scores_reduced
)


# ============================================================
# Full Categorical Naive Bayes model
# Features: X1, X2, X3, X4, X5, and X6
# ============================================================

# Naive Bayes assumes that the six survey answers are
# conditionally independent after conditioning on Y.
full_nb_features = list(
    X_train.columns
)

# Each survey response is a Likert-scale category from 1 to 5.
full_likert_categories = [
    [1, 2, 3, 4, 5]
    for feature in full_nb_features
]

full_nb_pipeline = Pipeline([
    (
        "encode",
        OrdinalEncoder(
            categories=full_likert_categories,
            dtype=np.int64
        )
    ),
    (
        "nb",
        CategoricalNB(
            min_categories=5
        )
    )
])

# Candidate smoothing values.
alpha_grid = [
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    1.00,
    2.00,
    3.00,
    5.00,
    10.00
]

# Select alpha based on mean cross-validation accuracy.
full_nb_search = GridSearchCV(
    estimator=full_nb_pipeline,
    param_grid={
        "nb__alpha": alpha_grid
    },
    scoring="accuracy",
    cv=inner_cv,
    refit=True,
    n_jobs=-1,
    return_train_score=True
)

full_nb_search.fit(
    X_train,
    Y_train
)

full_nb_test_pred = full_nb_search.predict(
    X_test
)

full_nb_test_prob = full_nb_search.predict_proba(
    X_test
)[:, 1]


print_heading(
    "CATEGORICAL NAIVE BAYES: FULL MODEL"
)

print(
    f"\nSelected alpha: "
    f"{full_nb_search.best_params_['nb__alpha']:g}"
)

print_metrics(
    y_true=Y_test,
    y_pred=full_nb_test_pred,
    y_prob=full_nb_test_prob
)

print_naive_bayes_equation(
    fitted_search=full_nb_search,
    feature_names=full_nb_features,
    original_categories=full_likert_categories
)


# ============================================================
# Reduced Categorical Naive Bayes model
# Feature: X1 only
# ============================================================

reduced_nb_features = ["X1"]

reduced_likert_categories = [
    [1, 2, 3, 4, 5]
    for feature in reduced_nb_features
]

reduced_nb_pipeline = Pipeline([
    (
        "encode",
        OrdinalEncoder(
            categories=reduced_likert_categories,
            dtype=np.int64
        )
    ),
    (
        "nb",
        CategoricalNB(
            min_categories=5
        )
    )
])

reduced_nb_search = GridSearchCV(
    estimator=reduced_nb_pipeline,
    param_grid={
        "nb__alpha": alpha_grid
    },
    scoring="accuracy",
    cv=inner_cv,
    refit=True,
    n_jobs=-1,
    return_train_score=True
)

reduced_nb_search.fit(
    X_train[reduced_nb_features],
    Y_train
)

reduced_nb_test_pred = reduced_nb_search.predict(
    X_test[reduced_nb_features]
)

reduced_nb_test_prob = reduced_nb_search.predict_proba(
    X_test[reduced_nb_features]
)[:, 1]


print_heading(
    "CATEGORICAL NAIVE BAYES: X1-ONLY MODEL"
)

print(
    f"\nSelected alpha: "
    f"{reduced_nb_search.best_params_['nb__alpha']:g}"
)

print_metrics(
    y_true=Y_test,
    y_pred=reduced_nb_test_pred,
    y_prob=reduced_nb_test_prob
)

print_naive_bayes_equation(
    fitted_search=reduced_nb_search,
    feature_names=reduced_nb_features,
    original_categories=reduced_likert_categories
)
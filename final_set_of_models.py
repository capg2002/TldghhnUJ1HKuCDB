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
    roc_auc_score,
    log_loss,
    confusion_matrix
)

# Loading data and variables
survey_df = pd.read_csv('ACME-HappinessSurvey2020.csv')
Y_var = survey_df["Y"]
X_full = survey_df[["X1", "X2", "X3", "X4", "X5", "X6"]]
X_reduced = survey_df[["X1", "X2", "X5"]]

seed = 22

# Creating train-test split

X_train, X_test, Y_train, Y_test = train_test_split(X_full, Y_var, test_size = 0.2, random_state=seed)

# Creating reduced X-var sets

X_train_reduced = X_train[["X1", "X2", "X5"]]
X_test_reduced = X_test[["X1", "X2", "X5"]]

# Ideal model threshold from predictive model exploration

model_threshold = 0.62

reduced_model = LogisticRegression() 

reduced_model = FixedThresholdClassifier(
    estimator=reduced_model,
    threshold=model_threshold,
    response_method= "predict_proba"
)

# Fitting logistic regression model

reduced_model.fit(X_train_reduced, Y_train)

Y_pred = reduced_model.predict(X_test_reduced)

Y_prbs = reduced_model.predict_proba(X_test_reduced)[:, 1]

# Defining stratified cross-validations

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=22
)

# Validated scores 

scores_reduced = cross_validate(
    reduced_model,
    X_reduced,
    Y_var,
    cv=cv,
    scoring=["accuracy", "roc_auc", "neg_log_loss"]
)

# ============================================================
# Reduced logistic regression results
# ============================================================

print("\n" + "=" * 60)
print("REDUCED LOGISTIC REGRESSION: X1, X2, X5")
print("=" * 60)

# The fitted LogisticRegression model is stored inside
# the FixedThresholdClassifier as estimator_.
fitted_logistic = reduced_model.estimator_

intercept = fitted_logistic.intercept_[0]
coefficients = fitted_logistic.coef_[0]

# Print the fitted log-odds equation.
print("\nFitted equation:")

print(
    f"log(p / (1 - p)) = "
    f"{intercept:.4f} "
    f"{coefficients[0]:+.4f}(X1) "
    f"{coefficients[1]:+.4f}(X2) "
    f"{coefficients[2]:+.4f}(X5)"
)

print(
    "\nwhere p = P(Y = 1 | X1, X2, X5)"
)

# The model predicts Y = 1 when the probability is at least 0.62.
print(
    f"Classification rule: predict Y = 1 when p >= "
    f"{model_threshold:.2f}"
)

print("\nTest-set results:")
print(f"Accuracy:          {accuracy_score(Y_test, Y_pred):.4f}")
print(f"ROC AUC:           {roc_auc_score(Y_test, Y_prbs):.4f}")
print(f"Log loss:          {log_loss(Y_test, Y_prbs):.4f}")

print("\nConfusion matrix:")
print(confusion_matrix(Y_test, Y_pred))

# cross_validate reports negative log loss because sklearn assumes
# that larger scoring values are better. Multiply by -1 to obtain
# the ordinary positive log-loss value.
print("\n5-fold cross-validation results:")
print(
    f"Mean accuracy: {scores_reduced['test_accuracy'].mean():.4f}"
)
print(
    f"Mean ROC AUC:  {scores_reduced['test_roc_auc'].mean():.4f}"
)
print(
    f"Mean log loss: {-scores_reduced['test_neg_log_loss'].mean():.4f}"
)



### Naive Bayes model, assuming independent answers given an observation for Y

feature_cols = list(
    X_train.columns
)

# Categories for each variable

likert_categories = [
    [1, 2, 3, 4, 5]
    for feature in feature_cols
]

categorical_nb_pipeline = Pipeline([
    (
        "encode",
        OrdinalEncoder( # Type of column, encoding with Likert categories
            categories=likert_categories,
            dtype=np.int64
        )
    ),
    (
        "nb",
        CategoricalNB( # Categorical Naive Bayes function
            min_categories=5
        )
    )
])

# Alpha comparison grid for Naive Bayes
alpha_grid = [ 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0 ]

# Search for best alpha with highest accuracy
nb_search = GridSearchCV(
    estimator=categorical_nb_pipeline,

    param_grid={
        "nb__alpha": alpha_grid
    },

    scoring="accuracy",
    cv=cv,
    refit=True,
    n_jobs=-1,
    return_train_score=True
)

# Fitting Naive Bayes
nb_search.fit(
    X_train,
    Y_train
)

nb_test_pred = nb_search.predict(
    X_test
)

nb_test_prob = nb_search.predict_proba(
    X_test
)[:, 1]

print("\n" + "=" * 60)
print("CATEGORICAL NAIVE BAYES: FULL MODEL")
print("=" * 60)

best_alpha = nb_search.best_params_["nb__alpha"]

print(f"\nBest alpha: {best_alpha}")

print("\nModel equation:")
print(
    "P(Y = c | X1, ..., X6) is proportional to\n"
    "P(Y = c) * P(X1 | Y = c) * P(X2 | Y = c) * "
    "P(X3 | Y = c) * P(X4 | Y = c) * "
    "P(X5 | Y = c) * P(X6 | Y = c)"
)

print("\nTest-set results:")
print(f"Accuracy:  {accuracy_score(Y_test, nb_test_pred):.4f}")
print(f"ROC AUC:   {roc_auc_score(Y_test, nb_test_prob):.4f}")
print(f"Log loss:  {log_loss(Y_test, nb_test_prob):.4f}")

print("\nConfusion matrix:")
print(confusion_matrix(Y_test, nb_test_pred))

# Access the fitted CategoricalNB model inside the best pipeline.
fitted_nb = nb_search.best_estimator_.named_steps["nb"]

# Convert stored log probabilities back into ordinary probabilities.
class_prior_probabilities = np.exp(
    fitted_nb.class_log_prior_
)

print("\nEstimated class probabilities:")

for class_value, probability in zip(
    fitted_nb.classes_,
    class_prior_probabilities
):
    print(
        f"P(Y = {class_value}) = {probability:.4f}"
    )

# feature_log_prob_ contains:
# P(Xj = category | Y = class)
#
# OrdinalEncoder converts original responses:
# 1, 2, 3, 4, 5
# into encoded categories:
# 0, 1, 2, 3, 4
print("\nConditional response probabilities:")

for feature_index, feature_name in enumerate(feature_cols):

    feature_probabilities = np.exp(
        fitted_nb.feature_log_prob_[feature_index]
    )

    print(f"\n{feature_name}:")

    for class_index, class_value in enumerate(
        fitted_nb.classes_
    ):
        print(f"  Given Y = {class_value}:")

        for category_index, original_response in enumerate(
            [1, 2, 3, 4, 5]
        ):
            probability = feature_probabilities[
                class_index,
                category_index
            ]

            print(
                f"    P({feature_name} = {original_response} "
                f"| Y = {class_value}) = {probability:.4f}"
            )



### Naive Bayes model, assuming independent answers given an observation for Y

feature_cols = list(
    X_train[["X1"]].columns
)

# Categories for each variable

likert_categories = [
    [1, 2, 3, 4, 5]
    for feature in feature_cols
]

categorical_nb_pipeline = Pipeline([
    (
        "encode",
        OrdinalEncoder( # Type of column, encoding with Likert categories
            categories=likert_categories,
            dtype=np.int64
        )
    ),
    (
        "nb",
        CategoricalNB( # Categorical Naive Bayes function
            min_categories=5
        )
    )
])

# Alpha comparison grid for Naive Bayes
alpha_grid = [ 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0 ]

# Search for best alpha with highest accuracy
nb_search = GridSearchCV(
    estimator=categorical_nb_pipeline,

    param_grid={
        "nb__alpha": alpha_grid
    },

    scoring="accuracy",
    cv=cv,
    refit=True,
    n_jobs=-1,
    return_train_score=True
)

# Fitting Naive Bayes
nb_search.fit(
    X_train[["X1"]],
    Y_train
)

# Best alpha is a = 2
print(
    "Best alpha:",
    nb_search.best_params_["nb__alpha"]
)

nb_test_pred = nb_search.predict(
    X_test[["X1"]]
)

nb_test_prob = nb_search.predict_proba(
    X_test[["X1"]]
)[:, 1]

# Accuracy is 0.730769, which is above adequate 

print("\n" + "=" * 60)
print("CATEGORICAL NAIVE BAYES: X1-ONLY MODEL")
print("=" * 60)

best_alpha = nb_search.best_params_["nb__alpha"]

print(f"\nBest alpha: {best_alpha}")

print("\nModel equation:")
print(
    "P(Y = c | X1) is proportional to "
    "P(Y = c) * P(X1 | Y = c)"
)

print("\nTest-set results:")
print(f"Accuracy:  {accuracy_score(Y_test, nb_test_pred):.4f}")
print(f"ROC AUC:   {roc_auc_score(Y_test, nb_test_prob):.4f}")
print(f"Log loss:  {log_loss(Y_test, nb_test_prob):.4f}")

print("\nConfusion matrix:")
print(confusion_matrix(Y_test, nb_test_pred))

fitted_nb = nb_search.best_estimator_.named_steps["nb"]

class_prior_probabilities = np.exp(
    fitted_nb.class_log_prior_
)

print("\nEstimated class probabilities:")

for class_value, probability in zip(
    fitted_nb.classes_,
    class_prior_probabilities
):
    print(
        f"P(Y = {class_value}) = {probability:.4f}"
    )

# Because this model has only X1, feature_log_prob_[0]
# contains the conditional probabilities for X1.
x1_probabilities = np.exp(
    fitted_nb.feature_log_prob_[0]
)

print("\nConditional probabilities for X1:")

for class_index, class_value in enumerate(
    fitted_nb.classes_
):
    print(f"\nGiven Y = {class_value}:")

    for category_index, original_response in enumerate(
        [1, 2, 3, 4, 5]
    ):
        probability = x1_probabilities[
            class_index,
            category_index
        ]

        print(
            f"P(X1 = {original_response} | "
            f"Y = {class_value}) = {probability:.4f}"
        )

import numpy as np
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, cross_val_predict, GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold, FixedThresholdClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import CategoricalNB

from sklearn.preprocessing import (
    StandardScaler,
    OrdinalEncoder
)

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    get_scorer_names,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    classification_report,
    confusion_matrix

)
import statsmodels.api as sm
from scipy.stats import chi2

survey_df = pd.read_csv('ACME-HappinessSurvey2020.csv')
Y_var = survey_df["Y"]
X_full = survey_df[["X1", "X2", "X3", "X4", "X5", "X6"]]

seed = 67

X_train, X_test, Y_train, Y_test = train_test_split(X_full, Y_var, test_size = 0.2, random_state=seed)

def iterate_subsets(iterable):
    s = list(iterable)
    # Loop through all possible lengths of subsets
    for r in range(1,(len(s) + 1)):
        # Generate and yield subsets of length r
        for subset in it.combinations(s, r):
            yield subset

lda_pipeline = Pipeline([
    (
        "scale",
        StandardScaler() # Standardizes terms, as assumed to be MVN
    ),
    (
        "lda",
        LinearDiscriminantAnalysis(
            solver="lsqr" # LDA function. lsqr to support shrinkage
        )
    )
])

shrinkage_grid = [ # Grid setup to test different shrinkage constants
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
]

outer_cv = RepeatedStratifiedKFold(
    n_splits = 5,
    n_repeats = 3,
    random_state = 68
)

inner_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=100
)

lda_search = GridSearchCV( # Testing different shrinkage coefficients to maximize accuracy
    estimator=lda_pipeline,

    param_grid={
        "lda__shrinkage": shrinkage_grid
    },

    scoring="accuracy",
    cv=inner_cv,
    refit=True,
    n_jobs=-1,
    return_train_score=True
)

# Fitting model
lda_search.fit(
    X_train,
    Y_train
)

# Calculating predicted values and probabilities

lda_test_pred = lda_search.predict(
    X_test
)

lda_test_prob = lda_search.predict_proba(
    X_test
)[:, 1]

print("\nSHRINKAGE LDA")

# Shrinkage LDA diagnostics

print(
    "Accuracy:",
    accuracy_score(
        Y_test,
        lda_test_pred
    )
)

print(
    "Balanced accuracy:",
    balanced_accuracy_score(
        Y_test,
        lda_test_pred
    )
)

print(
    "Precision:",
    precision_score(
        Y_test,
        lda_test_pred,
        zero_division=0
    )
)

print(
    "Recall:",
    recall_score(
        Y_test,
        lda_test_pred
    )
)

print(
    "F1:",
    f1_score(
        Y_test,
        lda_test_pred
    )
)

print(
    "ROC AUC:",
    roc_auc_score(
        Y_test,
        lda_test_prob
    )
)

print(
    "Log loss:",
    log_loss(
        Y_test,
        lda_test_prob
    )
)

print(
    "Brier score:",
    brier_score_loss(
        Y_test,
        lda_test_prob
    )
)

print("Confusion matrix:")
print(
    confusion_matrix(
        Y_test,
        lda_test_pred
    )
)

best_lda = (
    lda_search
    .best_estimator_
    .named_steps["lda"]
)

print(
    "Class priors:",
    best_lda.priors_
)

lda_means = pd.DataFrame(
    best_lda.means_,
    index=[
        f"class_{class_label}"
        for class_label
        in best_lda.classes_
    ],
    columns=X_train.columns
)

print(lda_means)

# Exhibits relatively weakly correlated variables. 
lda_covariance = pd.DataFrame(
    best_lda.covariance_,
    index=X_train.columns,
    columns=X_train.columns
)

print(lda_covariance)

# Full chosen model as
# P(Y | X_vars) = 0.492616*X1 - 0.150551*X2 - 0.105783*X3 + 0.114153*X4 + 0.358410*X5 - 0.059329*X6

lda_coefficients = pd.DataFrame({
    "feature": X_train.columns,
    "coefficient": best_lda.coef_[0]
})

print(
    lda_coefficients.sort_values(
        "coefficient",
        key=abs,
        ascending=False
    )
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
    cv=inner_cv,
    refit=True,
    n_jobs=-1,
    return_train_score=True
)

# Fitting Naive Bayes
nb_search.fit(
    X_train,
    Y_train
)

# Best alpha is a = 2
print(
    "Best alpha:",
    nb_search.best_params_["nb__alpha"]
)

nb_test_pred = nb_search.predict(
    X_test
)

nb_test_prob = nb_search.predict_proba(
    X_test
)[:, 1]

print("\nCATEGORICAL NAIVE BAYES")

# Accuracy is 0.730769, which is above adequate 

print(
    "Accuracy:",
    accuracy_score(
        Y_test,
        nb_test_pred
    )
)

print(
    "Balanced accuracy:",
    balanced_accuracy_score(
        Y_test,
        nb_test_pred
    )
)

print(
    "Precision:",
    precision_score(
        Y_test,
        nb_test_pred,
        zero_division=0
    )
)

print(
    "Recall:",
    recall_score(
        Y_test,
        nb_test_pred
    )
)

print(
    "F1:",
    f1_score(
        Y_test,
        nb_test_pred
    )
)

print(
    "ROC AUC:",
    roc_auc_score(
        Y_test,
        nb_test_prob
    )
)

print(
    "Log loss:",
    log_loss(
        Y_test,
        nb_test_prob
    )
)

print(
    "Brier score:",
    brier_score_loss(
        Y_test,
        nb_test_prob
    )
)

print("Confusion matrix:")
print(
    confusion_matrix(
        Y_test,
        nb_test_pred
    )
)

best_nb = (
    nb_search
    .best_estimator_
    .named_steps["nb"]
)

print(
    "Classes:",
    best_nb.classes_
)

print(
    "Class probabilities:",
    np.exp(
        best_nb.class_log_prior_
    )
)

for feature_index, feature_name in enumerate(
    feature_cols
):
    probability_table = pd.DataFrame(
        np.exp(
            best_nb.feature_log_prob_[
                feature_index
            ]
        ),
        index=[
            f"class_{class_label}"
            for class_label
            in best_nb.classes_
        ],
        columns=[1, 2, 3, 4, 5]
    )

    print(f"\n{feature_name}")
    print(
        probability_table.round(3)
    )


comparison = pd.DataFrame({
    "Model": [
        "Shrinkage LDA",
        "Categorical Naive Bayes"
    ],

    "Accuracy": [
        accuracy_score(
            Y_test,
            lda_test_pred
        ),
        accuracy_score(
            Y_test,
            nb_test_pred
        )
    ],

    "Balanced Accuracy": [
        balanced_accuracy_score(
            Y_test,
            lda_test_pred
        ),
        balanced_accuracy_score(
            Y_test,
            nb_test_pred
        )
    ],

    "Precision": [
        precision_score(
            Y_test,
            lda_test_pred,
            zero_division=0
        ),
        precision_score(
            Y_test,
            nb_test_pred,
            zero_division=0
        )
    ],

    "Recall": [
        recall_score(
            Y_test,
            lda_test_pred
        ),
        recall_score(
            Y_test,
            nb_test_pred
        )
    ],

    "F1": [
        f1_score(
            Y_test,
            lda_test_pred
        ),
        f1_score(
            Y_test,
            nb_test_pred
        )
    ],

    "ROC AUC": [
        roc_auc_score(
            Y_test,
            lda_test_prob
        ),
        roc_auc_score(
            Y_test,
            nb_test_prob
        )
    ],

    "Log Loss": [
        log_loss(
            Y_test,
            lda_test_prob
        ),
        log_loss(
            Y_test,
            nb_test_prob
        )
    ]
})

print(
    comparison.round(3)
)
outer_lda_results = []
outer_nb_results = []

for train_index, test_index in outer_cv.split(X_full, Y_var):
        
        X_train_val, X_test_val = X_full.iloc[train_index], X_full.iloc[test_index]
        Y_train_val, Y_test_val = Y_var.iloc[train_index], Y_var.iloc[test_index]

        lda_subset_scores = []
        nb_subset_scores = []

        for sub in iterate_subsets(X_train_val.columns):

            X_train_sub, X_test_sub = X_train_val.loc[:, sub], X_test_val.loc[:, sub]

            lda_search.fit(
                X_train_sub,
                Y_train_val
            )

            lda_subset_scores.append({
                "subset": sub,
                "score": lda_search.best_score_,
                "model": lda_search.best_estimator_,
                "params": lda_search.best_params_
            })

            feature_cols = list(
                X_train_sub.columns
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

            nb_search = GridSearchCV(
                estimator=categorical_nb_pipeline,

                param_grid={
                    "nb__alpha": alpha_grid
                },

                scoring="accuracy",
                cv=inner_cv,
                refit=True,
                n_jobs=-1,
                return_train_score=True
            )

            nb_search.fit(
                X_train_sub,
                Y_train_val
            )

            nb_subset_scores.append({
                "subset": sub,
                "score": nb_search.best_score_,
                "model": nb_search.best_estimator_,
                "params": nb_search.best_params_
            })

        best_lda_subset = max(
            lda_subset_scores,
            key=lambda result: (
                result["score"],
                -len(result["subset"])
            )
        )

        best_nb_subset = max(
            nb_subset_scores,
            key=lambda result: (
                result["score"],
                -len(result["subset"])
            )
        )

        lda_test_val_pred = best_lda_subset[
            "model"
        ].predict(
            X_test_val.loc[
                :, best_lda_subset["subset"]
            ]
        )

        outer_lda_results.append({
            "subset": best_lda_subset["subset"],
            "params": best_lda_subset["params"],
            "accuracy": accuracy_score(
                Y_test_val, lda_test_val_pred
            )
        })

        nb_test_val_pred = best_nb_subset[
            "model"
        ].predict(
            X_test_val.loc[
                :, best_nb_subset["subset"]
            ]
        )

        outer_nb_results.append({
            "subset": best_nb_subset["subset"],
            "params": best_nb_subset["params"],
            "accuracy": accuracy_score(
                Y_test_val, nb_test_val_pred
            )
        })

lda_outer_df = pd.DataFrame(
    outer_lda_results
)
nb_outer_df = pd.DataFrame(
    outer_nb_results
)

print(
    lda_outer_df["accuracy"].describe()
)
print(
    nb_outer_df["accuracy"].describe()
)

print(
    lda_outer_df["subset"].value_counts()
)
print(
    nb_outer_df["subset"].value_counts()
)

accuracy_difference = (
    nb_outer_df["accuracy"]
    - lda_outer_df["accuracy"]
)
print(accuracy_difference.describe())

# print(nb_best_params)
# print(lda_best_params)

# Categorical Naive Bayes achieved the highest performance on the held-out test set, 
# with an accuracy of 0.769 and ROC AUC of 0.840. However, repeated nested cross-validation 
# showed substantial variability, with mean accuracy near 0.56–0.57 for both models. The average 
# difference between the two models was less than one percentage point, indicating that neither 
# method demonstrated consistently superior out-of-sample accuracy across resamples.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    cross_validate
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

survey_df = pd.read_csv('ACME-HappinessSurvey2020.csv')
Y_var = survey_df["Y"]
X_full = survey_df[["X1", "X2", "X3", "X4", "X5", "X6"]]

seed = 100

X_train, X_test, Y_train, Y_test = train_test_split(X_full, Y_var, test_size = 0.2, random_state=seed)

en_pipeline = Pipeline([
    ( "scale", StandardScaler() ),
    ( "logistic", LogisticRegression(solver = "saga", l1_ratio = 0.5, C = 1.0, max_iter=20000, random_state=seed) )
])

C_grid = np.logspace(-3, 3, 15)

l1_ratio_grid = []

for i in range(20):
    l1_ratio_grid += [round(0 + 0.05*i, 3)]

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=seed
)

en_search = GridSearchCV(
    estimator=en_pipeline,
    param_grid={
        "logistic__C": C_grid,
        "logistic__l1_ratio": l1_ratio_grid
    },
    scoring=["accuracy", "precision", "balanced_accuracy", "recall", "f1", "roc_auc", "neg_log_loss", "d2_brier_score"],
    refit="neg_log_loss",
    cv=cv,
    n_jobs=-1,
    return_train_score=True
)

en_search.fit(X_train, Y_train)

print(
    "Best C:",
    en_search.best_params_[
        "logistic__C"
    ]
)

print(
    "Best l1_ratio:",
    en_search.best_params_[
        "logistic__l1_ratio"
    ]
)

print(
    "Best CV log loss:",
    -en_search.best_score_
)
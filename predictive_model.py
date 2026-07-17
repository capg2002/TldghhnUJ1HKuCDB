import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, cross_val_predict, StratifiedKFold, FixedThresholdClassifier
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

seed = 1900

X_train, X_test, Y_train, Y_test = train_test_split(X_full, Y_var, test_size = 0.2, random_state=seed)

full_model = LogisticRegression() 

model_threshold = 0.62

full_model = FixedThresholdClassifier(
    estimator=full_model,
    threshold=model_threshold,
    response_method= "predict_proba"
)

full_model.fit(X_train, Y_train)
full = sm.Logit(Y_var, X_full).fit()

Y_pred = full_model.predict(X_test)

Y_prbs = full_model.predict_proba(X_test)[:, 1]

Y_var = survey_df["Y"]
X_reduced = survey_df[["X1", "X2", "X5"]]

X_train_reduced = X_train[["X1", "X2", "X5"]]
X_test_reduced = X_test[["X1", "X2", "X5"]]

reduced_model = LogisticRegression() 

reduced_model = FixedThresholdClassifier(
    estimator=reduced_model,
    threshold=model_threshold,
    response_method= "predict_proba"
)

reduced_model.fit(X_train_reduced, Y_train)
reduced = sm.Logit(Y_var, X_reduced).fit()

Y_pred = reduced_model.predict(X_test_reduced)

Y_prbs = reduced_model.predict_proba(X_test_reduced)[:, 1]


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores_full = cross_validate(
    full_model,
    X_full,
    Y_var,
    cv=cv,
    scoring=["accuracy", "precision", "balanced_accuracy", "recall", "f1", "roc_auc", "neg_log_loss", "d2_brier_score"]
)

scores_reduced = cross_validate(
    reduced_model,
    X_reduced,
    Y_var,
    cv=cv,
    scoring=["accuracy", "precision", "balanced_accuracy", "recall", "f1", "roc_auc", "neg_log_loss", "d2_brier_score"]
)

scores_full_names = list(scores_full.keys())
# AUC is the frequentist probability that a true positive is assigned
# a higher probability than a true negative. Generally,
# we want a higher AUC.

# Log loss corresponds to the log sum of our loss metric. Generally,
# the lower the loss, the better.

for name in scores_full_names:
    print("Full", name, ":", scores_full[name].mean())
    print("Reduced", name, ":", scores_reduced[name].mean())
    print("Reduced vs full", name, "difference:", scores_reduced[name].mean() - scores_full[name].mean())



print(full.summary())
print(reduced.summary())

# For all metrics, the reduced model statistically improves the model's predictive ability
# Through additional 

lr_stat = 2 * (full.llf - reduced.llf)
df_difference = full.df_model - reduced.df_model
p_value = chi2.sf(lr_stat, df_difference)

print("Likelihood-ratio statistic:", lr_stat)
print("Degrees of freedom:", df_difference)
print("p-value:", p_value)

# Note that there is no evidence that adding X3, X4 or X6 meaningfully improves the likelihood.
# This, combined with the previous evaluations, implies that the reduced model performs better.
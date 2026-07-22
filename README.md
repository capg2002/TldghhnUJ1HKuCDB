# ACME Happiness Prediction Models

## 1. Summary of the Process

The goal of this project was to predict whether a survey respondent was happy (`Y = 1`) using six Likert-scale survey questions (`X1`-`X6`).

I first compared a full logistic regression using all six questions with a reduced logistic regression using only `X1`, `X2`, and `X5`. I also tested different probability thresholds instead of automatically using the standard 0.50 threshold. Repeated nested cross-validation was used to select the feature set and threshold without using the test data during model selection. This process produced an average selected threshold of approximately **0.55**.

Elastic-net logistic regression, shrinkage linear discriminant analysis, and categorical Naive Bayes were also tested as alternative models. Performance was evaluated using several classification metrics. Because the dataset is small, repeated cross-validation results were given more importance than the results from any single train-test split.

## 2. Models and Final Decision

### Reduced Logistic Regression

The reduced logistic regression uses `X1`, `X2`, and `X5` with a classification threshold of **0.55**. It performed similarly to or better than the full logistic regression while using fewer variables. The additional variables `X3`, `X4`, and `X6` did not provide clear evidence of improving the model.

This model is easy to explain, uses information from multiple survey questions, and produced more consistent results across validation samples.

### Categorical Naive Bayes

Categorical Naive Bayes was tested because the predictors are five-point categorical survey responses. It produced the strongest result on one test set, with an accuracy of approximately **76.9%** and a ROC AUC of **0.840**.

However, repeated nested cross-validation showed that this result was not consistent across different samples. Its average performance was closer to the other models, meaning that the strong test result may have depended on a favourable train-test split.

### Other Models

Elastic-net logistic regression did not improve performance enough to justify its additional complexity. Shrinkage LDA also did not outperform the simpler logistic regression models and relies on assumptions that are less appropriate for categorical Likert-scale data.

### Final Decision

The **reduced logistic regression using `X1`, `X2`, and `X5` with a 0.55 threshold** is recommended as the final model because it provides the best balance between interpretability, simplicity, and consistency.

Categorical Naive Bayes should be kept as an alternative model for future testing because it showed the potential for higher accuracy, but more data would be needed before concluding that it performs better consistently. 

The Categorical Naive Bayes model scored an accuracy above **76.9%**, while the reduced logistic regression scored an accuracy, on average, around **60.0%**, which is lower than the expected threshold of 73%, but might be easier to implement in the future. 

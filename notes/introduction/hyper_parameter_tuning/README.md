# Hyper-Parameter Tuning

To choose between different models or determine the optimal hyperparameter value, we can use holdout validation. Instead of relying solely on the test set, we create a separate validation set by setting aside a portion of the training set.

First, we train multiple models with different hyperparameters on the reduced training set (training set minus the validation set). Next, we evaluate the performance of each model on the validation set and select the one with the best performance. This selection process helps us avoid overfitting to the test set.

After choosing the best model through holdout validation, we train it on the full training set, including the validation set. This final model is then evaluated on the test set to estimate its generalization error, providing an assessment of its performance on new, unseen data.

By using holdout validation, we prevent the risk of overfitting the model to the test set and ensure that our final model has a better chance of performing well on new data.

## Cross-Validation

While holdout validation is effective, its success depends on the size of the validation set. If the validation set is too small, evaluations may be inaccurate, leading to the selection of a suboptimal model. Conversely, if the validation set is too large, the remaining training set becomes significantly smaller. This is problematic because the final model will be trained on the full training set, making it inappropriate to compare models trained on a much smaller subset.

To address this issue, repeated cross-validation can be employed. This approach involves using multiple small validation sets. Each model is trained on the remaining data and evaluated once on each validation set. By repeating this process, we obtain a more reliable and robust assessment of the models' performance, as they are evaluated on different subsets of the data.

Performing repeated cross-validation helps ensure that the model selection process is not biased by an inadequate validation set size and allows for a more accurate comparison of candidate models.
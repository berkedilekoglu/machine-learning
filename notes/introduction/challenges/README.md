# Challenges In Machine Learning

These are the some challenges in machine learning

## Data Quantity

To teach a baby what a pen is, you need to show them pens. For the human brain, this learning can occur with a few examples. However, in machine learning, the situation is not exactly the same. Learning improves when we have a sufficient number of examples, ranging from the simplest to the most powerful algorithms. Simply put, for a classification task, the more examples we have for each label in our training data, the better our performance will be.

## Poor-Quality Data

Noises, outliers and, errors are common problems for datasets. If our data consists of with a great number of problematic samples model learn them as well. Thus, the performance of the model will be affected. Therefore, we need to clear our data from these bad samples.
## Irrelevant Features
To enable learning, your system requires training data with sufficient relevant features and limited irrelevant ones. The success of a Machine Learning project heavily relies on selecting a suitable set of features for training, which involves the process of feature engineering. This process includes steps such as 
- _**Feature selection**_:the most useful features are chosen from existing ones
- _**Feature extraction**_ : combining existing features to create more valuable ones (where dimensionality reduction algorithms can be helpful)
- _**Generating new features**_: by gathering additional data. 

These steps collectively contribute to improving the effectiveness and performance of the learning process.

## Overfitting the Training Data
Overfitting happens when the model is too complex relative to the amount and noisiness of the training data, performing extremely well on the training set but failing to generalize well to new, unseen data. It happens when the model learns the noise or random fluctuations in the training data, rather than the underlying patterns or relationships. 

Here are possible solutions:
- Simplify the model  (e.g., a linear model rather than a high-degree polynomial model), by reducing the number of attributes by constraining the model.
- Gather more training data.
- Reduce the noise in the training data (e.g., fix data errors and remove outliers).
- Cross-validation and early stopping are also commonly used to detect and prevent overfitting.

Constraining a model to make it simpler and reduce the risk of overfitting is called _**Regularization**_. The amount of regularization to apply during learning can be controlled by a hyper‐parameter. 
> A hyperparameter is a parameter of a learning algorithm (not of the model) such as learning rate, batch size, epoch number etc.

## Underfitting the Training Data

It is opposite of overfitting. _**Underfitting**_ occurs when a machine learning model is too simple or lacks the capacity to capture the underlying patterns or relationships in the data. It typically results in poor performance on both the training data and new, unseen data.

Here are possible solutions:
- Increasing the complexity of the model by selecting a more powerful model, with more parameters.
- Feed better features to the learning algorithm (feature engineering).
- Reduce the constraints on the model (e.g., reduce the regularization hyperparameter).

> The aim is to strike a balance between model complexity and generalization performance, avoiding both underfitting and overfitting, to achieve optimal predictive performance on new, unseen data.
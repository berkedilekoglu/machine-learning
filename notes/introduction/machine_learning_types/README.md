# Machine Learning Types

There are many types of the machine learning algorithms.

- Machine learning algorithms can be categorized based on whether they are trained with human supervision or not. This includes **_supervised learning, unsupervised learning, semi-supervised learning, and reinforcement learning_**.

- Another factor to consider is whether the algorithms can learn incrementally as new data arrives **_(online learning)_** or if they require all the data to be available at once for training **_(batch learning)_**.

- Additionally, there is a distinction between algorithms that compare new data points to known data points to make predictions **_(instance-based learning)_** and algorithms that detect patterns in the training data and build a predictive model based on those patterns **_(model-based learning)_**

## Supervised/Unsupervised Learning

### Supervised Learning

In supervised learning, models are trained with human supervision. We have desired solutions for each training data instance, which are called labels. Two common tasks are classification and regression. 

We can say that the task is a **_classification_** task if we have finite number of labels. The spam filter is a good example of this: it is trained with many example emails along with their class (spam or ham), and it must learn how to classify new emails.

The task is a _**regression task**_ if we try to predict numbers (so we can say that we have infinite number of labels) such as the price of the houses. To train the system, you need to give it many examples of houses, including both their features and their labels (i.e., their prices).

> Logistic Regression is a classification algorithm that uses  logistic function, which maps the input values to a probability value between 0 and 1. Thus, a threshold can be used to make binary classification. You can use it also multi-class classification problems by [sklearn lib](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

Here are some examples for Supervised Learning Algorithms:
- [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [Random Forests](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Support Vector Machines (SVM)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [K-Nearest Neighbors (KNN)](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [Gradient Boosting Machines (GBM)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
- [Neural Networks (Multi-layer Perceptron)](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
- [Extreme Gradient Boosting (XGBoost)](https://xgboost.readthedocs.io/en/stable/)

### Unsupervised Learning
When the machine learning model tries to learn without human supervision, which means that when you have no labels in your training data, this is a unsupervised learning.
These are the most common used examples:
- **_Clustering_**
— K-Means
— DBSCAN
— Hierarchical Cluster Analysis (HCA)
- **_Anomaly detection and novelty detection_**
— One-class SVM
— Isolation Forest
- **_Visualization and dimensionality reduction_**
— Principal Component Analysis (PCA) 
— Kernel PCA
— Locally Linear Embedding (LLE)
— t-Distributed Stochastic Neighbor Embedding (t-SNE)
- **_Association rule learning_**
— Apriori 
— Eclat

> KNN is sometimes mistaken as an unsupervised learning algorithm due to its name. K-Nearest Neighbors (KNN) is a supervised learning algorithm. It is used for both classification and regression tasks. In the case of classification, KNN determines the class of a new instance based on the majority class of its k nearest neighbors. In the case of regression, KNN predicts the value of a new instance based on the average of the values of its k nearest neighbors. In both cases, KNN requires labeled training data to learn and make predictions. Therefore, KNN is a supervised learning algorithm.

> Before feeding your training data to another Machine Learning algorithm, such as a supervised learning algorithm, it is often beneficial to employ a dimensionality reduction algorithm to decrease its dimension. This can lead to faster processing, reduced disk and memory usage, and potentially improved performance in certain scenarios.

### Semisupervised learning

Unfortunately, we have not enough labeled data for many scenario in real life. In addition to that, labeling is a time consuming process. Some algorithms can deal with data that’s partially labeled. This is called semisupervised learning.

In semisupervised learning, you begin with a small labeled dataset and a larger unlabeled dataset. The labeled data provides initial knowledge about the problem, while the unlabeled data offers additional information to improve the learning process. The model learns from the labeled examples and uses the unlabeled examples to generalize and make predictions on unseen data.

By incorporating unlabeled data, semisupervised learning can potentially achieve higher accuracy and better generalization compared to using only labeled data. It is particularly useful when obtaining labeled data is challenging or expensive, as it allows you to make the most of the available resources and unlabeled data.

_**Deep belief networks (DBNs)**_ is an example for semisupervised learning algorithm.

### Reinforcement Learning

Reinforcement Learning is a distinct approach in machine learning. In this context, the learning system, referred to as an agent, has the ability to observe its environment, make decisions, and take actions. Based on these actions, the agent receives rewards (or penalties) as feedback. The goal of the agent is to autonomously discover the optimal strategy, known as a policy, that maximizes the total reward over time. The policy guides the agent on which action to select in different situations, enabling it to learn and improve its decision-making process.

## Batch and Online Learning

In _**batch learning**_, the system cannot learn gradually or incrementally. It requires training with the entire available data at once. This process usually requires significant time and computational resources, so it is commonly performed offline. The system is trained first, and once deployed, it operates without further learning. It simply applies the knowledge it has acquired. This approach is referred to as offline learning.


_**Online learning**_ enables incremental training by feeding data instances sequentially using mini batches. It suits systems with continuous data flow and rapid adaptation needs. It saves space by discarding old data after learning. The _learning rate_ controls adaptation speed; high rates adapt fast but risk neglecting old data, while low rates provide stability against noise and outliers. Bad data can harm performance, requiring close monitoring and timely action. Malfunctioning sensors or spamming can be sources of bad data. Monitoring, turning off learning, or reverting to a previous state can mitigate risks. Anomaly detection helps identify abnormal data. In summary, online learning offers adaptive training, space savings, but requires vigilant monitoring and data handling.

## Instance Based VS Model Based Learning

In _**instance-based learning**_, the system determines the output for a new instance by comparing it to the stored instances in the training data. It finds the most similar instances and uses their labels or values to make predictions or decisions. The similarity between instances is typically measured using distance metrics.

> Popular instance-based learning algorithms include k-nearest neighbors (KNN)

**_Model-based learning_** is a machine learning approach where a model is constructed during the training phase to represent the underlying patterns and relationships in the data. The model serves as a generalized representation of the training data and can be used to make predictions or decisions on new, unseen data.


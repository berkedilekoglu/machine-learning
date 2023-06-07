# Training, Testing and Validating

These are the 3 main steps in machine learning project. First of all we have a dataset and we need to train our model with the best parameters to achieve the best performance. The most important point is measuring the performance of the model. Our model will work on unseen data in the real life. Therefore, we need to measure its performance on unseen data as well. The first step is randomly seperating your data as Training and Testing. Thus, you can be sure that your test data will be unseen until the end of the training phase. After that point you can focus on training. So what is training:
> You will set your model parameters and learning parameters to start training. But how can we choose those parameters ?

Good hyper-parameters are important to achive higher performances. Finding the best hyper-parameters is called _**hyper-parameter tuning**_. This process basically consists of trying different parameters and measuring their performance. 

So for each parameter we need to train our model and measure its performance. Now again we need to consider that this performance should be measured on unseen data. Therefore, we need to randomly split our training data to create validation data. Another (Better) solution for that is using cross-validation.

The difference between the error rate on the training set and the test set is known as the generalization error or out-of-sample error.

If the model has a low error rate on the training set but a high generalization error, it indicates overfitting. Overfitting occurs when the model becomes too specialized to the training data and fails to generalize well to new, unseen instances.

While deploying the model in a production environment and monitoring its performance is one way to assess generalization, it may not be ideal if the model performs poorly. Therefore, using a separate test set allows us to estimate the model's performance on new instances without impacting user experience.
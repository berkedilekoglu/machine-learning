# ml-notes

- Türkçe Machine Learning Notları.
- Terimler ingilizce bırakılmış konular türkçe anlatılmıştır.
- Sadece Introduction kısmı ingilizcedir. Daha sonra çevrilecektir.

## Konular ##

- [Introduction (Bu kısım ingilizce daha sonra çevrilecek)](https://github.com/berkedilekoglu/machine-learning/tree/main/notes/introduction)
    - [Machine Learning Types](https://github.com/berkedilekoglu/machine-learning/tree/main/notes/introduction/machine_learning_types)
        - Supervised/Unsupervised Learning
        - Semisupervised learning
        - Reinforcement Learning
        - Batch and Online Learning
        - Instance Based VS Model Based Learning
    - [Challanges in Machine Learning](https://github.com/berkedilekoglu/machine-learning/tree/main/notes/introduction/challenges#challenges-in-machine-learning)
        - Data Quantity
        - Poor-Quality Data
        - Irrelevant Features
        - Overfitting
        - Underfitting
    - [Training, Validation and Test](https://github.com/berkedilekoglu/machine-learning/tree/main/notes/introduction/train_test_validation)
    - [Hyper Parameter Tuning](https://github.com/berkedilekoglu/machine-learning/tree/main/notes/introduction/hyper_parameter_tuning)
        - Cross-Validation
    
- [Decision Trees](https://github.com/berkedilekoglu/machine-learning/blob/main/notes/Decision_Trees/README.md#decision-trees)
    - Karar ağaçlarında eğitim nasıl yapılır ?
    - Entorpy nedir ve nasıl hesaplanır ?
    - Information Gain nedir ve nasıl hesaplanır ?
    - Gini Index nedir ve nasıl hesaplanır ?
    - Gini vs Entorpy
    - Regularization
        - Early Stopping
        - Pruning
    - Decision Tree Avantaj ve Dezavantajları nelerdir ?
    
- [Ensemble Learning](https://github.com/berkedilekoglu/machine-learning/tree/main/notes/Ensemble_Learning#ensemble-learning)
    - Ensemble Learning nedir ?
    - Voting nedir ?
        - Hard Voting (Majority Voting)
        - Soft Voting
    - Bagging (Bootstrap aggregating) ve Pasting
    - Out-of-Bag Evaluation / Score
    - More Diversity (Feature Selection)
    - Random Forest
    - Extra Trees
    - Boosting (Hypothesis Boosting)
        - AdaBoost
        - Gradient Boosting
    - Stacking
- [Training Models](https://github.com/berkedilekoglu/machine-learning/tree/main/notes/Training_Models#training-models)
    - [Gradient Descent](https://github.com/berkedilekoglu/machine-learning/tree/main/notes/Gradient_Descent)
        - Gradient nedir ve nasıl kullanılır ?
        - Matematiksel olarak Gradient hesaplamak
        - MSE Loss ve Linear Regression ile weight yenilemek
        - Farklı Loss Fonksiyonlarındaki Zorluklar
        - Local Minimum, Global Minimum ve Plato nedir ?
    - [Farklı Gradient Descent Yöntemlerinin Karşılaştırılması](https://github.com/berkedilekoglu/machine-learning/tree/main/notes/Training_Models/Batch_Stochastic_Mini_Batch_GD#farkl%C4%B1-gradient-descent-y%C3%B6ntemlerinin-kar%C5%9F%C4%B1la%C5%9Ft%C4%B1r%C4%B1lmas%C4%B1)
        - Epoch
        - Batch Size
        - Farklı Gradient Descent Yöntemleri
        - Batch Gradient Descent
        - Stochastic Gradient Descent
        - Mini-Batch Gradient Descent
        - 3 farklı gradient descent yönteminin linear regression ve MSE loss kullanılarak python ile kodlanması
    - [Bias Variance Trade-Off](https://github.com/berkedilekoglu/machine-learning/tree/main/notes/Training_Models/Bias_Variance_Overfitting_Underfitting#bias-variance-tradeoff)
        - Bias (Ön Yargı)
        - Variance (Çeşitlilik)
        - Overfitting 
            - Nedenleri ve nasıl üstesinden gelineceği
        - Underfitting
            - Nedenleri ve nasıl üstesinden gelineceği
        - Trade-off
    - [Polynomial Regression](https://github.com/berkedilekoglu/machine-learning/blob/main/notes/Training_Models/Polynomial_Regression/Readme.md#regression)
        - Polynomial Regression
        - Sklearn PolynomialFeatures fonksiyonu nasıl çalışır ?
        - Linear Regression vs Polynomial Regression
        - Curse of Dimensionality
        

## References
- [GeÌ ron, A. (2019). Hands-on machine learning with Scikit-Learn, Keras and TensorFlow: concepts, tools, and techniques to build intelligent systems (2nd ed.).](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
- [Jason Brownlee PhD](https://machinelearningmastery.com/)
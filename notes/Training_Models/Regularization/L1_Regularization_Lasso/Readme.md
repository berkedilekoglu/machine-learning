## L1 Regularization (Lasso) ##

>> Not: Daha önce girdiğim ML Engineer mülakatında overfitting konularından sonra Lasso Regularization ve feature seçmeye yaraması sorulmuştu. Bu yazıda net bir şekilde bunları öğrenebilirsiniz.

L1 Regularization ismi L1 Norm'dan gelmektedir. L1 Norm'un matematiksel denklemine bakacak olursak:

$$\text{w vektörünün L1 norm'u (Manhattan norm) } \mathbf{w}:||\mathbf{w}||_1 = \sum_{i=1}^{n} |w_i|$$

Lasso ismi ise *Least Absolute Shrinkage and Selection Operator*'dan gelmektedir. L1 norm absolute value yani mutlak değer kullanarak Loss fonksiyonuna etki ettiğinden Least Absolute Shrinkage, feature selection yaptığı için ise Selection Operator tanımları verilmiştir.

L1 Regularizationdaki amacımız Loss fonksiyonuna bir parametre daha ekleyerek yeni belirlenecek değerlerin sadece model parametrelerine ve datamıza göre şekillenmesini önlemektir. Bu sebeple aslında herhangi bir modele L1 Regularization eklediğinizde şunu yapmış olursunuz:

$$\text{Loss = Error +} \lambda\sum_{i=1}^{n} |w_i|$$

Burada lambda değeri bir hyperparametredir ve regularization'ın ne derecede etki edeceğini belirler. Denklemden de gördüğünüz gibi artık Loss değerimiz sadece modele ve dataya değil aynı zamanda lambda değerine de bağlı. Peki buradaki mantık nedir. Şöyle düşünün Loss değerimiz sadece model ve data ile belirlendiğinde modelimiz datamızı tamamen ezberleyip loss'u 0'a indirgemeye çalışacaktır. Yani Weightlerimiz öyle bir ayarlanacak ki trainingde verilen dataya tamamen uyumlu olacak ve loss en aza indirgenecek. Ancak böyle bir uyum beraberinde overfitting'i de getirecektir. İşte burada işin içine __Lambda__ hyperparametresini sokarak weightleri cezalandırdığımızda datayı ezberleme işini sekteye uğratmış oluyoruz. 

Şimdi aşağıdaki figürde linear regression üzerinde bir örnek yapacağız ve L1 Regularization'ın ne işe yaradığını çok daha net göreceğiz. Bu figürdeki Regression aynı zamanda __Lasso Regression__ olarak geçmektedir.

![Note 18 Aug 2027](https://github.com/berkedilekoglu/machine-learning/assets/19657350/43422a26-31bc-43b4-a554-7030eb103d4e)

Gelin sırayla figürdeki kısımlar üzerinden gidelim:

1. kısımı [Gradient Descent](https://github.com/berkedilekoglu/machine-learning/blob/main/notes/Training_Models/Gradient_Descent/Readme.md) konusunu işlerken gördük. Linear regression denklemi ve denklemdeki parametrelerin neler oldukları. Burada __W__ ve __b__ vektörlerini öğreniyoruz.
2. kısımda Linear regression üzerinde bir örnek yapacağımızdan yine MSE ile loss hesaplayacağız.
3. kısım MSE ve L1 Regularization'ın birleşmesiyle yani L1 Regularization kullandığımızda nasıl Loss hesaplayacağımızı gösteriyor.
4. kısımda artık 3. kısımda belirlediğimiz loss fonksiyonumuzun türevini alıcaz. Burada dikkat ederseniz türev alırken MSE ve L1 Reg'i ayırdım. Çünkü l1 Regularization'ın etkisini görmek istiyoruz. Bu sebeple MSE'nin türevini __psi__ altında birleştirdim. 
5. Şimdi burada esas mantığı anlamaya çalışacağımızdan kafamızı karıştıracak ve mantığı değiştirmeyecek şeyleri yok sayıcaz. İlk önce learning rate'i yok sayalım. İkinci olarak __W - Psi__'yi __Beta__'ya eşitleyelim. Şuna dikkat edin aslında L1 Regularization kullanmasak __Beta__ update edeceğimiz yeni __Weight__ vektörümüz olacaktı. Yani __Beta__'yı bir weight vektörü olarak düşünebilirim. Şimdi neler biliyorum: İlk olarak __Beta__'nın weight vektörü olduğunu biliyorum. İkinci olarak ise Regularization hyperparametresinin pozitif bir değer olduğunu biliyorum (Tabi kullanacaksam kullanmayacaksam 0'da olabilir). Denkleme bakarsak weightler pozitifken __Beta__'dan yani weightlerden __Lambda__'yı çıkartıyoruz. Pozitif bir sayıdan pozitif bir sayıyı çıkartırsak sıfıra doğru yanaşırız. weightler negatifken __Beta__ ile yani weightler ile __Lambda__'yı topluyoruz. Negatif bir sayı ile pozitif bir sayıyı toplarsak yine sıfıra doğru yanaşırız. Buradan çıkarılacak sonuç lambda bize weightleri sıfıra doğru yaklaştırma (hatta sıfır yapma) gücü verir.

Peki weightleri sıfıra yanaştırmak ne anlama geliyor. Aslında figürün en altında bunun örneği de var. Featurelara denk gelen weightleri sıfıra yakınlaştırmak demek o featureları çok küçük katsayılarla çarpmak yani yok saymak demektir. 

Yani L1 Regularization ne işe yarar derseniz özetle:
1) Loss'un sadece model parametreleri ve datadan hesaplanmasını önleyerek lambda parametresi ve weightlerin mutlak değerlerinin toplamı ile Loss'a etki eder. Bu başlı başına overfitting'i engellemek için bir yöntemdir.
2) Weightleri sıfıra çekerek gereksiz featureları eler.
3) Gereksiz featureların elenmesi [Polynomial Regression](https://github.com/berkedilekoglu/machine-learning/tree/main/notes/Training_Models/Polynomial_Regression)'dan da hatırlayacağınız gibi degree'yi düşürür ve complexity'miz azalır.
4) Complexity'nin azalması overfitting'i engeller.

# Regression
Bir önceki konuda [Gradient Descent'i](https://github.com/berkedilekoglu/machine-learning/tree/main/notes/Training_Models/Batch_Stochastic_Mini_Batch_GD#farkl%C4%B1-gradient-descent-y%C3%B6ntemlerinin-kar%C5%9F%C4%B1la%C5%9Ft%C4%B1r%C4%B1lmas%C4%B1) işlerken Linear Regression'ı gördük ve eğitim sırasında weight ve bias vektörlerinin nasıl öğrenildiğini kodladık. Bu nedenle buradan sonrasında ```scikit-learn``` paketiyle devam edeceğiz. Scikit-learn bize birçok ml modelinin ve ml alanındaki kullanılan yöntemlerin hazır halini sunan bir kütüphanedir.
``` bash
pip install scikit-learn
```
komutu ile indirebilirsiniz. 

Eğitim ve tahminleme yapmak da oldukça kolaydır. 
``` python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
y_preds = model.predict(X)
```
Modelimizi tanımladıktan sonra data örneklerimizi (X) ve labellarımızı (y) .fit() fonksiyonuna veriyoruz ve model eğitiliyor. Eğittiğimiz modeli kullanarak artık .predict() fonksiyonunu kullanarak istediğimiz data üzerinde tahminleme yapabiliyoruz. 

>> Burada önemli olan yer eğitim yaptığınız ve tahminleme yaptığınız data vektörlerinin boyutlarının aynı olması.

## Polynomial Regression ##

Peki problemimiz polynomial bir fonksiyona göre tanımlanmışsa napacağız ? Örneğin şöyle bir fonksiyonu tahminlemeye çalışıyoruz.

$$y = ax^2 + bx + c$$

Gelin bu fonksiyona göre data oluşturduğumuzda bunun neye benzediğine bakalım.

![curve_data](https://github.com/berkedilekoglu/machine-learning/assets/19657350/a992b31f-6044-45de-a1ae-7d417acdb5a4)


Gördüğünüz gibi X ve y noktaları parabolik bir şekilde çiziliyor. Ancak bu fonksiyonu lineer bir doğru ile tahmin etmeye çalışmak imkansız. Lineer bir doğruyu bu parabole yerleştirmeye çalıştığınızı düşünün, hata payınız çok olacaktır. 

![linear_error](https://github.com/berkedilekoglu/machine-learning/assets/19657350/a35b4557-55b1-4a15-9549-eaa2c2abe178)


Yukarıdaki figürde görüldüğü gibi lineer bir doğru üzerinde fonksiyonumuzu tahmin etmeye çalıştığımızda ve MSE ölçtüğümüzde hata payımız yaklaşık 1.847 çıktı. 

Peki nasıl daha iyi bir tahminleme yapabiliriz ? Şuan elimizde olan datalarımızda 1 boyutlu featurelarımız var. Yani her bir örneğimizin 1 adet feature'ı bulunmakta. Peki bu feature vektörlerimizi 2 boyutlu yapsak ve her bir örnek için ilk feature'ın karesini 2. feature olarak kullansak. Böylece elimizdeki feature sayısını arttırmış ve modelimizin datamızla alakalı daha fazla özellik görerek tahminleme yapmasını sağlamış oluruz. Ayrıca şuanda tahminlemeye çalıştığımız fonksiyonun X ve X'in karesi ile alakalı olduğunu da biliyoruz. Bu sebeple modele her datanın kendisini ve karesini gösterirsek fonksiyon tahminlememiz iyileşecektir diye düşünebiliriz.

Sklearn kütüphanesinin PolynomialFeatures fonksiyonunu kullanarak elimizdeki datalara feature ekleyebiliriz. Bu fonksiyonun çalışma prensibini sayıların üstlerini alma şeklinde düşünebilirsiniz. Aşağıdaki resimde detaylı anlatımını göstericem.

![Note 18 Aug 2024](https://github.com/berkedilekoglu/machine-learning/assets/19657350/00bb8caa-8686-44a6-9f72-d26e6aea058f)


Resimde de görebileceğiniz şekilde PolynomialFeatures fonksiyonunu kullanarak elimizdeki featureları polynomial olarak genişletebilir ve kendimize yeni featurelar bulabiliriz. 

``` python
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
```

Dediğim gibi sklearn ile bunu yapmak çok kolay. Şimdi yeni featurelarımızla aynı fonksiyonu çözmeye çalışalım. 

<img width="854" alt="Screenshot 2023-09-14 at 14 00 07" src="https://github.com/berkedilekoglu/machine-learning/assets/19657350/c2af1986-027f-4b4d-91d0-16cd7e165af3">


Figürlerde de görebileceğiniz gibi feature dimensionlarımızı arttırdıkça ölçtüğümüz MSE değerlerimiz düşüyor. Ayrıca figürlerdeki mavi noktalar tahmin edilecek datalarımızı ve kırmızı çizgi tahminlerimizi gösterdiği için direkt figüre bakarak 25-dim boyutunda feature kullandığımızda daha iyi tahminler yaptığımızı görebilirsiniz.

Peki gerçekten durum bu mu ? Yani dimensionımızı ne kadar arttırırsak o kadar iyi tahminleme mi yapmış oluruz ? İşte bu noktada karşımıza __Bias / Variance__ tradeoff dediğimiz konu çıkıyor. Ya da __overfitting__ ve __underfitting__ terimleri. Bunların ne olduğunu [şuradan](https://github.com/berkedilekoglu/machine-learning/blob/main/notes/Training_Models/Bias_Variance_Overfitting_Underfitting/Readme.md#bias-variance-tradeoff) detaylıca okuyabilirsiniz.

## Performance on Val Data ##

<img width="732" alt="Screenshot 2023-09-19 at 17 53 29" src="https://github.com/berkedilekoglu/machine-learning/assets/19657350/50beb56c-226f-47bc-88d2-ebd8667e7f18">


Dimensionımızı arttırdığımızda işler iyiye mi gidiyor kötüye mi gidiyor anlamamızın yolu modelimizin performansını validation datamız üzerinde test etmektir. Unutmamalıyız ki modelimiz ve kullanacağımız featurelarımızla alakalı kararlar verirken training datamız üzerindeki hatamıza değil validation set üzerindeki datamıza bakmamız gerekir. Çünkü modelimizin daha sonra hiç görmeyeceği bir data üzerindeki performansı önemlidir. Aslında bunu test etmenin en güzel yolu cross validation dediğimiz yöntemdir ancak bunu daha sonra anlatacağım. Şimdilik basitçe ```sklearn train_test_split``` fonksiyonunu kullanarak train ve test datamızı oluşturalım.

``` python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)
```

Figüre baktığımızda training datamız üzerinde oluşan hatalar 2dim için 0.088 25dim için 0.057. Eğer bu değerlere bakıp hareket edersek 25dim'lik featureları seçip yolumuza devam edebiliriz. Ancak validation data üzerindeki hatalarıma baktığımızda tam tersi bir durum söz konusu. Peki nasıl seçim yapacağız. Eğer sabit bir validation datası kullanıyorsanız dimension arttıkça hatanızın nasıl değiştiğine bakabilirsiniz. Bu yöntem cross validation yöntemi varken tercih edilmemelidir ama mantık her iki yöntemde de validation error'e göre karar vermek olduğundan gelin artan dimension'lara göre hatamız nasıl değişiyor bakalım.

![error_with_dimension](https://github.com/berkedilekoglu/machine-learning/assets/19657350/fe0ad84e-ecf6-4b7f-905b-990f48037d7d)


gördüğünüz gibi aslında dimension 7'yi geçtiğinde artık overfitting gözlemlemeye başlıyoruz. Bu sebeple doğru dimension olarak 7'yi seçebiliriz. Overfitting sorununu nasıl çözeceğinize [şuradan](https://github.com/berkedilekoglu/machine-learning/blob/main/notes/Training_Models/Bias_Variance_Overfitting_Underfitting/Readme.md#overfitting-underfitting) bakabilirsiniz. 

## Curse of Dimensionality ##

Bu konuda işlediğimiz problemimiz curse-of-dimensinality dediğimiz problemdir. Bir datada dimension arttıkça onu anlayabilmek için gerekli data sayısı üstel olarak artar. Bu sebeple datayı sabit bırakırken fazla dimension iyi değildir ve gördüğümüz gibi data sayımız artmadığı için overfittinge yol açar. Çünkü feature sayımızın artması yanında şunları da getirir:

1) Featurelarımız arttıkça datamızı çok boyutlu uzaya taşımız oluruz. Çok boyutlu uzayda yapılan matematiksel işlemler uzun sürer çünkü vektörlerimiz boyutları artmış olur.

2) Çok boyutlu uzaydaki dataya overfit etmek örneğimizde de gördüğümüz gibi kolaydır. Çünkü demin de bahsettiğim gibi boyut arttıkça datanın anlamlanabilmesi için örnek sayısının çok fazla artması gerekir. Buna terimsel olarak __Data Sparsity__ denir.

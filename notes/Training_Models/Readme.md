# Training Models

Bu bölümde machine learning modellerinin nasıl eğitildiklerini göreceğiz. Aslında bu kısımı bilmeden de framework'leri kullanarak modellerinizi eğitip optimize edebilirsiniz. Ancak neyin nasıl çalıştığını bildiğinizde modelleri geliştirmek daha kolay oluyor. Çünkü bu alanda çalışacaksanız şunu anlıyorsunuz, denenecek çok fazla model ve yapılacak birçok deney var. Bunların hepsini yapmak büyük bir zaman kaybı olacaktır. Eğer arkadaki çalışma prensiplerini anlarsanız probleminize ve veri kümenizin özelliklerine göre hangi deneyleri yapacağınızı belirleyebilirsiniz. Bu da size zaman kazandırıp projelerinizi daha iyi yönetme kabiliyeti sağlayacaktır.

Bu bölümde basit çalışma prensibine sahip machine learning modellerinin arkasındaki sistemle başlayıp şuanki kullanılan matematiksel hesaplama yöntemlerine doğru gideceğiz. İlk olarak __Gradient Descent (GD)__'ile nasıl hesaplamalar yapıldığını öğreneceğiz. Ardından __Batch Gradient Descent GD__, __Mini-batch GD__, ve __Stochastic GD__ konularına bakacağız. Daha sonrasında birkaç __Regularization__ yöntemini öğrenip nasıl overfitting'i engelliyorlar onu göreceğiz. Son olarak __Linear Regression__, __Polynomial Regression__, __Logistic Regression__ ve __Softmax Regression__ modellerine göz atacağız.

## Gradient Descent

İlk olarak Gradient Descent anlatmak istememin nedeni daha sonrasında modelleri gördüğümüzde nasıl eğitildikleri hakkında aklınızın karışmasını istememem. Gradient Descent'in ne işe yaradığını anlayabilmemiz için önce şunu ifade etmemiz lazım, makine öğrenmesinde amacımız bir veri seti kullanarak fonksiyon tahminlemesi yapabilmektir. Örnek olarak linear regression modelini düşünelim. Linear regression probleminde her bir tahminleme linear bir fonksiyon yardımıyla yapılır. Yani problemi şöyle özetleyebiliriz:

$$\hat{y} = wx + b$$

Burada $$\hat{y}$$ yaptığımız tahminler. Gerçek değerlerimiz yani eğitim sırasında kullanıp ulaşmak istediğimiz değerlerimiz ise $$y$$. Böylece hata payımızı $$\hat{y}-y$$ şeklinde ölçebiliriz. Hata ölçmenin birçok farklı yolu olsa da temel prensip hep aynıdır: tahminlerimiz olması gereken değerlere ne kadar uzak. Hata ölçtüğümüz fonksiyonumuza __Loss__ fonksiyonu denir ve her zaman amacımız Loss'umuzu azaltmaktır. 

Şimdi burada iki önemli konu var:

1) Loss'u nasıl azaltabiliriz ?

Loss fonksiyonunu matematiksel olarak yazıp yorumlayalım. Mesela popüler bir Loss fonksiyonu seçelim. __MSE__ yaygın kullanılan bir loss fonksiyonudur ve şu şekilde hesaplanır:

$$Loss = {1 \over 2n} \sum_{i=1}^n(\hat{y}_i - y_i)$$

Burada $$n$$ elimizdeki sample/örnek sayısıdır. $$y$$ gerçek değerler/labellarımızdır. Yani modelimiz aslında sadece $$\hat{y}$$ hesaplamasını iyileştirmeye çalışıyor. Onun da Linear Regression modeli için şöyle hesaplandığını biliyoruz: $$\hat{y} = wx + b$$. Eğer Loss fonksiyonu içinde yerine yazarsak:

$$Loss = {1 \over 2n} \sum_{i=1}^n(wx_i + b - y_i)$$

Böylece değerini sürekli azaltmak istediğimiz fonksiyonumuzu yazmış olduk. Bu fonksiyonda $$x$$ değerleri bizim örneklerimizin featureları. Yani fonksiyon içinde aslında değiştirebileceğimiz (Modelin öğrenmesini istediğimiz) sadece 2 değerimiz var. Birisi w (Weight), diğeri de b (bias). İlk hesaplama için bu değerler random bir şekilde belirlenir ve hesaplama yapılır. Random bir şekilde belirlendikten sonra bir Loss hesaplayabilecek konuma geliriz. İşte buradan sonrası çok önemli. Peki bu Weight ve Bias'ı ikinci hesaplama için nasıl ayarlamalıyız ki Loss azalsın ?

2) Bir fonksiyonun türevi ne anlama gelir ?

![Loss_function](https://github.com/berkedilekoglu/machine-learning/assets/19657350/59b57f21-5e35-48ee-a660-c6bbfc469e9a)
Figür 1: Loss Fonksiyonu

Yukarıdaki resimde gördüğümüz gibi Loss fonksiyonunu çizersek bu resim üzerinde fonksiyonun en küçük değerine ulaşmayı başarabiliriz. Sarı nokta ile gösterilen yer ulaşmak istediğimiz yer. Uygun $$W$$ ve $$b$$ değerleriyle sarı noktaya ulaşabiliriz. Peki matematiksel olarak bunu nasıl yapacağız.

![Page1 2 copy 2](https://github.com/berkedilekoglu/machine-learning/assets/19657350/192b8770-0fd3-47a2-b221-011fb51b2f8b)
Figür 2: Random başlatma sonrası loss değeri

Daha önce söylediğim gibi ilk loss değerimizi bulmak için $$W$$ ve $$b$$ değerlerini random olarak belirleyip bir loss hesaplıyoruz. Bu loss değeri yukarıdaki resimde mavi nokta ile belirtilen değerdir. Gördüğünüz gibi amacımız sarı noktaya ulaşabilmek.

Grafiğin yatay eksenini $$W$$ olarak düşünelim. Zaten elimizdeki fonksiyona göre değişkenlerimiz $$W$$ ve $$b$$ olduğu için ikisinden biri olabilir. Şimdi mevcut durumumuza göre $$W$$'yü arttırmalıyız ki Loss'umuz azalsın. Tabi eğer $$W$$ değerini çok arttırıp sarı noktanın sağına geçersek de bir sonraki adımda azaltmalıyız. Burada önemli olan $$W$$'yü arttırmalı mıyız azaltmalı mıyız onu bulabilmek. 

![Page1 2 copy](https://github.com/berkedilekoglu/machine-learning/assets/19657350/a4c734fa-2f7a-4779-99d2-56decd68e834)
Figür 3: Fonksiyonun belirli bir noktaya göre türevi

Gradient bir fonksiyonun, her bir parametresinin türevinden oluşan bir vektördür. Yani gradient'i bir fonksiyonun türevi olarak düşünürsek ne anlam ifade eder ? Matematiksel olarak bir fonksiyonun belli bir noktaya göre türevini aldığımızda, fonksiyonun o noktadaki yönü ve eğimini bulabiliriz. Bu da aslında bize hangi parametreye göre fonksiyonun türevini aldıysak, fonksiyonun o parametreye göre değişim hızını gösterir. 

>> Biraz karışmış gibi gelebilir o yüzden hemen tanıdık bir örneği düşünelim. Fizikten hatırlayacağınız gibi Yolun, Zamana göre türevi bize hızı verir. Bu şu demektir: Yol fonksiyonunun zamana göre değişim hızı ya da daha iyi ifade edersek Yolun birim zamandaki değişimi. Aynı şekilde eğer Hız fonksiyonunun zamana göre türevini alırsanız bu da size Hızın birim zamandaki değişimini yani ivmeyi verir. 

Yukarıdaki resimde görebileceğiniz gibi Loss fonksiyonumuzun mavi noktaya göre türevini aldığımızda fonksiyonumuzun $$W$$ ve $$b$$ ye göre değişim yönünü bulabiliriz. Eğer örnekteki gibi bu noktanın türevi negatif gelirse bu fonksiyonun artan $$W$$ veya $$b$$ değerine göre azaldığına işarettir. Eğer türevimiz pozitif gelirse bu da fonksiyonun bu değerlere göre artacağını gösterir.

O halde yeni $$W$$ ve $$b$$ değerline nasıl karar vericez ?

![Page2 3](https://github.com/berkedilekoglu/machine-learning/assets/19657350/e3faee3d-cc80-438b-b059-12bc3b53fdbc)
Figür 4: Weight veya bias'ın yeni değerlerinin belirlenmesi

Bu kısıma alanda optimizasyon deniyor ve birçok optimizasyon yöntemi bulunuyor. Bunlara daha sonra değineceğiz ama basitçe şu şekilde bir yöntem izleyebiliriz. Bu basit yöntem üzerinde geliştirmeler yapılarak diğer optimizasyon yöntemleri ortaya çıkmıştır.
Resimde göreceğiniz gibi:

$$W_{new} = W_{old} - \alpha * {dLoss \over dW}$$

$$b_{new} = b_{old} - \alpha * {dLoss \over db}$$

Formülleri ile yeni değişken değerlerimiz bulunur. Burada temel mantık eski değeri büyütmek veya küçültmektir. O sebeple arada $$-$$ işareti var. Çünkü türev negatifse değişkenlerin yeni değerleri artmalı türev npozitifse azaltılmalıdır. $$\alpha$$ ise bu artışın veya azalışın ne kadar büyüklükte olacağını kontrol eden __Learning Rate__ parametresidir. Learning Rate bir hyperparametredir ve tune edilerek en doğru değeri bulunur. Yukarıdaki resimde daha net bir şekilde bunu görebilirsiniz.


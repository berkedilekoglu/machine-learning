## Gradient Descent

İlk olarak Gradient Descent anlatmak istememin nedeni daha sonrasında modelleri gördüğümüzde nasıl eğitildikleri hakkında aklınızın karışmasını istememem. Gradient Descent'in ne işe yaradığını anlayabilmemiz için önce şunu ifade etmemiz lazım, makine öğrenmesinde amacımız bir veri seti kullanarak fonksiyon tahminlemesi yapabilmektir. Örnek olarak linear regression modelini düşünelim. Linear regression probleminde her bir tahminleme linear bir fonksiyon yardımıyla yapılır. Yani problemi şöyle özetleyebiliriz:

$$\hat{y} = wx + b$$

Burada $$\hat{y}$$ yaptığımız tahminler. Gerçek değerlerimiz yani eğitim sırasında kullanıp ulaşmak istediğimiz değerlerimiz ise __y__. Böylece hata payımızı $$\hat{y}-y$$ şeklinde ölçebiliriz. Hata ölçmenin birçok farklı yolu olsa da temel prensip hep aynıdır: tahminlerimiz olması gereken değerlere ne kadar uzak. Hata ölçtüğümüz fonksiyonumuza __Loss__ fonksiyonu denir ve her zaman amacımız Loss'umuzu azaltmaktır. 

Şimdi burada iki önemli konu var:

1) Loss'u nasıl azaltabiliriz ?

Loss fonksiyonunu matematiksel olarak yazıp yorumlayalım. Mesela popüler bir Loss fonksiyonu seçelim. __MSE__ yaygın kullanılan bir loss fonksiyonudur ve şu şekilde hesaplanır:

$$Loss = {1 \over 2n} \sum_{i=1}^n(\hat{y}_i - y_i)^2$$

Burada __n__ elimizdeki sample/örnek sayısıdır. __y__ gerçek değerler/labellarımızdır. Yani modelimiz aslında sadece $$\hat{y}$$ hesaplamasını iyileştirmeye çalışıyor. Onun da Linear Regression modeli için şöyle hesaplandığını biliyoruz: $$\hat{y} = wx + b$$ Eğer Loss fonksiyonu içinde yerine yazarsak:

$$Loss = {1 \over 2n} \sum_{i=1}^n(wx_i + b - y_i)^2$$

Böylece değerini sürekli azaltmak istediğimiz fonksiyonumuzu yazmış olduk. Bu fonksiyonda __x__ değerleri bizim örneklerimizin featureları. Yani fonksiyon içinde aslında değiştirebileceğimiz (Modelin öğrenmesini istediğimiz) sadece 2 değerimiz var. Birisi __W__ (Weight), diğeri de __b__ (bias). İlk hesaplama için bu değerler random bir şekilde belirlenir ve hesaplama yapılır. Random bir şekilde belirlendikten sonra bir Loss hesaplayabilecek konuma geliriz. İşte buradan sonrası çok önemli. Peki bu Weight ve Bias'ı ikinci hesaplama için nasıl ayarlamalıyız ki Loss azalsın ?

2) Bir fonksiyonun türevi ne anlama gelir ?

![Loss_function](https://github.com/berkedilekoglu/machine-learning/assets/19657350/59b57f21-5e35-48ee-a660-c6bbfc469e9a)
Figür 1: Loss Fonksiyonu

Yukarıdaki resimde gördüğümüz gibi Loss fonksiyonunu çizersek bu resim üzerinde fonksiyonun en küçük değerine ulaşmayı başarabiliriz. Sarı nokta ile gösterilen yer ulaşmak istediğimiz yer. Uygun __W__ ve __b__ değerleriyle sarı noktaya ulaşabiliriz. Peki matematiksel olarak bunu nasıl yapacağız.

![Page1 2 copy 2](https://github.com/berkedilekoglu/machine-learning/assets/19657350/192b8770-0fd3-47a2-b221-011fb51b2f8b)
Figür 2: Random başlatma sonrası loss değeri

Daha önce söylediğim gibi ilk loss değerimizi bulmak için __W__ ve __b__ değerlerini random olarak belirleyip bir loss hesaplıyoruz. Bu loss değeri yukarıdaki resimde mavi nokta ile belirtilen değerdir. Gördüğünüz gibi amacımız sarı noktaya ulaşabilmek.

Grafiğin yatay eksenini __W__ olarak düşünelim. Zaten elimizdeki fonksiyona göre değişkenlerimiz __W__ ve __b__ olduğu için ikisinden biri olabilir. Şimdi mevcut durumumuza göre __W__'yü arttırmalıyız ki Loss'umuz azalsın. Tabi eğer __W__ değerini çok arttırıp sarı noktanın sağına geçersek de bir sonraki adımda azaltmalıyız. Burada önemli olan __W__'yü arttırmalı mıyız azaltmalı mıyız onu bulabilmek. 

![Page1 2 copy](https://github.com/berkedilekoglu/machine-learning/assets/19657350/a4c734fa-2f7a-4779-99d2-56decd68e834)
Figür 3: Fonksiyonun belirli bir noktaya göre türevi

Gradient bir fonksiyonun, her bir parametresinin türevinden oluşan bir vektördür. Yani gradient'i bir fonksiyonun türevi olarak düşünürsek ne anlam ifade eder ? Matematiksel olarak bir fonksiyonun belli bir noktaya göre türevini aldığımızda, fonksiyonun o noktadaki yönü ve eğimini bulabiliriz. Bu da aslında bize hangi parametreye göre fonksiyonun türevini aldıysak, fonksiyonun o parametreye göre değişim hızını gösterir. 

>> Biraz karışmış gibi gelebilir o yüzden hemen tanıdık bir örneği düşünelim. Fizikten hatırlayacağınız gibi Yolun, Zamana göre türevi bize hızı verir. Bu şu demektir: Yol fonksiyonunun zamana göre değişim hızı ya da daha iyi ifade edersek Yolun birim zamandaki değişimi. Aynı şekilde eğer Hız fonksiyonunun zamana göre türevini alırsanız bu da size Hızın birim zamandaki değişimini yani ivmeyi verir. 

Yukarıdaki resimde görebileceğiniz gibi Loss fonksiyonumuzun mavi noktaya göre türevini aldığımızda fonksiyonumuzun __W__ ve __b__ ye göre değişim yönünü bulabiliriz. Eğer örnekteki gibi bu noktanın türevi negatif gelirse bu fonksiyonun artan __W__ veya __b__ değerine göre azaldığına işarettir. Eğer türevimiz pozitif gelirse bu da fonksiyonun bu değerlere göre artacağını gösterir.

O halde yeni __W__ ve __b__ değerline nasıl karar vericez ?

![Page2 3](https://github.com/berkedilekoglu/machine-learning/assets/19657350/e3faee3d-cc80-438b-b059-12bc3b53fdbc)
Figür 4: Weight veya bias'ın yeni değerlerinin belirlenmesi

Bu kısıma alanda optimizasyon deniyor ve birçok optimizasyon yöntemi bulunuyor. Bunlara daha sonra değineceğiz ama basitçe şu şekilde bir yöntem izleyebiliriz. Bu basit yöntem üzerinde geliştirmeler yapılarak diğer optimizasyon yöntemleri ortaya çıkmıştır.
Figür 4'de göreceğiniz gibi:

$$W_{new} = W_{old} - \alpha * {dLoss \over dW}$$

$$b_{new} = b_{old} - \alpha * {dLoss \over db}$$

Formülleri ile yeni değişken değerlerimiz bulunur. Burada temel mantık eski değeri büyütmek veya küçültmektir. O sebeple arada __-__ işareti var. Çünkü türev negatifse değişkenlerin yeni değerleri artmalı türev pozitifse azaltılmalıdır. $$\alpha$$ ise bu artışın veya azalışın ne kadar büyüklükte olacağını kontrol eden __Learning Rate__ parametresidir. Learning Rate bir hyperparametredir ve tune edilerek en doğru değeri bulunur. Yukarıdaki resimde daha net bir şekilde bunu görebilirsiniz.

![Note 18 Aug 2023](https://github.com/berkedilekoglu/machine-learning/assets/19657350/947f2105-61a7-42bc-835e-769c70dce561)
Figür 5: Matematiksel olarak Loss'un Weight ve bias'a göre türevinin alınması.

Figür 5'de ayrıntılı olarak MSE ve linear bir fonksiyon (Linear Regression) için türev kullanılarak nasıl optimizasyon formülüne sokulur görebiliriz.görebilirsiniz.

### Farklı Loss Fonksiyonları (Zorluklar)

Malesef loss fonksiyonları hiçbir zaman Figür 1'deki kadar kolay olmuyorlar. Bu da bize minimum noktalarını bulmakta zorluk çıkartıyor.

![loss](https://github.com/berkedilekoglu/machine-learning/assets/19657350/e1ebb2a7-f80c-438d-a396-5f753a2821ad)
Figür 6: Loss fonksiyon grafiği

Figür 6'e bakacak olursanız gerçek hayattaki problemlerde karşılaşabileceğiniz loss fonksiyon grafiklerinin benzerini görebilirsiniz. Fonksiyon grafiğine baktığımızda aslında en iyi ulaşılacak noktanın __Global Minimum__ noktasına ulaşmak olduğunu görebilirsiniz. Ancak random şekilde başlatılmış bir model bizi mavi noktayla işaretli yere ulaştırdığında __Local Minimum__ noktasını atlamamız çok zor olabilir. Veya Figür 6'de görebileceğiniz __Plato__ denilen yerde sıkışıp kalabiliriz. İşte tüm bunlar ml alanının aktif araştırma konuları. Uygun random değerleri bulup bizi en uygun yerden başlatma ve en uygun optimizasyon yöntemini kullanıp Local Minimum veya Plato'ları atlatma. Bu konuları ilerde daha detaylı anlatacağım.

# Farklı Gradient Descent Yöntemlerinin Karşılaştırılması #

Farklı gradient descent algoritmalarını görmeden önce iki tane temel kavramı anlatmak istiyorum. Bunları anlayabilirsek farklı algoritmaları anlamamız daha kolay olacaktır. Çünkü gradient descent iteratif bir optimizasyon yöntemidir. Yani modelimizi step step iyileştirmeye çalıştığımız ve her bir stepde loss'umuzu azaltmayı hedeflediğimiz bir yöntemdir. Burada öne çıkan iki kavram __Batch Size__ ve __Epoch__ kavramlarıdır.

## Epoch ##

Epoch sayısı, optimizasyonumuzun tüm training datamızı elden geçirmesini ifade eden bir __hyperparameter__'dir. __Hyperparameter__ tanımını ve nasıl bulunduklarını daha sonra kapsamlı anlatacağım ama kısaca belirli bir değeri olmayan, datadan dataya ve modelden modele değeri değişen değerlerdir. __Ancak model tarafından öğrenilmezler__. Bu parametreleri belirli şekillerde biz bulup kullanırız.

Kısaca 1 Epoch, modelin tüm training datasını görmesine denir. Eğer epoch sayınızı 10 yaparsanız bu tüm training datasının 10 kez tekrar tekrar model tarafından görülüp bir şeyler öğrenilmesine çalışılması anlamına gelir.

## Batch Size ##

Batch Size aynı Epoch gibi bir hyperparameter'dir. Epoch'da modelin tüm training datasını gördüğünü söylemiştik. Burada aklınıza şu soru gelmiş olabilir: Peki tüm training datasını aynı anda mı modele vermem gerekiyor ? Tek tek verebilir miyim veya küçük pakatler halinde verebilir miyim ? İşte __Batch Size__ bunu kontrol eden hyperparameter. Batch Size'ı 1'den başlayarak elinizdeki training örnek sayısı miktarına kadar verebilirsiniz. Biraz aklınız karışmış olabilir. Farklı gradient descent algoritmalarını gördüğünüzde bu konu tamamen oturucak :)

## Farklı Gradient Descent Yöntemleri ##
Gradient Descent, Epoch ve Batch Size'ın ne olduklarını öğrendik. Şimdi gelin 3 temel gradient descent yöntemini karşılaştıralım. Bunlar sırasıyla:

- Batch Gradient Descent
- Stochastic Gradient Descent
- Mini-Batch Gradient Descent

Yöntemleridir. Eğer gradient descent'in mantığını anladıysanız 3 yöntem de aslında aynı hesaplamaları kullanır. Sadece işleme alacağımız örnek sayısında (Yani Batch Size'ımızda) ve bu örneklerin seçiminde değişimler yapılır. Şimdi bu üç Gradient örneği için de __MSE__ loss fonksiyonumuzu kullanalım. Hatırlayacağınız gibi loss fonksiyonumuz:

$$Loss = {1 \over 2n} \sum_{i=1}^n(wx_i + b - y_i)^2$$

Eğer W'ye göre türevini alırsak:

$${dLoss \over dW} = 2*{1 \over 2n} \sum_{i=1}^n(wx_i + b - y_i)x_i$$

Bias'a göre türevini alırsak:

$${dLoss \over db} = 2*{1 \over 2n} \sum_{i=1}^n(wx_i + b - y_i)$$

Şimdi burada önemli olan bu işlemleri vektör formatında yazmak. 

$${dLoss \over dW} = {1 \over N} X^T(XW + b - Y)$$
$${dLoss \over db} = {1 \over N} (XW + b - Y)$$

Bu işlemlerdeki her bir vektör çarpımının dot product olmasına dikkat edin. X feature vektörümüz, Y doğru değerlerimiz, b bias vektörü ve W weight vektörümüzdür. N değeri ise işleme alınacak örnek sayısını ifade eder. 

## Batch Gradient Descent ##

İlk olarak Batch Gradient Descent'i anlayalım. Çünkü diğer yöntemlerde de mantık hep aynı olacak. Zaten Gradient Descent'in ne anlam ifade ettiğini görmüştük. 1 Epoch'un tüm training data üzerinden 1 kez geçilmek olduğunu öğrendik ve Batch Size'ın datamızı kaçlık paketler haline bölüp optimizasyona vereceğimiz olduğunu gördük. 

Eğer Batch Size'ınız training datanızdaki sample sayınıza eşitse, yani tüm datayı birden optimizasyona aynı anda sokacaksanız buna __Batch Gradient Descent__ yöntemi denir. 

>> Not: Unutmayalım gradient descent iterative bir algoritmadır ve verdiğimiz örnekleri kullanarak yeni __Weight__ ve __Bias__ ları öğrenmemiz için türevleri bulmamızı sağlar. 

Örneğin elimizde 100 adet sample varsa ve batch size'ımız 100 ise tüm örnekleri aynı anda yukarıdaki türevlerde kullanmış oluruz. Böylece 1 epoch tamamlanmış olur.

Tüm datayı kullanarak türevlerimizi hesapladığımız için Batch Gradient Descent daha istikrarlı hesaplamalar sağlar. Ancak günümüzde veri setleri ve model boyutları çok arttığı memory'nize datanız sığmayabilir veya sığsa bile 1 epoch içerisindeki matris çarpımlarının hesaplanması çok uzun zaman alabilir. 

## Stochastic Gradient Descent ##

Tüm datayı kullanmak ne kadar güzel sonuçlar verse de günümüzde çok da mümkün değildir. Peki veri setimizdeki her bir örnek için hesaplama yapsak. Yani her bir veriyi alıp tek tek türevlerimizi hesaplayıp yeni __Weight__ ve __Bias__ larımızı öğrensek.

Aslında Batch Size'ımızı 1 yapmış oluyoruz. İşte bunu yapmak Stochastic Gradient Descent hesaplaması demektir. Memory'nizi minimumda kullanır. Örneğin elinizde 100 örnek varsa 1 epochta her bir örnek için ayrı ayrı __Weight__ ve __Bias__ hesaplanır. Yani 1 epoch'da __Weight__ ve __Bias__ vektörleriniz 100 kez güncellenmiş olur. Burada güncellemeler birer örnek için yapıldığından dolayı nitekim hızlı olacaktır.

>> Stochastic Gradient Descent için örneklerin seçilme şekli çeşitlilik gösterir. Ancak en çok kullanılan yöntem her bir epoch başlangıcında training datasını random bir şekilde karıştırıp Epoch içerisinde sırayla baştan sona doğru örnekleri kullanarak parametreleri güncellemektir.

Stochastic Gradient Descent avantajlı gözükse de her bir data için parametreleri güncellemek sıkıntılar doğurabilir. Datalarımızda genelde noise bulunur ve bazı datalar ekstra noisy olabilir. SGD ile parametrelerimiz bu datalardan çok etkilenebilir.

## Mini-Batch Gradient Descent ##

Aslında tam olarak Batch Gradient Descent ve Stochastic Gradient Descent'in ortasını bulan bir yöntemdir. 1'den büyük ve data miktarınızdan az olarak seçtiğiniz her Batch Size ile Gradient hesaplama yöntemi Mini-Batch Gradient Descent'dir. 

Yukarıda öğrendiğimiz gibi batch size'ı arttırdığımızda daha çok data görerek parametre öğrenmek, hız ve memory kullanımı arasında trade off vardır. Bu sebeple Batch Size probleme göre Tune edilip bulunmalıdır. 

>> Ayrıca bazı kaynaklarda SGD yönteminin hızlı olduğu yazsa da günümüzdeki GPU'ların paralel computing özelliği batch size kullanmayı çok avantajlı kılar. Mini-Batch Gradient kullanarak batch size'ınızı arttırdığınızda işlemlerinizin hızlandığını gözlemleyebilirsiniz. Ancak bu parametrenin mutlaka Tune edilmesi gerektiğini unutmayın. 

## Code Example ##

Şimdi sıfırdan kendimiz bu algoritmaları yazarak bir gözlemleme yapalım istiyorum. Şimdi öncelikle datamızı oluşturalım:

```python
X = 2 * np.random.rand(100, 2)
y = 4 + 3 * X + np.random.randn(100, 1)
```

Örneğimizde yine MSE üzerinden ve Linear Regression üzerinden gideceğiz. Size bu problem için __W__ ve __b__ vektörlerinin boyutlarını nasıl seçeceğinizi şu resimlerle gösterebilirim:

Resimler

```python
# Rastgele başlangıç değerleri atanmış W ve b vektörlerimizi oluşturalım
np.random.seed(42)
# y = xw+b -> x'in boyutu mxn ise ve y'nin boyutu mxk ise w'nün boyutu nxk olmalıdır
W_ = np.random.randn(X.shape[1],y.shape[1])
# Bias vektörümüzü oluşturalım -> bias vektörü bir column vektörüdür yani ilk boyutu 1'dir. ikinci boyutu ise w'den gelir ve eğer w nxk ise b 1xk'dır.
b_ = np.random.randn(1,W_.shape[1]) 
```
__W__ ve __b__ vektörlerimiz de hazır olduğuna göre gelin önce basit bir data shuffle için fonksiyon yazalım

```python
def shuffle_data(X,y):
    # Create an array of indices and shuffle them
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    # Shuffle X and y based on the shuffled indices
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    return X_shuffled, y_shuffled
```

Bu ```shuffle_data``` fonksiyonu bize mini_batch ve SGD yöntemlerinde yardımcı olacak. Şimdi ana fonksiyonumuzu yazalım. 

```python
def gradient_descent(X, y, learning_rate, epoch_num, batch_size):

    W = W_.copy()
    b = b_.copy()
    all_weights = []
  
    for iteration in range(epoch_num):
        
        X_shuffled, y_shuffled = shuffle_data(X,y) #Her epochta datamızı shuffle edelim
        # Verileri batch'lere bölmek için bir for döngüsü kullanın
        for i in range(0, len(X_shuffled), batch_size):

            x_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            all_weights.append(W)
            
            #Gradientleri formüllere göre bulduk
            weight_gradients = (1/batch_size) * x_batch.T.dot(x_batch.dot(W)+b - y_batch)
            bias_gradients = (1/batch_size) * (x_batch.dot(W)+b - y_batch)
            #Burada önemli olan nokta bias_gradient'in bir scalar olmasıdır. Çünkü Bias bir column vektör
            bias_gradients = np.sum(bias_gradients,axis=0,keepdims=True)
            
            # Aslında buradan sonrası optimizasyon algoritmasıdır. İleride göreceğiz
            W = W - learning_rate * weight_gradients
            b = b - learning_rate * bias_gradients

    return np.asarray(all_weights)
```


Her bir gradient descent algoritması için genel bir fonksiyon yazdık. Sadece batch size'ı değiştirerek istediğimize erişebiliriz. 

```python
learning_rate = 0.01
epoch_num = 1000

batch_size = len(y) #Batch Gradient Descent
weight_array_batch = gradient_descent(X, y, learning_rate, epoch_num, batch_size)
batch_size = 1 #Stochastic Gradient Descent
weight_array_stochastic = gradient_descent(X, y, learning_rate, epoch_num, batch_size)
batch_size = 4 #Mini-Batch Gradient Descent
weight_array_mini_batch = gradient_descent(X, y, learning_rate, epoch_num, batch_size)
```

Şimdi gelin __Weight__ parametrelerinin parametre uzayında nasıl güncellendiklerine bakalım. 

```python

w1_batch = weight_array_batch[:, 0, 0]
w2_batch = weight_array_batch[:, 1, 0]

w1_sgd = weight_array_stochastic[:, 0, 0]
w2_sgd = weight_array_stochastic[:, 1, 0]

w1_mini = weight_array_mini_batch[:, 0, 0]
w2_mini = weight_array_mini_batch[:, 1, 0]

plt.figure(figsize=(8, 6)) 

# Create the plot
plt.plot(w1_sgd, w2_sgd, label='SGD', marker='x', linestyle='-', markersize=1)
plt.plot(w1_mini, w2_mini, label='Mini-Batch GD', marker='s', linestyle='-', markersize=1)
plt.plot(w1_batch, w2_batch, label='Batch GD', marker='o', linestyle='-', markersize=1) 

# Add labels and a legend
plt.xlabel('w1')
plt.ylabel('w2')
plt.legend()

# Display the plot
plt.show()
```

Aşağıdaki figürden de görebileceğiniz gibi Batch GD yöntemi kendinden emin adımlarla optimal noktaya doğru ilerlemiş ve durmuş. SGD yönteminin çizdiği zikzakları görüyorsunuz. İşte bahsettiğim her bir örnekten spesifik olarak etkilenme olayı burada karşımıza çıkıyor. Ayrıca optimal parametreleri aramaya da devam ediyor ama bir noktada sabit kalmak yerine geniş bir alanda zikzaklar çizmiş. İkisinin tam ortasında mini-batch gd'in olduğunu görüyoruz. 

IMAGE

>> Notlar: Size şu yöntem daha iyidir diyemiyorum. Örneğin çok büyük Transformer modelleri train edecekseniz mecburen Batch Size'ınız 1 olabilir. Ayrıca Batch Size arttıkça accuracy'niz artar da diyemiyorum çünkü problemden probleme değişeceği için Tune edilmesi gereken bir parametre. Günümüzdeki optimizasyon algoritmaları figürde gördüğümüz noisy dolanmaları veya istediğimiz yere ulaşamama problemlerini çözebiliyorlar. 

>> Genelde Batch Size 2 ve üsleri olarak girilir (2,4,8...). Buna verilen genel cevap bilgisayar sistemleri 2'li sistem olduğundan ve CPU/GPU'ların memoryleri mimarilerinin 2 ve üslerine göre yapıldığındandır. Ancak bu gibi şeyleri hep forumlarda okudum. Eğer alanında uzman bu konularda çalışan bir elektronikçi arkadaşınız varsa sorabilirsiniz :)
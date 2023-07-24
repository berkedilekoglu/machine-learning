# Ensemble Learning

Elinizde bir matematik sorusu olduğunu düşünün. Bu soruyu matematik eğitimi almış bir kişiye sormaktansa matematik eğitimi almış 1000 kişiye sorup birbirleriyle tartışarak size cevap vermelerini isterseniz doğru cevabı bulma olasılığınız artacaktır. Buna benzer bir şekilde bir grup classifier veya regressor'ı kullanarak yapacağınız tahminler daha doğru sonuçlar verecektir. Günümüzde deep learning teknikleri çok gelişmiş olsa da Machine Learning yarışmalarında ensemble teknikleri popüler olarak kullanılıyor.

Örneğin elinizdeki training datasını random bir şekilde küçük parçalara böldüğünüzü düşünün. Her böldüğünüz random bölgede decision tree eğitelim. Böylece datanın farklı yerlerini farklı decision treelere öğretme şansımız olacak. Eğer bu decision treelerin tahminlerini birleştirirsek her bir decision tree'nin yapabileceğinden daha iyi bir tahminleme yapmış oluruz. Bu yöntem çok popüler olarak kullanılan Random Forest yöntemidir. Çok basit gözükse de oldukça iyi sonuçlar verebilir.

## Ensemble Teknikleri

Bu yazıda aşağıdaki ensemble tekniklerini inceliycez:

- Voting
- Bagging/Pasting (Random Forest, Extra Trees)
- Boosting (AdaBoost, Gradient Boosting (XGBoost))
- Stacking

## Voting

Voting yöntemleri yapılarından dolayı classification problemleri için kullanılan basit ensemble yöntemleridir.
Elimizde bir data üzerinde eğitilmiş ayrı modeller olduğunu düşünün. Mesela Logistic Regression, SVM, Random Forest ve birkaç tane daha modelden ayrı ayrı %80 civarı accuracy aldığımızı farz edelim. 

>> Burada dikkat etmemiz gereken şey eğitimler için aynı datayı kullandığımızdan dolayı farklı şeyler öğrenebilmek için farklı algoritmalar kullanmaya özen göstermektir. Eğer ensemble learning yöntemi olarak voting kullanacaksanız farklı algoritmalar kullanmanız başarı oranınızı arttıracaktır. Örneğin Random Forest ve Extra Tree algoritmaları benzer yapıda olduğundan dolayı aynı hataları yapma ihtimalleri yüksek olur. Aynı hataların yapıldığı predictionlardan voting yaparak daha iyi sonuçlar alma ihtimalimiz düşer.

### Hard Voting (Majority Voting)
(Buraya resim çizip koy)
Bu modellerden daha iyi bir accuracy almak için bir ensembling yapmanın en kolay yöntemi **_hard voting_** dediğimiz yöntemdir. Hard voting yöntemi tüm modellerin yaptığı tahminlere bakarak en çok tahmin edilen sınıfı seçer. 
Örneğin: 2 class tahminlemesi yapılan bir problemde bir test örneği üzerinde aşağıdaki tahminler yapılsın.

- SVM: (0.65,0.35) -> 1. sınıf tahmin edildi
- LR: (0.45, 0.55) -> 2. sınıf tahmin edildi
- Random Forest: (0.72, 0.28) -> 1. sınıf tahmin edildi

Bu tahminlere göre **_hard voting_** yöntemi kullanılırsa bu test örneği üzerinde 1. sınıf tahmini yapılacaktır. Örnekte gördüğümüz gibi eğer **_hard voting_** yöntemi kullanılacaksa sınıf tahminlemesi yapılır. Yani yazacağınız algoritma size bir sınıf döndürecektir. Eğer siz sınıf olasılıklarını tahmin etmek istiyorsanız o zaman **_soft voting_** yöntemini kullanmanız gerekir.

>> **_Hard voting_** kullanırken tahmin edilen sınıflar eşit sayıdaysa Örneğin SVM: 1. sınıf, LR: 2. sınıf, Random Forest: 1. sınıf, KNN: 2. sınıf tahmin etti. O zaman sınıflar arasında random bir sınıf seçilerek döndürülür.

### Soft Voting
(Buraya resim çizip koy)

**_Hard voting_** kısmında bahsettiğim gibi eğer tahminlerinizi olasılık olarak yapmak istiyorsanız **_Soft Voting_** yöntemini kullanmanız gerekiyor. Soft voting, her bir modelin tahminleri üzerinde ağırlıklar kullanır ve bu ağırlıklar, her modelin güvenilirliğini veya performansını yansıtır. Sınıf olasılıklarının toplamı üzerinde bir normalizasyon işlemi yapılır ve en yüksek toplam olasılığa sahip olan sınıf seçilerek tahmin edilir. Olasılığı yüksek olan modelin etkisi çok olacağı için kendinden emin model daha çok etkide bulunur.

Örneğin, üç farklı modelin olduğunu düşünelim ve her bir modelin sınıf olasılıkları şöyle olsun:

- SVM: 1. Sınıf olasılığı = 0.6, 2. Sınıf olasılığı = 0.4
- LR: 1. Sınıf olasılığı = 0.8, 2. Sınıf olasılığı = 0.2
- RF 3: 1. Sınıf olasılığı = 0.5, 2. Sınıf olasılığı = 0.5

Soft voting kullanılıyorsa, bu olasılıklar toplanır ve normalizasyon işlemi yapılır:

Sınıf 1 için olasılıklar toplamı = 0.6 + 0.8 + 0.5 = 1.9
Sınıf 2 için olasılıklar toplamı = 0.4 + 0.2 + 0.5 = 1.1

Normalizasyon işlemi yapılır:

Normalizasyonlu 1. sınıf olasılığı = 1.9 / (1.9 + 1.1) = 0.633
Normalizasyonlu 2. sınıf olasılığı = 1.1 / (1.9 + 1.1) = 0.367

Sonuç olarak, soft voting yöntemiyle tahmin edilen sınıf 1. sınıf olur.

Bu şekilde, sınıf olasılıklarının ağırlıklı bir şekilde hesaplanmasıyla soft voting, daha hassas ve esnek bir tahminleme yöntemi sağlar. Ancak, hard voting kadar basit ve hızlı olmayabilir.

>> *__Soft Voting__* kullanmanız için tüm classifierların olasılık tahminlemesi yapıyor olabilmeleri lazım. 

>> *__Soft Voting__* kendinden emin modele daha fazla ağırlık verdiği için genelde *__hard voting__* modelinden daha iyi çalışır. Ama en iyisine karar vermek için ikisini de denemekte fayda var.

## Bagging (Bootstrap aggregating) ve Pasting

*__Hard Voting__* ve *__Soft Voting__* yöntemlerinde aynı training data üzerinde eğitilmiş farklı modeller kullanmıştık. *__Bagging__* ve *__Pasting__* yönteminde ise farklı training datalar üzerinde eğitilmiş modeller kullanmayı hedefliyoruz. Aslında hepsinde aynı şeyi yapmaya çalışıyoruz: 

>>Eğitilen her modelin olabildiğince farklı şeyler öğrenmesi ve birbirlerinin hatalarını örtmeleri.

Tabiki training datamız değişmeyeceği için farklı datalarda eğitimi training datamızı random bir şekilde bölerek yapabiliriz. Burada datayı nasıl böldüğümüze göre yöntemimizin ismi *__Bagging__* veya *__Pasting__* oluyor. 

>>Örneğin 1000 tane decision tree eğiteceğiz. Bunun için 1000 adet training datasına ihtiyacımız var. Datamızı 1000 kez random sampling yaparak aynı boyutlarda 1000 adet random subset oluşturabiliriz. Her bir decision tree için bir adet training dataseti oluşturmuş olduk. Ancak aynı training datası üzerinde 1000 kez random sampling yaparken bazı subsetlere aynı training datalarını katma ihtimalimiz hayli yüksek. İşte bu yönteme yani modellerin aynı verileri görmesine izin verdiğimiz yönteme *__Bagging__* diyoruz. Eğer oluştuğumuz tüm subsetler farklı elemanlar içeriyorsa buna da *__Pasting__* diyoruz.

Training kısmını bu şekilde yaptıktan sonra test kümemizde bu decision treeler ile tahminleme yapabiliriz. Tahminleme yaparken genelde classification için hard/soft voting, regression problemleri için ise ortalama alma yöntemi kullanılır. 

Bagging/Pasting yöntemi, tahminleyici modellerin birbirinden bağımsız olarak çalıştığı ve sonuçların birleştirildiği için varyansı azaltma ve genelleme performansını artırma potansiyeline sahiptir. Aynı zamanda, overfitting'e karşı da daha dirençli olabilir. Bagging, özellikle karar ağaçları gibi yüksek varyanslı tahminleyicilerle kullanıldığında etkili sonuçlar elde etmek için yaygın olarak kullanılan bir tekniktir.

>> Bagging/Pasting yönteminin en güzel yanı paralel şekilde eğitimin yapılabilmesidir. Verdiğim örnekte de görebileceğimiz gibi 1000 adet decision tree ayrı ayrı eğitilebilir. Bu sebeple Paralel Processing kullanabilirsiniz. Yani bilgisayarınızdaki farklı cpu corelarına işlemi bölerek eğitimleri aynı anda gerçekleştirebilir ve süreci hızlandırabilirsiniz.

>> Genelde kullanılan default yöntem Bagging'dir. Ama ikisininde artıları ve eksileri veri setine göre değiştiği için eğer vaktiniz varsa Pasting'i de test etmenizi öneriririm.
## Out-of-Bag Evaluation / Score

Bagging yöntemini kullanırken nasıl bir evaluation kullanabiliriz sorusunun aslında iki cevabı var. Tabiki de training'e başlamadan önce kendi evaluation setinizi oluşturup ensemble tahminleyicinizi burada evaluate edebilirsiniz. En doğru ölçüm bu şekilde yapılır. 

Ancak elinizde az sayıda training datanız varsa, datanızı ekstra bölmek istemeyebilirsiniz. Böyle bir durumda şunu düşünün. Bagging yönteminde her bir tahminleyicimiz training datadan random olarak seçilen bir subsamples üzerinde eğitiliyor. Yani aslında geriye hiç kullanılmayan bir data kısmı kalmış oluyor. Bu olay her bir tahminleyici için oluşuyor ve her tahminleyicinin kendi evaluation setini oluşturmuş oluyoruz. 

Peki ölçümü nasıl yapıyoruz ? Mesela elimizde bagging yöntemi kullanılarak ensemble öğrenimi yapılan 10 adet Decision Tree olsun. Her bir tree için ayrı ayrı subsetler oluşturduk ve ayrı ayrı training ve evaluation setleri oluşturduk. Ardından her bir tree'nin kendi evaluation setlerindeki örnekler için tahminlerini alıyoruz. Ardından bu örneklerin ortak olduğu treelerin tahminlerinin ortalamasını alabiliriz. Örneğin bir örnek 1., 6. ve 9. Decision Tree'lerin evaluation setlerinde ortak bulunuyor. Bu örneği bu üç treede de tahminleyip ortalama cevabı (herhangi bir voting yöntemi ile) alıyoruz. Bunu tüm evaluation set'e yaptığımızda artık elimizde tahminler ve doğru cevaplar oluyor. Bundan sonrasında istediğiniz ölçümü yapabilirsiniz.


## More Diversity (Feature Selection)

Yukarıdaki kısımlarda anlattığım gibi ensemble learning yaparken esas amacımız farklı yöntemler kullanarak training datası üzerinde farklı yanlışlar yapılmasını sağlamaktır. Böylelikle bir modelin yanlışını başka bir model kapatabilir. 

Bunu uygulamanın bir başka yöntemi ise random olarak feature seçimidir. Bagging ve Pasting yöntemleriyle nasıl farklı modeller için training datamızdan random şekilde örnekler seçiyorsak, aynı şekilde farklı modeller için farklı featurelar da seçebiliriz.

### Random Forest

Bagging yöntemlerinin belki de en ünlüsü ve en çok kullanılanı Random Forest'tır. Aşağıdaki sklearn kodları hemen hemen aynı şeyleri yapar:

```python
bagging_classifier = BaggingClassifier(
DecisionTreeClassifier(), n_estimators=500,
bootstrap=True, n_jobs=-1)
```

```python
from sklearn.ensemble import RandomForestClassifier
randomForest_classifier = RandomForestClassifier(n_estimators=500, n_jobs=-1)
```

Random Forest algoritması içerisindeki Decision Treelere daha fazla çeşitlilik katmak için ağaçlar büyürken tüm featurelar arasından en iyi feature'ı seçmek yerine [Decision Tree Splitting](https://github.com/berkedilekoglu/machine-learning/tree/main/notes/Decision_Trees#information-gain) random olarak seçtiği featurelar arasından en iyi feature'ı seçmeye çalışır.

>> Random Forest'ı oluşturan ağaçların kullandığı featureların impurity değerleri hesaplanarak (impurity'i ne kadar azalttığı) tüm ağaçlara göre ortalaması alınarak feature importance ölçümü yapılabilir. Scikit-learn kütüphanesi bu yöntemi kullanarak size feature importance ölçümü verir.

### Extra Trees

Extra Tree'lerde bootstrap yöntemi kullanılmaz. Yani her bir ağaç aynı (Tüm) training datası üzerinde eğitilir. Ancak sklearn kullanırsanız bootstrap kullanması için hyperparametresini True olarak set edebilirsiniz.

Extra Treeler Random Forest'ın random olarak feature subset'i kullanmasının yanısıra split edilecek karar threshold'unu da random belirler.

Decision Tree oluştururken, her bir düğümde kullanılan bir özelliğin sınıflandırma yapmak için bir eşik değeri belirlenir. Bu eşik değeri, veri örneklerini iki gruba ayırmak için kullanılır. Örneğin, bir sayısal özelliğin eşik değeri, veri noktalarını belirli bir değerden büyük veya küçük olacak şekilde ikiye bölebilir.

Ancak Extra Trees'da (Extremely Randomized Trees), normal karar ağaçlarından farklı olarak, ağaç oluşturma sürecinde her bir özellik için rastgele bir eşik değeri seçilir. Bu rastgele seçim, özellik değerlerinin tamamını dikkate alarak yapılır, yani belirli bir özelliğin tüm değer aralığına göre bir eşik değeri belirlenir. Bu, her bir özelliğin farklı noktalarda bölmeler yapmasını sağlar ve ağaçların daha bağımsız ve rastgele olmasını sağlar.

Örneğin, Decision Tree algoritmasında bir özelliğin değeri 5'ten büyük veya küçük olacak şekilde bir eşik değeriyle bölebilirken, Extra Trees algoritmasında bu eşik değeri rastgele olarak belirlenebilir. Bu, her bir ağacın farklı bölmelerle oluşturulmasını sağlar ve ağaçların daha çeşitli ve bağımsız olmasını sağlar.

Bu rastgele eşik değerleri seçimi, ağaçların daha da çeşitlendirilmesini sağlar. Bu çeşitlilik, ensemble yöntemi olan Extra Trees'ın daha fazla varyansı azaltma potansiyeline sahip olmasını sağlayabilir.


## Boosting (Hypothesis Boosting)

Boosting methodunda genel amaç modelleri ard arda train ederek birbirlerinin yanlışlarını öğrenmelerini sağlamaktır. En popüler boosting methodları:

* AdaBoost (Adaptive Boosting)
* Gradient Boosting

Burada modelleri ard arda train etme deyimi aslında boosting ve bagging'i ayıran en önemli ayrımlardan biridir. Modeller birbiri ardına train edileceği ve birbirlerinin yanlışlarını öğreneceği için boosting yöntemlerinde paralel processing yapılamamaktadır.

### AdaBoost

(Image Taken From Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems)[https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/]
Aslında AdaBoost yöntemini anlamak için biraz weighted loss üzerine düşünmemiz gerekiyor. Bir öğrenme gerçekleşirken esas amacımız loss'u azaltmaktır. Bunu yapılan hatayı azaltmak olarak da düşünebilirsiniz. Zaten loss fonksiyonları tahmin edilen değer ve gerçek değer arasındaki farkı ölçerek hesaplanır. Hata payı ne kadar fazla ise loss'umuz o kadar fazladır. 

Genelde loss hesaplanırken her bir örneğe eşit önem verilir. Ancak bazı örnekler modeller tarafından kolaylıkla tahmin edilebilirken ve öğrenilebilirken bazı örnekleri modelin öğrenmesi zor olmaktır. Bunlara çoğunlukla "Hard Examples" ismi verilir. Bu öğrenilmesi zor örnekleri belirleyip bu örnekler için yapılan hatalara biraz weight eklersek modeller eğitim sırasında bu örneklere daha fazla dikkat ederler.

AdaBoost'un çalışma prensibi de bu prensipe dayanır. Bir weak learner eğitildikten sonra training seti üzerinde tahmin yaptırılır ve yaptığı hatalı tahminler işaretlenir. Bu örneklere weight eklenerek bir sonraki weak learner eğitilir. Böylece 2. eğitilen model 1. eğitilen modelin tahmin edemediği örnekleri daha iyi öğrenir. Bu süreç istenilen sayıda learner eğitildiğinde son bulur. 

Son olarak kullanılan tüm modellerin predictionları weighted vote yapılarak output prediction elde edilir.

>> Yukarıda anlatılan işlemler ard arda yapılmak zorunda olduğundan paralel bir eğitim işlemi yapılamaz. Sadece weak learnerlar kendi içlerinde paralel eğitilebilir (Eğer weak learner paralel eğitime müsaitse).

### Gradient Boosting

Gradient Boosting'de AdaBoost gibi sequential olarak hata öğrenimine dayalı bir ensemble yöntemidir. Ancak bu yöntemde sample'lar için weight kullanılmaktansa Residual Error dediğimiz arda kalan hatalar kullanılır. Classification ve Regressionda problemlerinin her ikisi için de bu yöntem kullanılsa da classification'da matematiksel olarak sınıf olasılıklarını hesaplarken birkaç hesaplama daha yapılmaktadır. Bu sebeple Gradient Boosting'in çalışma prensibinin daha iyi anlaşılması için bir regression problemi üzerinden anlatıcam:

1) Regression problemimizde training datamızı X_train ve y_train olarak düşünün. 
2) Bir weak learner seçelim, diyelim ki Decision Tree.
3) Bu Decision Tree'yi X_train ve y_train kullanarak eğitelim ve tahminleri y_pred_1 olarak alalım.
4) Residual error dediğimiz değerler y_residual = y_train - y_pred_1 değerleridir.
5) Şimdi başka bir Decision Tree'yi X_train ve y_residual üzerinde eğiteceğiz.
6) Yeni residual error'umuz y_residual = y_train - (y_pred_1 + y_pred_2) olacak
7) Bunu istediğimiz kadar learner eğitene kadar devam edebiliriz.

En sonunda ensemble predictionlar için basitçe tüm modellerin predictionları toplanabilir veya voting yöntemleri kullanılabilir.

>> **_XGBoost_**, Gradient Boosting yöntemini kullanan en popüler algoritmalardan başında gelmektedir. Unstructured datalarda çok iyi çalışır ve çoğu Kaggle yarışmasında tercih edilir. Çoğu problemde Neural Networklere yakın veya daha iyi sonuçlar almanızı sağlar. XGBoost weak learner olarak Decision Tree kullanır ve Gradient Boosting yöntemi ile tahminleme yapar. Ancak çok daha hızlı olması ve optimum performans için tasarlanmıştır. [Şuradan](https://xgboost.readthedocs.io/en/stable/install.html) indirebilirsiniz.

### Stacking

Stacking ensemble, birden fazla modelin eğitildiği ve bu modellerin tahminlerinin birleştirilerek nihai bir tahmin oluşturulduğu bir tekniktir. Bu genellikle, temel modellerin tahminlerini girdi olarak alan ve bir nihai tahmin çıktısı üreten bir 'meta-model' eğiterek gerçekleştirilir. Stacking'in temel fikri, birden fazla modelin tahminlerinin birleştirilmesinin, tek başına herhangi bir modelden daha iyi performansa yol açabileceğidir.

Yukarıdaki şekilde görebileceğiniz gibi, bir stacking ensemble farklı modellerin tahminlerini kullanarak nihai bir sınıflandırıcıyı eğitmektedir. Burada dikkat edilmesi gereken önemli bir nokta, sklearn stacking sınıflandırıcı içinde bir Keras modeli kullanıyorsak girdi şeklini belirtmemiz gerektiğidir. Yukarıdaki şekilde gösterildiği gibi, sinir ağımıza veri girişinin boyutu artık eğitim verilerimizle aynı olmayabilir.


### Kişisel Not

Son zamanlarda oldukça kullanılan oto-ml frameworkleri genelde farklı modellerin stack edilmiş hallerini kullanıyorlar. Özellikle kurulan 4-5 stacking ensemble modeli soft voting ile ensemble yapıldığında modellerin hataları oldukça iyileştirilebiliyor. Tabiki bu her ml taskında olduğu gibi datadan dataya değişen bir durum ancak artık modelinizi geliştiremediğiniz durumlarda bunu deneyebilirsiniz.
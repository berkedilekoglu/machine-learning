# Decision Trees

Türkçeye 'Karar Ağaçları' olarak çevrilse de ML alanında çalışacak kişilerin bu terimlerin ingilizcelerini bilmesi gerektiğini düşündüğüm için terimleri ingilizce yazmaya devam edicem. Decision Treeler SVM'ler gibi hem regression hem de classification konularında kullanılabilen algoritmalardır. Random Forest gibi daha kompleks algoritmaları oluşturan bileşenler olduğundan mülakatlarda sorulabilir.

## Training

Decision Treeler eğitilirken training setimizdeki featurelara bakarak kuralları öğrenirler. Aslında basitçe bir if-then kurallar bütünü olarak da değerlendirebiliriz. Oluşan ağaca baktığımızda, anlaşılması en kolay algoritmalardandır.
![[decision_tree_example](https://www.datacamp.com/tutorial/decision-tree-classification-python)](https://github.com/berkedilekoglu/machine-learning/assets/19657350/0af77f4a-6fcc-4bc3-a28b-1be47d668b72)

Training için önemli adım ağacı split ederken nasıl split etmemiz gerektiğidir. Bunun için genel olarak __Impurity__ kullanılır. 

![[impurity](https://www.baeldung.com/cs/impurity-entropy-gini-index)](https://github.com/berkedilekoglu/machine-learning/assets/19657350/bd949c3f-92c8-4a40-a076-716c0c30466f)

Figürde gördüğümüz gibi en sağda bulunan ve hepsi aynı class'a ait olan örnek minimum impurity örneğidir. Figürde sola doğru gidildikçe impurity artar. Farklı impurity ölçme şekilleri şunlardır:
- Entorpy
- Gini index
- Misclassification Error

### Entorpy
```math
Entropy = \sum_{i} -P(i)\log_2(P(i))
```
Örneğin en soldaki figürde elimizde 8 adet mavi 8 adet kırmızı örnek var. Her iki class için şu şekilde hesaplama yapmalıyız:
* 8/16 mavi olma olasılığı ve $`log_2(8/16)`$
* 8/16 kırmızı olma olasılığı ve $`log_2(8/16)`$
* Toplarsak -> $`Entorpy = (-1/2)*(-1) + (-1/2)*(-1)`$
* Yani Entorpy = 1 olacaktır.

En sağdaki figür için hesaplarsak elimizde 16 adet kırmızı 0 adet mavi örnek var:
* 16/16 mavi olma olasılığı ve $`log_2(16/16)`$
* 0/16 kırmızı olma olasılığı -> Burada direkt 0 olarak ölçüm yapılır
* Toplarsak -> $`Entorpy = (-1)*(0) + (0)`$
* Yani Entorpy = 0 olacaktır.

### Information Gain

Peki Entorpy kullanarak nasıl split işlemi uygulanır ? İşte information gain burada devreye giriyor. Tüm split opsiyonları arasından Information Gain'i en yüksek olan split'i seçerek ilerlemek yaygın kullanılan bir yöntem. Bunu hesaplamak için de aslında Entropy'nin o split seçeneği ile ne kadar azaldığına bakıyoruz. Basitçe:
- Information Gain = Parent Entropy - Weighted Average of Child Entropies

![entropy](https://github.com/berkedilekoglu/machine-learning/assets/19657350/67293d29-a9f7-4ee6-87dd-57cae1ee2a53)


Örnek information gain hesaplaması resimde gösterilmiştir. __Information Gain__'in fazla olduğu split seçeneği en iyi opsiyon olarak değerlendirilebilir. Bu çıkarımı şöyle de yapabiliriz. En iyi split tüm classların tamamen ayrılacağı splittir. Mesela 2 adet class için bir parent node'umuz var. Bu node'da 2 adet class 1, 2 adet ise class 2 örneği bulunuyor. Bu durumda ilk Entorpy hesaplama örneğinde hesapladığımız gibi parent Entorpy'miz 1 olacaktır. Split durumunda 2 adet yeni node oluştuğunu ve classları tamamen ayırdığımızı düşünelim. Böylece entropy tam olarak ikinci entropy hesaplama örneğimizdeki gibi her bir node için 0 çıkacaktır. Bu sebeple en iyi split durumu için Information Gain 1 olarak hesaplanır. Tam tersi durum için ise 0 olarak hesaplanır.

### Gini Index

Gini index deafult olarak sklearn kütüphanesinin decisition tree classifier'ı için kullanılan impurity ölçme yöntemidir. Bir node eğer Gini değeri '0' ise _pure_ olarak ifade edilir. Buna o node'daki her örneğin aynı class'a üye olması diyebiliriz. Matematiksel olarak şu şekilde ifade edilir:

```math
GINI_{i} = 1 - \sum_{k=1}^{n}P(i,k)^2
```
-  $`n`$ toplam class sayısıdır 
-  $`P(i,k)`$ i node'unda bulunan k class'ına ait örneklerin, i node'unda bulunan toplam örnek sayısına oranıdır.

![gini](https://github.com/berkedilekoglu/machine-learning/assets/19657350/5385b922-45c4-4fa2-991d-c46233cbaad3)


Gini hesaplamasıyla yaptığımız örnekten de göreceğiniz gibi sonucu seçerken __Information Gain__ kullanıyoruz. 

> Misclassification Error aslında loss dediğimiz hata hesaplama yöntemidir. Bunu detaylı daha sonra anlatıcam. Ancak basitçe bir node'da 1 adet class 1, 12 adet class 2 den varsa buradaki hatayı $`1-max(1/13,12/13) = 1-12/13 = 1/13`$ olarak bulabiliriz. 

### Impurity Ortak Özellikler

Farklı impurity ölçme yöntemlerinin ortak özellikleri vardır:
- _Pure_ nodelar için yani node içindeki örneklerin hepsinin sadece 1 class'a ait olduğu nodelar için tüm impurity ölçme yöntemleri 0 değerini alır.
- Eğer node içindeki örnekler uniform olarak dağıldıysa impurity ölçme teknikleri maximum değerlerini alır.

### Gini vs Entropy

Aslında çoğu zaman Entropy veya Gini kullanmak çok bir fark oluşturmaz ve aynı ağaçları yaratmayı sağlar. Sklearn'de Gini default seçenek olarak kullanılmıştır. Bunun başlıca sebebi Gini ölçümünü kullanmanın birazcık daha zamandan kazandırmasından dolayıdır. Ancak bazı durumlarda Gini en sık bulunan class'ı nodelarda izole etmeye başlar. Entropy ise genellikle daha balans ağaçlar oluşturur.

## Regularization Hyperparamters

Decision tree'ler training datası hakkında bir öngörüde bulunmazlar. Yani datayı öğrendikçe koşullar oluşur ve ağaç büyür. Bu sebeple hiçbir müdahalede bulunmadığımız bir ağaç tüm datayı ayırmaya çalışıp çok derinleşecektir.

## Yeteri kadar derin bir decision tree mükemmel midir ?

Yukarıda gördüğümüz özelliklere göre bir decision tree'yi yeteri kadar derinleştirirsek tüm leaf nodeları pure hale getirebiliriz. Peki bunu yaptığımızda o mükemmel modele ulaşmış mı olacağız. Malesef işler pek böyle yürümüyor. Burda unutmamamız gereken şey bir model ne kadar genelleyici olursa o kadar iyi olacağıdır. Eğer çok derin bir tree kurarsak sadece training setimizi ezberlemiş oluruz. Yani __Overfitting__ ile karşılaşmış oluruz. Peki bunu nasıl çözmeliyiz:
- Early Stopping
- Pruning

## Early Stopping in Decision Trees

1) Ağacımızın maximum depth parametresini (```max_depth``` in sklearn (default olarak None'dır yani limit konulmamıştır)) ayarlarsak bu training sırasında o depth'e ulaştığımızda trainingi sonlandırır.
2) Bir node'da çok az sayıda örnek varsa, o node noise'lara karşı hassas hale gelmiş demektir. Şöyle düşünün elinizdeki 16 örnekten 3 tanesi noisy örnekler. Eğer split ettikten sonra elde edeceğiniz node'da 3 örnek varsa ve bunların hepsi noisy datadan gelirse o node tamamen yanlış seçimler yapacaktır. Bu sebeple node'un minimum örnek sayısını belirleyerek training'i durdurabiliriz. Eğer bir node şu sayıda örneğin altına düşerse training'i durduralım gibi. (```min_samples_split``` (bir node'un split edilebilmesi için içerisinde olması gereken minimum örnek sayısı), ```min_samples_leaf``` (leaf node'un içerisinde bulunması gereken minimum örnek sayısı) in sklearn)

## Pruning

Pruning aslında bir engelleme olmadan training'i tamamlamakla başlar. Mesela bir ağaç oluşturdunuz. Artık en alt node'lara (leaf node) bakıp yukarı doğru (bir üst sub-tree) birleştirme yapabilirsiniz. Eğer performans artarsa, subtree'yi leaf node ile birleştirebilirsiniz.

> Not: Her iki yöntemin de içerisinde testler geçtiğini unutmayın. Bu sebeple her iki yöntem de validasyon datası kullanır. Böylece overfitting'in azalacağına emin olurlar.

## Decision Trees For Regression

Regression için split işlemi tüm child nodelardaki hatalar ölçülerek yapılır. $`\sum(y_{target}-y_{predicted})^2`$

Bir prediction verilirken'de, test datasının tahminleneceği node'un train edildiği örneklerin ortalaması verilebilir.

## AVANTAJLAR

- Decision tree'leri oluşturmak maliyetli değildir.
- Prediction konusunda çok hızlılardır. -> $`O(log_{2}(m))`$
- Ağaca bakıldığında anlamlandırmak çok kolaydır.
- Farklı tiplerdeki featurelar için iyi çalışırlar (kategoriler, numaralar, vb.) 
- Basit decision tree'ler daha iyi ensemble modeller için birleştirilebilir

## Dezavantajlar

- Optimal bir decision tree oluşturmak NP-Complete bir problemdir
- Deneysel (Heuristic) yaklaşımlarla optimal tree'ler oluşturabilir
- Pruning veya early stopping kullanılmazsa kompleks datalar için training çok süre alabilir
- Decision treeler çok hassastır. Yani bir training örneğini silmek ya da eklemek decision boundary'leri tamamen değiştirebilir.
- Linear ayrılabilen datalarda çok iyi çalışırken, Linear ayrılamayan datalarda genelleme problemleri yaratabilir. Buna çözüm olarak PCA(Principal Component Analysis) ile featureların düzenlenmesi denenebilir.

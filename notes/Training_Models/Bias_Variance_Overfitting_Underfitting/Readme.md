# Bias Variance Tradeoff #

Bias ve Variance kavramlarınızı anlatırken direkt türkçeye çevirileri üzerinden gideceğim çünkü ifade ettikleri anlamlar tam olarak istediğimiz şeyleri karşılıyor. 

## Bias (Ön Yargı) ##

Bias ön yargı demektir. Peki bir modelin ön yargısının fazla olması ne anlam ifade eder ? Aslında makine öğrenmesi modellerini çoğu zaman insanlara benzetiyorum. Ön yargılı insanların ortak özelliği bir konuda daha önceden inandıkları şeyler olduğu için öğrenmeye isteksiz olmaları ve o konuda bir şey öğrenmeden yorum yapmalarıdır. İşte tüm olay burada yatıyor: __bir şeyler öğrenmeden yorum yapmak__. Makine öğrenmesi modelleri de aynı bu şekildedir. Eğer yüksek bias'a sahipse o konuda çok fazla bir şey öğrenmemiş ve ona göre tahminleme yapacak demektir. Bu da training datasından yeteri kadar şey öğrenilmediğini gösterir. Bu modellerin hata oranları hep training datası için hem de test/validation datası için fazla olacaktır. 

Bunu şöyle de düşünebilirsiniz. Teknolojiye karşı çok ön yargılı bir insanla söyleşi yapacaksınız ve konunuz son model akıllı telefonlar. Kendisine bir haftalığına kurcalaması için herhangi bir akıllı telefon bırakıyorsunuz. Bu aslında kişiye verdiğiniz training setiniz. Ancak o kişi ön yargıları nedeniyle bu telefonu açıp kurcalamaktansa birazcık bakıp rafa kaldırıyor. Bir hafta sonra o kişiye başka bir akıllı telefon üzerinden sorular yöneltiyorsunuz. Yani aslında test veya validation setiniz üzerinde performansını değerlendiriyorsunuz. Ama hiçbir fikri olmadığı için kendi genellemelerini söylecek ve hatalı cevaplar verecek. Peki diyorsunuz o zaman sana verdiğimiz yani training seti üzerinden sorular soralım. Ona da doğru düzgün bakmadığı için yine kötü bir performans sergileyecektir.


## Variance (Çeşitlilik) ##

Variance ismindeki çeşitlilik anlamı aslında modelin eğitim setindeki verilere göre farklı farklı tahminler yapmasından geliyor. Evet biz modellerimizde çeşitlilik istiyoruz ama genelleme kavramının dışına çıkacak kadar çeşitli tahminler hiçbir zaman istemeyiz. Çünkü bu hata anlamına gelir. 

Bunu teknik ve teknik olmayan iki örnekle anlamaya çalışalım. Normalde olması gereken yani istediğimiz şey modelin bir datadan bir şeyler öğrenip daha sonra yine aynı konuda daha önce görmediği şeylere de doğru cevaplar verebilmesi. Yani burda bir stabillik söz konusu. Eğer modelimiz öğrenmesi için verdiğimiz datadaki her noktayı ezberlerse burada hem noise'ları hem de dataları öğrenmiş olacak. Bu sebeple daha önce görmediği konulara cevap verirken iyi cevaplar veremeyecektir.

Yine telefon örneğine bakalım. Bu sefer teknolojiye karşı öğrenmeye aç ama daha önce hiç akıllı telefon görmemiş bir insanla söyleşi yapacaksınız ve konunuz yine son model akıllı telefonlar. Kendisine bir haftalığına kurcalaması için piyasaya sürülmüş ilk android telefonu bırakıyorsunuz. Bu aslında kişiye verdiğiniz training setiniz. Bu sefer bu kişi verdiğiniz telefonun her şeyini öğreniyor. Bir hafta sonra gelip o kişiye öncelikle verdiğiniz telefonla alakalı sorular soruyorsunuz. Sorularınızı çok iyi bilecektir. Ancak çıkartıp son model bir iphone hatta son model bir android telefonla alakalı sorular sorarsanız verdiği yanıtların çoğu doğru olmayacaktır. Bu kişinin eğitim datasına bir de ilk iphone modelini eklediğinizi düşünün ve yine bu modeli de ezberlesin. Ancak ona yine son model telefonlar üzerinden sorular sorarsanız bu sefer verdiği cevaplar değişir. Çünkü bu sefer farklı bir telefonu da ezberine kattı. Buradaki sorun yine doğru cevaplar vermeyecek olmasıdır. İşte burada gelen çeşitlilik kavramı Variance'dır ve fazlası risklidir :)


## Overfitting Underfitting ##

Machine Learning'de amacımız her zaman çok iyi bir şekilde genelleme yapmaktır. Bunu gerçek hayattan şöyle düşünebilirsiniz. Örneğin matematik çalışıyorsunuz ve yaş problemlerine çalışacaksınız. Size çalışmanız için 100 adet yaş problemi sorusu veriliyor. Sonrasında yaş problemi içeren bir sınava gireceksiniz. Çalışırken aslında 2 seçeneğiniz olur. İlki bu 100 soruyu hiç anlamadan cevaplarıyla beraber ezberlemek. İkincisi ise bu 100 soruyu anlayarak mantığı ile birlikte çalışmak ve öğrenmek. İlk seçeneği yaparsanız size bu 100 soruyu tekrar sorarsak hepsini doğru bilirsiniz çünkü artık ezberinizde. Ama bu sorulardan farklı yaş problemleri sorduğumuzda büyük ihtimalle çok fazla yanlışınız çıkacaktır. O sebeple genel olarak mantığıyla öğrenmek, ezberlemekten daha iyidir.

İşte bu örnekte size verilen 100 soru sizin training setiniz. Daha sonra yapılan ve hiç görmediğiniz sorular ise validation setinizdir. Eğer training setinizde çok iyi performans gösteriyor ama validation setinizde kötü bir performans gösteriyorsanız __Overfit__ etmiş olursunuz. Eğer ikisinde de kötü bir performans gösteriyorsanız __Underfitting__ etmiş olursunuz yani öğrenme gerçekleşmemiştir.

Peki __Underfitting__'e neler yol açabilir ve nasıl çözülür:

1) Modeliniz fazla basittir ve eğitim setini anlayamamış olabilir. Buna güzel bir örnek polynomial bir fonksiyonu lineer regression ile çözmeye çalışmak gösterilebilir. Bu sebeple daha komplex bir model kullanabilirsiniz.

2) Datanızdaki featurelar problemi çözmeye yetmiyor olabilir. Daha fazla alakalı feature eklemek bu sorunu giderecektir. Bunu da polynomial regression konusunda görmüştük.

3) Datanızda çok fazla noise olabilir. Yani kirli bir dataya sahip olabilirsiniz ve model bu sebeple datadan bir şey öğrenemiyor hale gelebilir. Noise temizlemek bu sorunu hafifletecektir.

4) Eğitim süreniz az gelmiş olabilir. Eğitim süresini arttırarak datanızı daha iyi öğrenebilirsiniz.

Peki __Overfitting__'e neler yol açabilir ve nasıl çözülür:

1) Modeliniz çok kompleks olabilir ve datanızı ezberliyordur. Bu sebeple modeli daha basitleştirmek overfitting'i önler.
2) Training datanız yeteri sayıda değildir. Bu sebeple modeliniz eldeki veriyi çok fazla öğreniyor (ezberliyor olabilir). Datanızı arttırmak öğrenmenizi genişletir ve overfittingi azaltır.
3) Early stopping kullanabilirsiniz. Early stopping modelinizin validation datanız üzerindeki hatasına bakarak uygulanan bir tekniktir. Validation datanızdaki hata artmaya başlayınca eğitimi durdurur.
4) Regularization teknikleri: Aslında bu tekniklerde amaç öğrenme anına müdahale ederek işin içine randomlık koymak ve ezberi bozmaktır. Bunları görücez.

## Trade-off ##

Img

Figürden ve örneklerden fark edeceğiniz gibi yüksek variance ve düşük bias bize overfitting getirir. Düşük variance ve düşük bias ise bizi düşlediğimiz mükemmel sonuca götürür :) Ama arada bir denge vardır. Yani overfit ediyorsanız modeli basitleştirin demek aslında bias'ı arttırın demektir. Ama bias'ı çok arttırırsanız bu sefer de modeliniz underfit etmiş olacaktır yani variance'ınız düşecektir. Aslında bu alanda yapacağınız tüm deneyler bu aradaki dengeyi en iyi şekilde sağlamaya çalışmakla geçer :)
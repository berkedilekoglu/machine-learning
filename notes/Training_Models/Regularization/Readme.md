# Regularization #

[Overfitting](https://github.com/berkedilekoglu/machine-learning/tree/main/notes/Training_Models/Bias_Variance_Overfitting_Underfitting#overfitting-underfitting) konusunu daha önce görmüştük. Gerçek hayatta uğraşacağınız hemen hemen her problemde sorunumuz bunun üzerinden gelmek olacak. Şöyle düşünün size verilen dataya uygun bir model seçmeniz gerekecek ve çeşitli denemeler yapacaksınız. Denediğiniz modeller kompleksleştikçe training datasını daha iyi öğrenecek ama büyük ihtimalle overfit edip validation datanız üzerindeki hata oranınız fazla olacak. Peki bu overfittingle nasıl savaşacağız yani nasıl engellemeye çalışacağız. Bundan kısaca bahsetmiştim şimdi ise detaylarına giricez ve bu yöntemlerden bazılarını Linear Regression'a entegre etmeye çalışacağız.

Overfitting ile savaşmanızı sağlayan ve modelinizin overfit etmesini engelleyen yöntemlere kısaca Regularization yöntemleri denir. 

>> Karşınıza bir mülakatta overfitting nedir ve nasıl engellersiniz diye bir soru gelirse "Regularization yöntemlerini kullanırım bu yöntemler şunlardır diyip açıklama yapabilirsiniz."
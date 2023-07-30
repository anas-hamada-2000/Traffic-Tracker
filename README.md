# Traffic-Tracker
## YAPILAN ÇALIŞMALARIN ÖZETİ
Bir yol için belirli bir zaman dilimindeki trafik yoğunluğuna ilişkin bir rapor oluşturabilmek.

Bir yoldaki trafiğin yoğunluğunu bilmek için, belirli bir zamanda o yoldan geçen araba sayısını bilmemiz gerekir.

İlk önce bazı trafik kavşak videoları topladım.

Videodaki arabaları tespit etmek için yolo algoritması kullandım.

Videoda yolları belirttikten sonra yolo algoritmasını kullanarak arabaların bu yollardaki konumlarını ve araba olduklarına dair güvenin ne olduğunu belirtiyoruz.

güven yüksekse arabanın çevresine bir dikdörtgen çizeriz ve o yoldaki toplam arabaya 1 ekleriz.

arabaları saymak için SORT(Simple Online Realtime Tracking) algoritmasını uyguladım.
## MATERYAL VE METOT
### Materyal
* TensorFlow: makine öğrenimi için ücretsiz ve açık kaynaklı bir yazılım kütüphanesidir

* Matplotlib: 2 boyutlu grafikler hazırlamamızı sağlayan bir python kütüphanesidir

* NumPy: Python programlama dili için büyük, çok boyutlu dizileri ve matrisleri destekleyen, bu diziler üzerinde çalışacak

* OpenCV: bilgisayarla görü, makine öğrenimi, görüntü işleme, video analizi gibi uygulamalar için kullanılan devasa bir açık kaynak kodlu kütüphanedir

### Metot
* YOLO: konvolüsyonel sinir ağları kullanarak nesne tespiti yapan bir algoritmadır.

  Algoritma aşağıdaki dört yaklaşıma dayalı olarak çalışır:
  
      * Kalan bloklar: Bu ilk adım, orijinal görüntüyü eşit şekle sahip NxN ızgara hücrelerine bölerek başlar.
  
      * Sınırlayıcı kutu regresyonu: Bir sonraki adım, görüntüdeki tüm nesneleri vurgulayan dikdörtgenlere karşılık gelen sınırlayıcı kutuları belirlemektir. Y = [pc, bx, by, bh, bw, c1, c2],(Y, her sınırlayıcı kutu için son vektör temsilidir)
  
      * Birlikler Üzerinden Kesişme: Bu mekanizma, gerçek kutuya eşit olmayan sınırlayıcı kutuları ortadan kaldırır.
  
      * Maksimum Olmayan Bastırma: MOB'yi yalnızca en yüksek algılama puanına sahip kutuları tutmak için kullanabiliriz.

* SORT Algoritması: nesne izlemeyi gerçek zamanlı olarak işleyen ilk algoritmalardan biri,  algoritma aşağıdaki gibi 4 temel bileşenden oluşur:

      * Algılama: Takip etme modülündeki ilk adımdır. Bu adımda, bir nesne detektörü çerçevedeki izlenecek nesneleri algılar. Bu tespitler daha sonra bir sonraki adıma aktarılır.

      *Tahmin: Bu adımda, tespitleri geçerli çerçeveden sonraki çerçeveye yayarız; bu, sabit hız modeli kullanarak bir sonraki çerçevedeki hedefin konumunu tahmin eder.

      * Veri ilişkilendirme: Artık hedef sınırlayıcı kutumuz ve algılanan sınırlayıcı kutumuz var. Bu nedenle, bir maliyet matrisi, her bir algılama ile mevcut hedeflerden tahmin edilen tüm sınırlayıcı kutular arasındaki birleşim üzerinden kesişim (IOU) mesafesi olarak hesaplanır. Tespit ve hedefin IOU'su, IOUmin olarak adlandırılan belirli bir eşik değerinden düşükse, bu atama reddedilir. Bu teknik, tıkanma sorununu çözer ve kimliklerin korunmasına yardımcı olur.

      * İzleme Kimliklerinin Yapılandırması ve Silinmesi: Bu modül, kimliklerin kurulumundan ve silinmesinden sorumludur. Temel kimlikler, IOUmin'e göre oluşturulur ve yok edilir. Tespit ve hedef sınırı olması IOUmin'den azsa, izlenmeyen nesneyi belirtir.

* KALMAN Filter: zaman içinde ölçülen değişkenlerin değerlerini tahmin etmek için tasarlanmış bir algoritma

* Hungarian Algoritması: geçerli çerçevedeki bir nesnenin önceki çerçevedeki ile aynı olup olmadığını söyleyebilir

* Intersection Over Union (IOU): iki kutu arasındaki örtüşme derecesini ölçen bir sayıdır
## UYGULAMA VE SONUÇLARI
kodumuz videonun her framinde derin öğrenme kullandığı için onu GPU üzerinde çalıştıracağız

![Screenshot 2023-06-21 222213](https://github.com/anas-hamada-2000/Traffic-Tracker/assets/68608987/d88a6321-192b-4194-b986-3be605ac3e06)

oluşturduğumuz modeli iki video üzerinde uygulayadım

![Screenshot 2023-06-21 222644](https://github.com/anas-hamada-2000/Traffic-Tracker/assets/68608987/86f2983c-5b81-40a8-b8d7-3fbbfa0907df)

maskeler kullanarak fotoğrafın sadece bir kısımını (En doğru sinir ağı tahmininin olduğu yer) derin sinir ağa göndeririz

<img src="https://github.com/anas-hamada-2000/Traffic-Tracker/assets/68608987/67c7b681-3251-4dbd-bafa-b15c0ebe74fe" width=40% height=40%>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/anas-hamada-2000/Traffic-Tracker/assets/68608987/9b04ac69-0d59-43db-ba8b-2199382f8bc8" width=40% height=40%>

![Screenshot 2023-06-21 222925](https://github.com/anas-hamada-2000/Traffic-Tracker/assets/68608987/4798e673-2f0f-4cfc-82fa-c100a2ce931e)

Herhangi bir araba kırmızı çizgiye çok yaklaşırsa daha önce sayıp saymadığımıza bakarız, yeni ise arabayı (yani id numarasını) eski araba dizisine ekler ve o yoldaki toplam araba sayısına 1 ekleriz. Ve o framdeki çizginin rengini kırmızıdan yeşile değiştiriyoruz.

![Screenshot 2023-06-21 222256](https://github.com/anas-hamada-2000/Traffic-Tracker/assets/68608987/ba164331-d3ee-4a7f-bcd9-56fe0ed0cf39)


over-explained isimli dosyadakinde terimler çok kullanılmadan prompta direkt nasıl entity extraction yapılması gerektiğini ufak örneklerle destekleyerek
anlattık.

another_overexplained isimli dosyada ise çok fazla direkt raporlardaki hastalıklar üzerinden örnekler vermiş model için bu diğer chest ct raporlarının entity extractionlarını yaparken sorun çıkarabilir.

streamlined ise over-explained dosyasındaki prompt'u sadeleştirerek ürettiğim prompt oluyor.

karşılaştırmalarda over-explained en güvenilir sonuçları verdi gibi ancak diğer chest-ct raporları üzerinden test etmedim.

another_overexplained isimli dosyayı ise çok fazla örneklerle desteklendiği için sorun çıkarabileceğini düşündüğüm için 18 tane rastgele rapor üzerinden test ettim. 

claude opus 4.6'dan altın standart saymak için direkt prompt girip entity extraction yapmasını istedim sonrasında bu dosyalarda çıkardığımız schema mappinglerin
yüksek oranda benzediklerini tespit ettim ancak halen çok doğru değil. opus'a girdiğim promptla verdiği şemalar daha tam gibiydi gemma birkaç tane şema kaçırdı.

birbirleriyle karşılaştırdığımda over-explained iyi bir sonuç elde aralarından en kötüsü ise streamlined adlı dosyadaki prompt oldu.
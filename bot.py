from bs4 import BeautifulSoup
import urllib.request
import codecs
import jpype
 
def startJVM():
    # JVM başlat
    # Aşağıdaki adresleri java sürümünüze ve jar dosyasının bulunduğu klasöre göre değiştirin
    jpype.startJVM(jpype.getDefaultJVMPath(),
             "-Djava.class.path=zemberek-tum-2.0.jar", "-ea")
    # Türkiye Türkçesine göre çözümlemek için gerekli sınıfı hazırla
    Tr = jpype.JClass("net.zemberek.tr.yapi.TurkiyeTurkcesi")
    # tr nesnesini oluştur
    tr = Tr()
    # Zemberek sınıfını yükle
    Zemberek = jpype.JClass("net.zemberek.erisim.Zemberek")
    # zemberek nesnesini oluştur
    zemberek = Zemberek(tr)
    return zemberek

zemberek = startJVM()
  
def yorumDuzelt(yorum, zemberek):
    yorumEdit = ""
    for kelime in yorum.split():
        if zemberek.kelimeDenetle(kelime) == 0:
            turkce = []
            turkce.extend(zemberek.asciidenTurkceye(kelime))
            if len(turkce) == 0:
                oneriler = []    
                oneriler.extend(zemberek.oner(kelime))
                if len(oneriler) > 0:
                    yorumEdit = yorumEdit + " " + oneriler[0]
                else:
                    yorumEdit = yorumEdit + " " + kelime
            else:
                yorumEdit = yorumEdit + " " + turkce[0]
        else:
            yorumEdit = yorumEdit + " " + kelime
    return yorumEdit

result = u""
film_id = "144022"
site = "http://www.beyazperde.com/filmler/film-" + film_id +"/kullanici-elestirileri/"

urls = []
for i in range(0,2):
    if i == 0:
        url = site + "puani-0/"
    else:
        url = site + "puani-0/?page=" + str(i+1) 
    urls.append(url)


for url in urls:
    url_oku = urllib.request.urlopen(url)
    soup = BeautifulSoup(url_oku, 'html.parser')
     
    icerik = soup.find_all('p',attrs={'class':'review-content'})
    if(len(icerik) > 0):
        for yorum in icerik:
            result = result + film_id + "\t" + yorumDuzelt(yorum.text.strip(),zemberek) + "\t" + "0\n" 


file = codecs.open("test.tsv","a","utf-8") 
file.write(result)
file.close()

#JVM kapat
jpype.shutdownJVM()   
    
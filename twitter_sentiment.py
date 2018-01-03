import tweepy
import jpype
import re

import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


# Bu kısım sadece örnek verisetine göre Tokenizer yapısının oluşturulması için kullanılır.
# Ağın elle verilen örneklerin anlam analizini yapabilmesi için bunları tam sayı dizilerine dönüştürmesi gerekir.
# Bu kısım tam sayı dizisine dönüştürme için gerekli olan Tokenizer yapısının örnek veriseti ile oluşturulmasını sağlar.
turkish_characters = "a|b|c|ç|d|e|f|g|ğ|h|ı|i|j|k|l|m|n|o|ö|p|r|s|ş|t|u|ü|v|y|z|0-9"
data = pd.read_csv("data/data.tsv", header=0,delimiter="\t", quoting=3)
data = data[['Review','Sentiment']]
data['Review'] = data['Review'].apply(lambda x: x.lower())
data['Review'] = data['Review'].apply((lambda x: re.sub('[^'+turkish_characters+'\s]','',x)))
tokenizer = Tokenizer(split=' ',num_words=25000)
tokenizer.fit_on_texts(data['Review'].values)



def startJVM():
    # JVM başlat
    # Aşağıdaki adresleri java sürümünüze ve jar dosyasının bulunduğu klasöre göre değiştirin
    jpype.startJVM(jpype.getDefaultJVMPath(),
             "-Djava.class.path=zemberek_jar\\zemberek-tum-2.0.jar", "-ea")
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

# Gönderilen metinler Zemberek kütüphanesi fonksiyonları ile düzenlenir.
# Örnek: calismiyor -> çalışmıyor, dugun -> düğün veya söylersn -> söylersin
# Zemberek kütüphanesinin "oner" ve "asciidenTurkceye" fonksiyonları kullanılır.
# https://github.com/ahmetaa/zemberek/ Ayrıntılı bilgiler bu sayfada mevcut
def yorumDuzelt(yorum, zemberek):
    yorumEdit = ""
    for kelime in yorum.split():
        if ("et" == kelime.strip()) or ("https" in kelime.strip()):
            continue
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
    return re.sub(u'[^'+turkish_characters+'\s]','',yorumEdit.strip().lower())

# Twitter apiye bağlanmak için nesne oluşturulur.
# https://apps.twitter.com/app/new adresinden yeni bir uygulama oluşturup gerekli anahtarları elde
# edebilirsiniz.
def initTwitterAPI():    
    auth = tweepy.OAuthHandler("WhB0ZbVImPLdsr5XXTjyy0fIp", "1bwBprcnYVNVyEYyCI20OO5cugbWUlVmkQ4322acfYL1g6Fisn")
    auth.set_access_token("317070831-od04c2LJfGvVpCkyiR3VQhLF0k6sErymFYBK5WO3", "dNR3Nym9H24N3vnYbk8l3wkZDtbuH5zoE8Ro4nsMwjgZj")
    api = tweepy.API(auth)
    return api


api = initTwitterAPI()

tweets = []

# Verilen sorgu ile ilgili olan "n" adet tweet çekilir ve "yorumDuzelt" fonksiyonu ile temizlenir.
# Bu örnekte 10 tweet çekilir.
for tweet in tweepy.Cursor(api.search, q="istanbul", lang="tr").items(10):
    editedTweet = ""
    for word in tweet.text.split():
        editedTweet = editedTweet + " " + yorumDuzelt(word,zemberek)    
    tweets.append(editedTweet)

# Daha önce eğitilmiş olan model yüklenir.
model = load_model("models/sentiment_3440_model.h5")

# Verilen örnekler Tokenizer yapısı ile tam sayı dizisine dönüştürülür
# Daha sonra eğitilen modele sırayla verilerek anlam analizi sonuçları elde edilir.
# Her Cümlenin yüzde kaç olumlu ve olumsuz olduğuna dair bilgiler çıktı olarak verilir.
sequences = tokenizer.texts_to_sequences(tweets)
data = pad_sequences(sequences, maxlen=400) 
predictions = model.predict(data)
print(predictions)
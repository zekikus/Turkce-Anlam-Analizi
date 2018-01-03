import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.models import load_model
from keras.utils.np_utils import to_categorical
import re

turkish_characters = "a|b|c|ç|d|e|f|g|ğ|h|ı|i|j|k|l|m|n|o|ö|p|r|s|ş|t|u|ü|v|y|z|0-9"
data = pd.read_csv("data/data.tsv", header=0, \
                    delimiter="\t", quoting=3)

data = data[['Review','Sentiment']]

# Veri Setinin temizlenmesi.
data['Review'] = data['Review'].apply(lambda x: x.lower())
data['Review'] = data['Review'].apply((lambda x: re.sub('[^'+turkish_characters+'\s]','',x)))
    

# Veri setimizin işlenebilmesi için text verileri numaralara çevirmemiz gerekir.
# Keras bu işlem için hazır bir mekanizma sunmaktadır.
# Tokenizer sınıfı data içerisinde verilen cümleleri analiz ederek. Kelimelerin sıklıklarını hesaplar.
# Parameter: num_words = En sık geçen 25000 kelimeye odaklan. Diğerleri önemli değil
tokenizer = Tokenizer(split=' ',num_words=25000)
# Her bir kelimenin sıklığını(frekansını) hesaplar.
tokenizer.fit_on_texts(data['Review'].values)
# Tüm cümleler tam sayı dizisine dönüştürülür.
X = tokenizer.texts_to_sequences(data['Review'].values)
# Bütün metinlerimiz 400 sütundan oluşan bir dizi ile temsil edilecek.
# Çok kısa metinler 0'lar ile doldurulacak. Çok uzun metinler ise kesilecek.
X = pad_sequences(X,maxlen=400)


embed_dim = 128
lstm_out = 128
def build_model():
    model = Sequential()
    # Her bir kelimenin temsil edileceği vektör boyutu. Bu örnek için her bir kelime 128 boyutunda
    # bir vektör ile temsil edilir.
    model.add(Embedding(25000, embed_dim,input_length = X.shape[1], dropout=0.2))
    model.add(LSTM(lstm_out, dropout_U=0.2, dropout_W=0.2))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return model

# Çıktılarımızı kategorik hale getirdik. (Opsiyonel)
Y = pd.get_dummies(data['Sentiment']).values

# Verinin %80'i train, %20'si test verisi olacak şekilde ayrılır.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 2)

model = build_model()
# Oluşturulan model train verileri ile eğitilir. Yapay Sinir Ağı eğitilmeye başlar.
# nb_epoch: İterasyon sayısı
model.fit(X_train, Y_train, nb_epoch = 3, batch_size=32, verbose = 2)
# Train verileri ile model eğitildikten sonra test dataları ile doğruluk oranlarına bakılır.
score = model.evaluate(X_test, Y_test, verbose = 2)
print("score: %.2f" % (score[1]))

# Save Model
#model.save("models/sentiment_3440_model.h5")

# Load Model. Daha önce eğitilmiş olan model. Veriseti yada model parametreleri değişirse
# bu model geçersiz olur. Eğitim işlemi uzun sürdüğü için ağı bir defa eğitip oluşan modeli kaydettim.
model = load_model("models/sentiment_3440_model.h5")

my_text = ["güzel bir film tavsiye ederim",
           "hayat zor","bu filmi hiç beğenmedim",
           "daha iyi olabilir",
           "Dünyanın en aşağılık filmi.Bu filmde oynayanlardan filme çekene kadar hepsine yazıklar olsun diyorum.Kötü film demek bu filme iltifat kalır 1/10",
           "Babam bizimle hiç konuşmaz",
           "Senaryo harika değil, ama oyunculuk iyi ve sinematografi mükemmel",
           "roger Dodger bu temanın en etkileyici varyasyonlarından biridir",
           "vader akıllı, yakışıklı ve komik",
           "Gerçekten kötü, korkunç bir kitap"]

# Verilen örnekler Tokenizer yapısı ile tam sayı dizisine dönüştürülür
# Daha sonra eğitilen modele sırayla verilerek anlam analizi sonuçları elde edilir.
# Her Cümlenin yüzde kaç olumlu ve olumsuz olduğuna dair bilgiler çıktı olarak verilir.
sequences = tokenizer.texts_to_sequences(my_text)
data = pad_sequences(sequences, maxlen=400)
predictions = model.predict(data)
print(predictions)

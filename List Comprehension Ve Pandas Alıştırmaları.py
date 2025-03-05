
### List Comprehension Alıştırmaları

import pandas as pd
import numpy as np
import seaborn as sns

pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)

df = sns.load_dataset("car_crashes")
df.columns
df.info()
df.head()

## Görev 1 : List Comprehension Yapısı Kullanarak car_crashes Verisinde ki Numeric Değişkenlerin İsimlerini Büyük Harfe Çeviriniz.

["NUM_"+col.upper() if df[col].dtype != "O" else col.upper() for col in df.columns]

## Görev 2 : List Comprehension Yapısı Kullanarak car_crashes Verisinde ki İsminde "no" Barındırmayan Değişkenlerin İsimlerinin Sonuna "FLAG" yazınız.
# Notlar :
# Tüm Değişkenlerin İsmi Büyük Olmalı
# Tek Bir List Comp İle Yapınız.

[col.upper()+ "_FLAG" if "no" not in col  else col.upper() for col in df.columns]

## Görev 3 : List Comprehension Yapısı Kullanarak Aşağıda Verilen Değişken İsimlerinden Farklı Olan Değişkenlerin İsimlerini Seçiniz Ve Yeni Bir Dataframe Oluşturunuz.

## og_list=["abbrev","no_previous"]

# Notlar :
# Önce Yukarıdaki Listeye Göre List Comprehension Kullanarak new_cols Adında Yeni Bir Liste Oluşturunuz.
# Sonra df[new_cols] İle Bu Değişkenleri Seçerek Yeni Bir Df Oluşturunuz. Adını new_df Olarak İsimlendiriniz.

og_list=["abbrev","no_previous"]
new_cols=[col for col in df.columns if col not in og_list]
new_df=df[new_cols]
new_df.head()



### Pandas Alıştırmaları

import numpy as np
import pandas as pd
import seaborn as sns

pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)
pd.set_option("display.width",1000)

## Görev 1 : Seaborn Kütüphanesi İçerisinde Titanic Veri Setini Tanımlayınız.

df = sns.load_dataset("titanic")
df.head()
df.info()
df.shape

## Görev 2 : Yukarıda Tanımlanan Titanic Veri Setindeki Kadın Ve Erkek Yolcuların SayısınI Bulunuz.

df["sex"].value_counts()

## Görev 3 : Her Bir Sütuna Ait Unique Değerlerinin Sayısını Bulunuz.

df.nunique()

## Görev 4 : Pclass Değişkeninin Unique Değerlerini Bulunuz.

df["pclass"].unique()

## Görev 5 : Pclass ve Parch Değişkenlerinin Unique Değerlerinin Sayısını Bulunuz.

df[["pclass","parch"]].nunique()

## Görev 6 : Embarked Değişkeninin Tipini Kontrol Ediniz. Tipini Category Olarak Değiştiriniz.Tekar Tipini Kontrol Ediniz.

df["embarked"].dtypes

df["embarked"]= df["embarked"].astype("category")

df.info()

## Görev 7 : Embarked Değeri C Olanların Tüm Bilgilerini Gösteriniz.

df[df["embarked"]== "C"].head(10)

## Görev 8 : Embarked Değeri S Olmayanların Tüm Bilgilerini Gösteriniz.

df[df["embarked"]!="S"].head(10)

## Görev 9 : Yaşı 30 dan Küçük ve Kadın Olan Yolcuların TÜM Bilgilerini Gösteriniz.
df[(df["age"]<30) & (df["sex"] == "female")].head(10)

## Görev 10 : Fare'i 500'den Büyük Veya Yaşı 70'den Büyük Olan Tüm Yolcuların Bilgilerini Gösteriniz.

df[(df["fare"] >500) | (df["age"] >70)].head()

## Görev 11 : Her Bir Değişkendeki Boş Değerlerin Toplamını Bulunuz.

df.isnull().sum()

## Görev 12 : Who Değişkenini Dataframe'den Düşürünüz.

df.drop("who",axis=1,inplace=True)

## Görev 13 : Deck Değişkenindeki Boş Değerleri Deck Değişkeninin En Çok Tekrar Eden Değeri (mode) ile doldurunuz.

type(df["deck"].mode())
df["deck"].mode()[0]
df["deck"].fillna(df["deck"].mode()[0],inplace=True)
df["deck"].isnull().sum()

## Görev 14 : Age Değişkenin'deki Boş Değerleri Age Değişkenin Medyanı İle Doldurun

df["age"].fillna(df["age"].median(),inplace=True)
df["age"].isnull().sum()

df.isnull().sum()

## Görev 15 : Survived Değişkeninin Pclass  ve Cinsiyet Değişkenleri Kırılımında Sum,Count,Mean Değelerini Bulunuz.

df.groupby(["pclass","sex"]).agg({"survived": ["sum","count","mean"]})

## Görev 16 : 30 Yaşın Altında Olanlar 1 , 30'a Eşit Ve Üstünde Olanlara 0 Vericek Bir Fonksiyon Yazınız.
# Yazdığınız Fonksiyonu Kullanarak Titanic Veri Setinde age_flag Adında Bir Değişken Oluşturunuz.(Apply Ve Lambda Yapılarını Kullanınız)

def age_30(age):
    if age<30 :
        return 1
    else :
        return 0

df["age_flag"] = df["age"].apply(lambda x : age_30(x))

df["age_flag"] = df["age"].apply(lambda x : 1 if x<30 else 0)

## Görev 17 : Seaborn Kütüphanesi İçerisinden Tips Veri Setini Tanımlayınız.

df=sns.load_dataset("tips")

df.head()
df.shape

## Görev 18 : Time Değişkeninin Kategorilerine (Dinner,Lunch) Göre total_bill Değerlerinin Toplamını,min,max ve Ortalamasını Bulunuz.

df.groupby("time").agg({"total_bill": ["sum","min","mean","max"]})

## Görev 19 : Günlere Ve Time'a Göre  Total_bill Değerlerinin Toplamını,min,max ve Ortalamasını Bulunuz.

df.groupby(["day","time"]).agg({"total_bill": ["sum","min","mean","max"]})

## Görev  20 : Lunch Zamanına Ve Kadın Müşterilere Ait Total_bill ve Tip Değerlerinin day'a Göre Toplamını , min,max Ve Ortalamasını Bulunuz.

df[(df["time"] == "Lunch") & (df["sex"] == "Female")].groupby("day").agg({"total_bill": ["sum","min","mean","max"],
                                                                          "tip": ["sum","min","mean","max"]})

## Görev  21 : Size'ı 3'ten Küçük, Total_bill'i 10'dan Büyük Olanların Siparişlerinin Ortalaması Nedir?

df.loc[(df["size"]<3 ) & (df["total_bill"]> 10 ) , "total_bill"].mean()

## Görev 22 : Total_bill_tip_sum Adında Yeni  Bir Değişken Oluşturun. Her Bir Müşterinin Ödediği Total_bill Ve Tip'in Toplamını Versin

df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]

df.head(3)

## Görev  23 : Total_bill_tip_sum Değişkenine Göre Büyükten Küçüğe Sıralayınız Ve İlk 30 Kişiyi Yeni Bir Dataframe'e Aktarınız.

new_df =df.sort_values("total_bill_tip_sum", ascending=False)[:30]
new_df.shape




# Kural Tabanlı Sınıflandırma İle Potansiyel Müşteri Getirisi Hesaplama

# İş Problemi

# Gezinomi yaptığı satışların bazı özelliklerini kullanarak seviye tabanlı (level based) yeni satış tanımları
# oluşturmak ve bu yeni satış tanımlarına göre segmentler oluşturup bu segmentlere göre yeni gelebilecek müşterilerin şirkete
# ortalama ne kadar kazandırabileceğini tahmin etmek istemektedir.
# Örneğin Antalya'dan Herşey Dahil bir otele yoğun bir dönemde gitmek isteyen bir müşterinin ortalama ne kadar kazandırabileceğini
# belirlemek isteniyor.

## PROJE GÖREVLERİ
#######################

## Görev 1 : Aşağıdaki Soruları Yanıtlayınız.

# Soru 1 : miuul-gezinomi.xlxs dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.

import pandas as pd
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

df= pd.read_excel("miuul_gezinomi.xlsx")

print(df.head())
print(df.shape)
print(df.info())

# Soru 2 : Kaç unique şehir vardır ? Frekansları nedir ?

df["SaleCityName"].nunique()
df["SaleCityName"].value_counts()
# Soru 3 : Kaç unique Concept vardır ?

df["ConceptName"].nunique()

# Soru 4 : Hangi Concept'dan kaçar tane satış gerçekleşmiştir?

df["ConceptName"].value_counts()

# Soru 5 : Şehirlere göre satışlarından toplam ne kadar kazanılmıştır?

df.groupby("SaleCityName").agg({"Price"  : "sum"})

# Soru 6  : Concept türlerine göre ne kadar kazanılmıştır?

df.groupby("ConceptName").agg({"Price"  : "sum"})

# Soru 7 : Şehirlere göre PRICE ortalamaları nedir ?

df.groupby("SaleCityName").agg({"Price"  : "mean"})

# Soru 8 : Conceptlere göre PRICE ortalamaları nedir ?

df.groupby("ConceptName").agg({"Price"  : "mean"})

# Soru 9 : Şehir-Concept kırılımında PRICE ortalamaları nedir ?

df.groupby(["SaleCityName","ConceptName"]).agg({"Price" :"mean"})

# Görev 2 : satış_checkin_day_dif dedğişkeninni EB_Score adında yani yeni bir kategorik değişkene çeviriniz

bins=[-1,7,30,90,df["SaleCheckInDayDiff"].max()]
labels = ["Last Minuters","Potential Planners","Planners","Early Bookers"]

df["EB_Score"]=pd.cut(df["SaleCheckInDayDiff"],bins,labels=labels)
df.head(50).to_excel("eb_scorew.xlsx",index=False)

# gÖREV 3 : Şehir,Concepti["EB_Score,Sezon,CInday] kırılımında ücret ortalamalarının ve frekanslarına bakınız.

# Şehir-Concept-EB Score kırılımında ücret ortalamaları

df.groupby(["SaleCityName","ConceptName","EB_Score"]).agg({"Price": ["mean","count"]})

# Şehir-Concept-Sezon kırılımında ücret ortalamaları

df.groupby(["SaleCityName","ConceptName","Seasons"]).agg({"Price": ["mean","count"]})

# Şehir-Concept-CInday kırılımında ücret ortalamaları

df.groupby(["SaleCityName","ConceptName","CInDay"]).agg({"Price": ["mean","count"]})

# Görev 4 : City-Concept-Season kırılımın çıktısını PRICE'a göre sıralayınız.

# Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde
# Çıktıyı agg_df olarak kaydediniz.

agg_df = df.groupby(["SaleCityName","ConceptName","Seasons"]).agg({"Price": "mean"}).sort_values("Price",ascending=False)

agg_df.head(20)

# Görev 5 : Indekste yer alan isimleri değişken ismine çeviriniz.

# Üçüncü sorunun çıktısında yer alan PRICE dışındaki ve değişkenler index isimleridir.
# Bu isimleri değişken isimlerine çeviriniz.
# İpucu : reset_index
agg_df.reset_index(inplace=True)
agg_df.head(20)

# Görev 6 : Yeni level based satışları tanımlayınız ve veri setine değişken olarak ekleyiniz.

# sales_level_based_ adında bir değişken tanımyanınız ve veri setine bu değişkeni atayınız.

agg_df["sales_level_based"] = agg_df[["SaleCityName","ConceptName","Seasons"]].agg(lambda x:"_".join(x).upper(),axis=1 )

# Görev 7 : Personları segmentlere ayırınız.

# Prıce'a göre segmentlere ayırınız.
# Segmentleri "SEGMENT" isimlendirmesi ile agg_df'e ekleyiniz.
# Segmentleri betimleyiniz.

agg_df["SEGMENT"]= pd.qcut(agg_df["Price"], 4, labels=["D","C","B","A"])

agg_df.head(15)

agg_df.groupby("SEGMENT").agg({"Price": ["min","mean","max"]})

# Görev 8 : Oluşan son df'i price değişkenine göre sıralayınız.
# "ANTALYA_HERSEY_DAHIL_HIGH" hangi segmenttedir ve ne kadar ücret beklenmektedir.

agg_df.sort_values(by="Price")

new_user = "ANTALYA_HERSEY_DAHIL_HIGH"

agg_df[agg_df["sales_level_based "] == new_user]
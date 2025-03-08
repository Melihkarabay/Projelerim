### İŞ PROBLEMİ
#Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin
#edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir. Modeli
#geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarını
#gerçekleştirmeniz beklenmektedir

# VERİ SETİ HİKAYESİ

#Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin parçasıdır. ABD'deki
#Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan Pima Indian kadınları üzerinde
#yapılan diyabet araştırması için kullanılan verilerdir.
#Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,roc_auc_score
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
pd.set_option("display.float_format",lambda x:"%.3f" %x)

df = pd.read_csv('diabetes.csv')

# Görev 1 :KEŞİFÇİ VERİ ANALİZİ

def check_df(dataframe, head=5):
    print("######## SHAPE #########")
    print(dataframe.shape)
    print("######## TYPE #########")
    print(dataframe.dtypes)
    print("######## HEAD #########")
    print(dataframe.head(head))
    print("######## TAİL #########")
    print(dataframe.tail(head))
    print("######## NA #########")
    print(dataframe.isnull().sum())
    print("######## QUANTİLES #########")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    ""

check_df(df)

### GÖREV 2 : NUMERİC VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI

def grab_col_names(dataframe,cat_th=10, car_th=20):

    cat_cols = [col for col in df.columns if df[col].dtypes in ["object", "category", "bool"]]

    num_but_cat = [col for col in df.columns if (df[col].nunique() < 10) and df[col].dtypes in ["int64", "float64"]]

    cat_but_car = [col for col in df.columns if (df[col].nunique() > 20) and df[col].dtypes in ["object", "category"]]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]

    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables :  {dataframe.shape[1]}")
    print(f"cat_cols : {len(cat_cols)}")
    print(f"num_cols : {len(num_cols)}")
    print(f"cat_but_car : {len(cat_but_car)}")
    print(f"num_but_cat : {len(num_but_cat)}")

    return cat_cols,num_cols,cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

### GÖREV 3 : KATEGORİK DEĞİŞKENLERİN ANALİZİ

def cat_summary(dataframe,col_name,plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio" : 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("########################################################################")

    if plot :
        sns.countplot(x=dataframe[col_name],data=dataframe)
        plt.show()

cat_summary(df,"Outcome",plot=True)

### GÖREV 4 : NUMERİC DEĞİŞKENLERİN ANALİZİ

def num_summary (dataframe,numerical_col,plot=False):
    quantiles=[0,0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    print("#######################################################################")

    if plot :
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df,col,plot=True)

### GÖREV 5 : NUMERIC DEĞİŞKENLERİN TARRGET'A GÖRE ANALİZİ

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}))

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)


### KORELASYON ANALİZİ

df.corr()

# Korelasyon Matrisi

f,ax=plt.subplots(figsize=[18,13])
sns.heatmap(df.corr(),annot=True,fmt=".2f",ax=ax,cmap="magma"),
ax.set_title("Corelasyon Matrisi", fontsize=20)
plt.show()


### GÖREV 6 : BASE MODEL KURULUMU


y = df["Outcome"]
X = df.drop("Outcome", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_test, y_pred), 2)}")
print(f"Recall: {round(recall_score(y_test, y_pred), 3)}")
print(f"Precision: {round(precision_score(y_test, y_pred), 2)}")
print(f"F1: {round(f1_score(y_test, y_pred), 2)}")
print(f"AUC: {round(roc_auc_score(y_test, y_pred), 2)}")

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")


plot_importance(rf_model,X)


### FEATURE ENGİNEERİNG

### EKSİK GÖZLEM ANALİZİ

zero_columns=[col for col in df.columns if (df[col].min()== 0 and col not in ["Pregnancies","Outcome"])]

zero_columns

for col in zero_columns:
    df[col] = np.where(df[col] == 0 , np.nan,df[col])

df.isnull().sum()
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

    na_columns = missing_values_table(df,na_name=True)

### EKSİK DEĞERLERİN BAĞIMLI DEĞİŞKEN İLE İLİŞKİLERİNİN İNCELENMESİ

    def missing_values_table(df, target, na_columns):
        temp_df = df.copy()
        for col in na_columns:
            temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)

        na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

        for col in na_flags:
            result = pd.DataFrame({
                "TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                "Count": temp_df.groupby(col)[target].count()
            })
            print(result, end="\n\n\n")

    missing_values_table(df,"Outcome",na_columns)

#### EKSİK DEĞER DOLDURMA

for col in df.columns :
    df.loc[df[col].isnull(),col] = df[col].median()

df.isnull().sum()

### AYKIRI DEĞER ANALİZİ

def thresholds_outlier(dataframe,col_name,Q1=0.05,Q3=0.95):

 quartile1=dataframe[col_name].quantile(Q1)
 quartile3=dataframe[col_name].quantile(Q3)
 interquartile_range= Q3-Q1
 up_limit = quartile3 + 1.5 * interquartile_range
 low_limit = quartile1 - 1.5 * interquartile_range
 return low_limit,up_limit

def check_outlier(dataframe,col_name):
    low_limit,up_limit=thresholds_outlier(dataframe,col_name)
    if dataframe[(dataframe[col_name]> up_limit )  | (dataframe[col_name]< low_limit)].any(axis=None):

        return True
    else:
        return False

def replace_with_thresholds(dataframe,variable):
    low_limit,up_limit = thresholds_outlier(dataframe,variable,Q1=0.05,Q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit),variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit),variable] = up_limit


### AYKIRI DEĞER ANALİZİ VE BASKILAMA İŞLEMİ


for col in df.columns:
    print(col),check_outlier(df,col)
    if check_outlier(df,col):
        replace_with_thresholds(df,col)

for col in df.columns:
    print(col,check_outlier(df,col))


### ÖZELLİK ÇIKARIMI

# Yaş değişkenini kategorilere ayırıp yeni yaş değişkeni oluşturulması

df.loc[(df["Age"] >= 21) & (df["Age"] < 50 ), "NEW_AGE_CAT"] = "mature"
df.loc[(df["Age"] >= 50), "NEW_AGE_CAT" ] = "senior"

## BMI 18,5 aşağısı underweight, 18,5 ile 24.9 arası normal, 24.9 ile 29.9 arası Overweight ve 30 üstü obez

df["NEW_BMI"]= pd.cut(x=df["BMI"],bins=[0,18.5,24.9,29.9,100],
                      labels=["Underweight","Healthy","Overweight","Obese"])
### GLİKOZ degerini kategorik değişkene çevirme

df["NEW_GLUCOSE"]= pd.cut(x=df["Glucose"],bins=[0,140,200,300],
                      labels=["Normal","Prediabets","Diabets"])

### Yaş Ve Beden Kitle İndeksini bir arada düşünerek kategorik değişken üretme

df.loc[(df["BMI"] < 18.5)& ((df["Age"] >= 21) & (df["Age"] < 50)),"NEW_AGE_BMI_NOM"]= "underweightmature"

df.loc[(df["BMI"] > 50) & (df["Age"]> 50),"NEW_AGE_BMI_NOM"] = "underweightsenior"

df.loc[((df["BMI"] >18.5) & (df["BMI"] < 25)) & (
    (df["Age"] >=21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "healtymature"

df.loc[((df["BMI"] >18.5) & (df["BMI"] < 25)) & (df["Age"] >=50), "NEW_AGE_BMI_NOM"] = "healtysenior"

df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (
    (df["Age"] >=21) & (df["Age"] < 50 )),"NEW_AGE_BMI_NOM"]= "overweightmature"

df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] < 50 ),"NEW_AGE_BMI_NOM"]= "overweightsenior"

df.loc[(df["BMI"] > 18.5) & ((df["BMI"] >= 21 ) & (df["Age"] < 50 )), "NEW_AGE_BMI_NOM" ] = "obesemature"

df.loc[(df["BMI"] > 18.5) & (df["Age"] <= 50), "NEW_AGE_BMI_NOM"] = "obesesenior"

df.head()


### Yaş ve Glikoz Değerlerini bir arada düşünerek kategorik değişken oluşturma

df.loc[(df["Glucose"] < 70) & ((df["Age"] >=21) & (df["Age"] > 50)) , "NEW_AGE_GLUCOSE_NOM"]= "lowmature"
df.loc[(df["Glucose"] < 70) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"]= "lowsenior"

df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100 )) & (
        (df["Age"] >=21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"]= "normalmature"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100 )) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "normalsenior"

df.loc[((df["Glucose"] >= 100) & (df["Glucose"] < 125 )) & (
        (df["Age"] >=21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"]= "hiddenmature"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] < 125 )) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "hiddensenior"

df.loc[(df["Glucose"] >= 125)  & (
        (df["Age"] > 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"]= "hightmature"
df.loc[(df["Glucose"] > 125) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "hightsenior"


### İnsulin değeri ile kategorik değişken türetmek

def set_insulin(dataframe,col_name="Insulin"):
    if 16 <= dataframe[col_name] <=166:
        return "Normal"
    else :
        return "Abnormal"


df["NEW_INSULIN_SCORE"] = df.apply(set_insulin,axis=1)

df["NEW_GLUCOSE*INSULIN"] = df["Glucose"] * df["Insulin"]
df["NEW_GLUCOSE_PREGNANCIES"] = df["Glucose"] * df["Pregnancies"]


### Kolon isimlerinin büyütülmesi

df.columns= [col.upper() for col in df.columns]


df.shape

df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)


### Label Encoding

def label_encoder(dataframe,binary_col):
    label_encoder= LabelEncoder()
    dataframe[binary_col] = label_encoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols=[col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2 ]
binary_cols

for col in binary_cols:
    df = label_encoder(df,col)

### One-Hot Encoding İşlemi
# cat_cols listesinin güncellenme işlemi

# Kategorik sütunlar listesini filtrelemek
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Outcome"]]

# Birinci adımı kontrol etmek için yazdırma
print(cat_cols)

# One-hot encoding fonksiyonu
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

# One-hot encoding uygulama
df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()

df.info()

df.isnull().sum()

cat_cols

#### Standartlaştırma

num_cols

scaler=StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


### Modelleme

y = df["OUTCOME"]
x= df.drop("OUTCOME", axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train,y_train)
y_pred = rf_model.predict(X_test)


print(f"Accuracy: {round(accuracy_score(y_pred,y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test), 3)}")
print((f"Precision : {round(precision_score(y_pred,y_test),  2)}"))
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc : {round(roc_auc_score(y_pred,y_test), 2 )}")


def plot_importance(model,features, num=len(X), save=False):
    feature_imp= pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    print(feature_imp.sort_values("Value", ascending=False))
    plt.figure(figsize=(10,10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])

    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.jpg")

plot_importance(rf_model,X)



# ============================================================
#  Yapay Zeka Temelleri - Vize/Final Ödevi
#  Konu   : Şeker Hastalığı (Diabetes) Teşhisi
#  Problem : Sınıflandırma (Diyabetli mi? Evet:1 / Hayır:0)
#  Dataset : Pima Indians Diabetes (Kaggle)
# ============================================================

# Hazır araçları içeri alıyoruz
# pandas  → tablo işlemleri (Excel gibi çalışır)
# numpy   → sayısal hesaplamalar
# matplotlib / seaborn → grafik çizme
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split   # veriyi ikiye bölmek için
from sklearn.preprocessing import StandardScaler       # değerleri aynı ölçeğe getirmek için
from sklearn.linear_model import LogisticRegression    # 1. model
from sklearn.tree import DecisionTreeClassifier        # 2. model
from sklearn.ensemble import RandomForestClassifier    # 3. model
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix)

sns.set_theme(style="whitegrid", palette="muted")


# ============================================================
# BÖLÜM 1 – VERİ SETİ TANITIMI
# ============================================================

# CSV dosyasını okuyup tabloya çeviriyoruz
# df artık bizim veri tablomuzdur
df = pd.read_csv("diabetes.csv")

print("=" * 55)
print("  VERİ SETİ GENEL BİLGİLERİ")
print("=" * 55)

# Tablonun kaç satır ve sütundan oluştuğunu göster
print(f"Satır sayısı  : {df.shape[0]}")
print(f"Sütun sayısı  : {df.shape[1]}")

# İlk 5 satırı ekrana yazdır, veriye göz at
print("\nİlk 5 satır:")
print(df.head())

# Her sütunun veri tipini göster (sayı mı, metin mi?)
print("\nSütunlar ve veri tipleri:")
print(df.dtypes)

# Kaç kişi sağlıklı (0), kaç kişi diyabetli (1)?
print("\nHedef sınıf dağılımı (0=Sağlıklı, 1=Diyabetli):")
print(df["Outcome"].value_counts())


# ============================================================
# BÖLÜM 2 – VERİ ANALİZİ VE ÖN İŞLEME (EDA)
# ============================================================

# --- Eksik Veri Analizi ---
# Kan şekeri, kan basıncı gibi değerler 0 olamaz
# Bu sütunlardaki 0'lar aslında eksik veriyi temsil ediyor
sifir_olamaz = ["Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI"]

print("\nMantıksal olarak 0 olamayacak sütunlardaki sıfır sayıları:")
print((df[sifir_olamaz] == 0).sum())

# 0 değerlerini NaN (boş) yap, sonra o sütunun ortancasıyla doldur
df[sifir_olamaz] = df[sifir_olamaz].replace(0, np.nan)
df[sifir_olamaz] = df[sifir_olamaz].fillna(df[sifir_olamaz].median())
print("\nEksik değerler ortanca (medyan) ile dolduruldu ✅")

# --- Temel İstatistikler ---
# Ortalama, min, max gibi özet bilgileri göster
print("\nTemel istatistikler:")
print(df.describe().round(2))

# --- Aykırı Değer Analizi (IQR Yöntemi) ---
# Çok uç değerleri tespit et (çok yüksek veya çok düşük)
print("\nAykırı değer sayıları (IQR yöntemi):")
for col in df.columns[:-1]:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    aykiri = ((df[col] < Q1 - 1.5*IQR) |
              (df[col] > Q3 + 1.5*IQR)).sum()
    print(f"  {col:20s}: {aykiri} aykırı değer")

# --- Grafik 1: Hedef Sınıf Dağılımı ---
# Kaç kişi sağlıklı, kaç kişi diyabetli? Pasta grafikle göster
fig, ax = plt.subplots()
df["Outcome"].value_counts().plot.pie(
    labels=["Sağlıklı (0)", "Diyabetli (1)"],
    autopct="%1.1f%%", colors=["#4CAF50", "#F44336"],
    startangle=90, ax=ax
)
ax.set_ylabel("")
ax.set_title("Hedef Sınıf Dağılımı")
plt.tight_layout()
plt.savefig("grafik1_sinif_dagilimi.png", dpi=150)
plt.show()

# --- Grafik 2: Özellik Histogramları ---
# Her sütunun değer dağılımını göster
df.hist(bins=20, figsize=(12, 8), color="#5C85D6", edgecolor="white")
plt.suptitle("Özelliklerin Dağılımı", fontsize=14)
plt.tight_layout()
plt.savefig("grafik2_dagilimlar.png", dpi=150)
plt.show()

# --- Grafik 3: Korelasyon Isı Haritası ---
# Hangi özellikler birbirini etkiliyor? Renk ne kadar koyu → ilişki o kadar güçlü
plt.figure(figsize=(9, 7))
sns.heatmap(df.corr(), annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5)
plt.title("Korelasyon Isı Haritası")
plt.tight_layout()
plt.savefig("grafik3_korelasyon.png", dpi=150)
plt.show()

# --- Grafik 4: Kan Şekeri ve VKİ İlişkisi ---
# Diyabetliler ile sağlıklıları nokta grafikle karşılaştır
plt.figure()
sns.scatterplot(data=df, x="Glucose", y="BMI",
                hue="Outcome", palette={0: "#4CAF50", 1: "#F44336"},
                alpha=0.7)
plt.title("Kan Şekeri - Vücut Kitle İndeksi İlişkisi")
plt.legend(title="Durum", labels=["Sağlıklı", "Diyabetli"])
plt.tight_layout()
plt.savefig("grafik4_scatter.png", dpi=150)
plt.show()


# ============================================================
# BÖLÜM 3 – MODEL KURMA
# ============================================================

# X → modelin bakacağı özellikler (kan şekeri, yaş vs.)
# y → doğru cevaplar (0: sağlıklı, 1: diyabetli)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Veriyi %80 eğitim / %20 test olarak ikiye böl
# Sınava girmeden önce 80 soruyla çalış, 20 soruyla sınava gir gibi düşün
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Değerleri aynı ölçeğe getir (Lojistik Regresyon için gerekli)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# 3 farklı algoritma tanımlıyoruz
modeller = {
    "Lojistik Regresyon": LogisticRegression(max_iter=1000, random_state=42),
    "Karar Ağacı"       : DecisionTreeClassifier(max_depth=5, random_state=42),
    "Rastgele Orman"    : RandomForestClassifier(n_estimators=100, random_state=42),
}

sonuclar = {}

for isim, model in modeller.items():
    # Lojistik Regresyon ölçeklenmiş veri kullanır, diğerleri kullanmaz
    if isim == "Lojistik Regresyon":
        model.fit(X_train_s, y_train)      # modeli eğit
        y_pred = model.predict(X_test_s)   # test verisini tahmin et
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Sonuçları kaydet
    sonuclar[isim] = {
        "Doğruluk"   : accuracy_score(y_test, y_pred),
        "Hassasiyet" : precision_score(y_test, y_pred),
        "Duyarlılık" : recall_score(y_test, y_pred),
        "F1"         : f1_score(y_test, y_pred),
        "y_pred"     : y_pred,
    }


# ============================================================
# BÖLÜM 4 – MODEL DEĞERLENDİRME
# ============================================================

# Tüm modellerin skorlarını yan yana yazdır
print("\n" + "=" * 60)
print("  MODEL KARŞILAŞTIRMASI")
print("=" * 60)
print(f"{'Model':<22} {'Doğruluk':>9} {'Hassasiyet':>11} {'Duyarlılık':>11} {'F1':>8}")
print("-" * 60)
for isim, m in sonuclar.items():
    print(f"{isim:<22} {m['Doğruluk']:>9.4f} {m['Hassasiyet']:>11.4f} "
          f"{m['Duyarlılık']:>11.4f} {m['F1']:>8.4f}")

# En yüksek F1 skoruna sahip modeli seç
en_iyi = max(sonuclar, key=lambda k: sonuclar[k]["F1"])
print(f"\n✅ En iyi model (F1 skoruna göre): {en_iyi}")

# --- Confusion Matrix ---
# Modelin neyi doğru, neyi yanlış tahmin ettiğini göster
# Köşegen → doğru tahminler | Köşegen dışı → yanlış tahminler
cm = confusion_matrix(y_test, sonuclar[en_iyi]["y_pred"])
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Sağlıklı", "Diyabetli"],
            yticklabels=["Sağlıklı", "Diyabetli"])
plt.title(f"Karışıklık Matrisi – {en_iyi}")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek Değer")
plt.tight_layout()
plt.savefig("grafik5_confusion_matrix.png", dpi=150)
plt.show()

# --- Model Karşılaştırma Grafiği ---
# Tüm modellerin skorlarını çubuk grafik olarak göster
metrik_df = pd.DataFrame(
    {k: {m: v for m, v in sonuclar[k].items() if m != "y_pred"}
     for k in sonuclar}
).T

metrik_df[["Doğruluk", "Hassasiyet", "Duyarlılık", "F1"]].plot(
    kind="bar", figsize=(9, 5), edgecolor="white"
)
plt.title("Model Performans Karşılaştırması")
plt.ylabel("Skor")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("grafik6_model_karsilastirma.png", dpi=150)
plt.show()

# --- Özellik Önem Sıralaması (Rastgele Orman) ---
# Hangi özellik diyabet tahmininde en çok belirleyici?
rf_model = modeller["Rastgele Orman"]
onem = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(7, 5))
onem.plot(kind="barh", color="#5C85D6")
plt.title("Rastgele Orman – Özellik Önem Sıralaması")
plt.xlabel("Önem Skoru")
plt.tight_layout()
plt.savefig("grafik7_ozellik_onem.png", dpi=150)
plt.show()

print("\n✅ Tüm grafikler kaydedildi.")
print("✅ Proje tamamlandı!")

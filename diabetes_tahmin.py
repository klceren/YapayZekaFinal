# ============================================================
#  Yapay Zeka Temelleri - Vize/Final Ödevi
#  Konu   : Şeker Hastalığı (Diabetes) Teşhisi
#  Problem: Sınıflandırma
#  Dataset: Pima Indians Diabetes (Kaggle)
#  Algoritma: Lojistik Regresyon + Karar Ağacı + Random Forest
# ============================================================

# ── Kütüphaneler ────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)

# Grafik stili
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (8, 5)


# ============================================================
# BÖLÜM 1 – VERİ SETİ TANITIMI
# ============================================================

# Veri setini yükle (CSV dosyası proje klasöründe olmalı)
# Kaggle linki: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
df = pd.read_csv("diabetes.csv")

print("=" * 55)
print("  VERİ SETİ GENEL BİLGİLERİ")
print("=" * 55)
print(f"Satır sayısı  : {df.shape[0]}")
print(f"Sütun sayısı  : {df.shape[1]}")
print("\nİlk 5 satır:")
print(df.head())

print("\nSütunlar ve veri tipleri:")
print(df.dtypes)

print("\nHedef sınıf dağılımı (0=Sağlıklı, 1=Diyabetli):")
print(df["Outcome"].value_counts())


# ============================================================
# BÖLÜM 2 – EDA (KEŞİFSEL VERİ ANALİZİ)
# ============================================================

# --- 2.1 Eksik Veri Analizi ---
# Bazı sütunlarda 0 değeri gerçekte eksik veriyi temsil eder
sifir_olamaz = ["Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI"]
print("\nMantıksal olarak 0 olamayacak sütunlardaki sıfır sayıları:")
print((df[sifir_olamaz] == 0).sum())

# 0'ları NaN ile değiştir, ardından medyan ile doldur
df[sifir_olamaz] = df[sifir_olamaz].replace(0, np.nan)
df[sifir_olamaz] = df[sifir_olamaz].fillna(df[sifir_olamaz].median())
print("\nEksik değerler medyan ile dolduruldu.")

# --- 2.2 İstatistiksel Özet ---
print("\nTemel istatistikler:")
print(df.describe().round(2))

# --- 2.3 Aykırı Değer Analizi (IQR yöntemi) ---
print("\nAykırı değer sayıları (IQR yöntemi):")
for col in df.columns[:-1]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    aykiri = ((df[col] < Q1 - 1.5 * IQR) |
              (df[col] > Q3 + 1.5 * IQR)).sum()
    print(f"  {col:20s}: {aykiri} aykırı değer")

# --- 2.4 Görselleştirmeler ---

# Grafik 1 – Hedef sınıf dağılımı (pasta grafik)
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

# Grafik 2 – Özellik histogramları
df.hist(bins=20, figsize=(12, 8), color="#5C85D6", edgecolor="white")
plt.suptitle("Özelliklerin Dağılımı", fontsize=14)
plt.tight_layout()
plt.savefig("grafik2_dagilimlar.png", dpi=150)
plt.show()

# Grafik 3 – Korelasyon ısı haritası
plt.figure(figsize=(9, 7))
sns.heatmap(df.corr(), annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5)
plt.title("Korelasyon Isı Haritası")
plt.tight_layout()
plt.savefig("grafik3_korelasyon.png", dpi=150)
plt.show()

# Grafik 4 – Glucose vs BMI (sınıfa göre renkli)
plt.figure()
sns.scatterplot(data=df, x="Glucose", y="BMI",
                hue="Outcome", palette={0: "#4CAF50", 1: "#F44336"},
                alpha=0.7)
plt.title("Glucose - BMI İlişkisi")
plt.legend(title="Durum", labels=["Sağlıklı", "Diyabetli"])
plt.tight_layout()
plt.savefig("grafik4_scatter.png", dpi=150)
plt.show()


# ============================================================
# BÖLÜM 3 – MODEL KURMA
# ============================================================

# Özellikler ve hedef değişken
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Eğitim / Test bölme (%80 - %20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Ölçeklendirme (Lojistik Regresyon için gerekli)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# --- Modeller ---
modeller = {
    "Lojistik Regresyon": LogisticRegression(max_iter=1000, random_state=42),
    "Karar Ağacı"       : DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest"     : RandomForestClassifier(n_estimators=100, random_state=42),
}

sonuclar = {}

for isim, model in modeller.items():
    # Lojistik Regresyon ölçeklenmiş veri kullanır
    if isim == "Lojistik Regresyon":
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    sonuclar[isim] = {
        "Accuracy" : accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall"   : recall_score(y_test, y_pred),
        "F1"       : f1_score(y_test, y_pred),
        "y_pred"   : y_pred,
    }


# ============================================================
# BÖLÜM 4 – MODEL DEĞERLENDİRME
# ============================================================

print("\n" + "=" * 55)
print("  MODEL KARŞILAŞTIRMASI")
print("=" * 55)
print(f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print("-" * 55)
for isim, m in sonuclar.items():
    print(f"{isim:<22} {m['Accuracy']:>9.4f} {m['Precision']:>10.4f} "
          f"{m['Recall']:>8.4f} {m['F1']:>8.4f}")

# En iyi modeli bul
en_iyi_isim = max(sonuclar, key=lambda k: sonuclar[k]["F1"])
print(f"\n✅ En iyi model (F1 skoruna göre): {en_iyi_isim}")

# Confusion matrix – en iyi model
cm = confusion_matrix(y_test, sonuclar[en_iyi_isim]["y_pred"])
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Sağlıklı", "Diyabetli"],
            yticklabels=["Sağlıklı", "Diyabetli"])
plt.title(f"Confusion Matrix – {en_iyi_isim}")
plt.xlabel("Tahmin"); plt.ylabel("Gerçek")
plt.tight_layout()
plt.savefig("grafik5_confusion_matrix.png", dpi=150)
plt.show()

# Metrik karşılaştırma çubuğu
metrik_df = pd.DataFrame(
    {k: {m: v for m, v in sonuclar[k].items() if m != "y_pred"}
     for k in sonuclar}
).T

metrik_df[["Accuracy", "Precision", "Recall", "F1"]].plot(
    kind="bar", figsize=(9, 5), edgecolor="white"
)
plt.title("Model Performans Karşılaştırması")
plt.ylabel("Skor"); plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("grafik6_model_karsilastirma.png", dpi=150)
plt.show()

# Random Forest özellik önemleri
rf_model = modeller["Random Forest"]
onem = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(7, 5))
onem.plot(kind="barh", color="#5C85D6")
plt.title("Random Forest – Özellik Önemleri")
plt.xlabel("Önem Skoru")
plt.tight_layout()
plt.savefig("grafik7_ozellik_onem.png", dpi=150)
plt.show()

print("\n✅ Tüm grafikler kaydedildi.")
print("✅ Proje tamamlandı!")

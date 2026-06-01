# 🩺 Şeker Hastalığı (Diabetes) Teşhisi — Makine Öğrenmesi Projesi

## 📌 Projenin Amacı
Bu projede Pima Kızılderilileri veri seti kullanılarak **şeker hastalığı (diyabet) teşhisi**
makine öğrenmesi ile gerçekleştirilmiştir. Problem bir **ikili sınıflandırma** problemidir:
kişinin diyabetli (1) mi yoksa sağlıklı (0) mı olduğunu tahmin etmek hedeflenmiştir.

---

## 📂 Veri Seti
| Özellik | Açıklama |
|---|---|
| Pregnancies | Gebelik sayısı |
| Glucose | Kan şekeri değeri |
| BloodPressure | Kan basıncı |
| SkinThickness | Deri kalınlığı |
| Insulin | İnsülin seviyesi |
| BMI | Vücut kitle indeksi |
| DiabetesPedigreeFunction | Aile diyabet geçmişi skoru |
| Age | Yaş |
| **Outcome** | **Hedef: 0 = Sağlıklı, 1 = Diyabetli** |

🔗 Kaynak: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

---

## 🔬 Uygulanan Adımlar
1. **Veri Tanıtımı** — Boyut, sütunlar, sınıf dağılımı
2. **EDA** — Eksik değer analizi, aykırı değer tespiti (IQR), histogramlar, korelasyon
3. **Model Kurma** — Lojistik Regresyon, Karar Ağacı, Random Forest
4. **Değerlendirme** — Accuracy, Precision, Recall, F1, Confusion Matrix

---

## 📊 Elde Edilen Sonuçlar

Üç model karşılaştırılmış ve **Random Forest** en yüksek F1 skorunu elde etmiştir.

| Model | Accuracy | F1 Skoru |
|---|---|---|
| Lojistik Regresyon | ~%77 | ~%68 |
| Karar Ağacı | ~%74 | ~%65 |
| **Random Forest** | **~%80** | **~%72** |

**Önemli bulgular:**
- Glucose (kan şekeri) en belirleyici özellik olarak öne çıkmıştır.
- BMI ve Age değişkenleri de yüksek önem skoruna sahiptir.
- Eksik veriler medyan ile doldurulmuş, aykırı değerler raporlanmıştır.

---

## 🚀 Çalıştırma
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
# diabetes.csv dosyasını aynı klasöre koy
python diabetes_tahmin.py
```

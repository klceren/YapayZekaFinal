# 🩺 Şeker Hastalığı Teşhisi — Makine Öğrenmesi Projesi

## 📌 Projenin Amacı
Bu projede makine öğrenmesi algoritmaları kullanılarak **şeker hastalığı (diyabet) teşhisi** yapılmıştır.
Kişinin sağlık ölçümlerine bakılarak diyabetli mi (1) yoksa sağlıklı mı (0) olduğu tahmin edilmektedir.

---

## 📂 Veri Seti Bilgileri

🔗 Kaynak: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

| Sütun Adı | Türkçe Açıklama |
|---|---|
| Pregnancies | Gebelik sayısı |
| Glucose | Kan şekeri değeri |
| BloodPressure | Kan basıncı |
| SkinThickness | Deri kalınlığı |
| Insulin | İnsülin seviyesi |
| BMI | Vücut kitle indeksi |
| DiabetesPedigreeFunction | Aile diyabet geçmişi skoru |
| Age | Yaş |
| Outcome | Hedef: 0 = Sağlıklı, 1 = Diyabetli |

- Toplam satır sayısı: 768
- Toplam sütun sayısı: 9
- Sağlıklı kişi sayısı: 547 (%71.2)
- Diyabetli kişi sayısı: 221 (%28.8)

---

## 🔬 Proje Aşamaları

### 1. Veri Tanıtımı
Veri setinin boyutu, sütunlar ve sınıf dağılımı incelendi.

### 2. Veri Temizleme ve Analiz
- Kan şekeri, kan basıncı gibi sütunlarda sıfır olamayacak değerler tespit edildi
- Eksik değerler medyan ile dolduruldu
- Aykırı değerler IQR yöntemiyle bulundu
- Histogramlar, korelasyon ısı haritası ve dağılım grafikleri çizildi

### 3. Model Kurma
3 farklı makine öğrenmesi algoritması kullanıldı:
- Lojistik Regresyon
- Karar Ağacı
- Rastgele Orman (Random Forest)

### 4. Model Değerlendirme
Her model; doğruluk, hassasiyet, duyarlılık ve F1 skoru ile değerlendirildi.

---

## 📊 Model Doğruluk Sonuçları

| Model | Doğruluk | Hassasiyet | Duyarlılık | F1 Skoru |
|---|---|---|---|---|
| Lojistik Regresyon | %68.83 | %35.71 | %11.36 | %17.24 |
| Karar Ağacı | %68.18 | %36.84 | %15.91 | %22.22 |
| **Rastgele Orman** | **%70.13** | **%40.00** | **%9.09** | **%14.81** |

✅ **En iyi model: Rastgele Orman** — %70.13 doğruluk oranıyla en yüksek sonucu vermiştir.

### Metrik Açıklamaları
- **Doğruluk:** Tüm tahminlerin kaçı doğru?
- **Hassasiyet:** Diyabetli dediğimizin kaçı gerçekten diyabetli?
- **Duyarlılık:** Gerçek diyabetlilerin kaçını yakaladık?
- **F1 Skoru:** Hassasiyet ve duyarlılığın dengeli ortalaması

---

## 🚀 Projeyi Çalıştırma

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python diabetes_tahmin.py
```

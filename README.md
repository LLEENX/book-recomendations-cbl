# Laporan Proyek Machine Learning - Rifal Ariya Yusuftrian

## Domain Proyek

Proyek ini bertujuan untuk membangun **sistem rekomendasi buku** menggunakan pendekatan **Collaborative Filtering berbasis Neural Network**. Menurut Su & Khoshgoftaar (2009), *collaborative filtering* adalah metode yang umum digunakan dalam sistem rekomendasi karena kemampuannya dalam menangkap pola preferensi pengguna berdasarkan rating historis. Permasalahan ini relevan karena dalam era digital, pengguna menghadapi banyak pilihan buku, sehingga sistem rekomendasi yang efektif sangat membantu meningkatkan pengalaman membaca.

Dataset yang digunakan merupakan data rating buku dari Amazon yang tersedia di [Kaggle](https://www.kaggle.com/datasets/saurabhbagchi/books-dataset), yang berisi informasi buku, pengguna, dan penilaian buku oleh pengguna.

---

## Business Understanding

### Problem Statements

1. Bagaimana memberikan rekomendasi buku kepada pengguna berdasarkan rating yang telah mereka berikan sebelumnya?
2. Bagaimana menyarankan buku yang belum pernah dibaca pengguna tetapi berpotensi disukai?

### Goals

- Mengembangkan model sistem rekomendasi berbasis rating pengguna.
- Menghasilkan **top-N rekomendasi buku** yang relevan untuk setiap pengguna.

### Solution Approach

Dalam proyek ini, digunakan dua pendekatan sistem rekomendasi:

1. **Collaborative Filtering berbasis Neural Network**  
   → Menggunakan embedding untuk merepresentasikan user dan item (buku), dilatih dengan data rating yang telah diberikan pengguna.

2. **Content-Based Filtering (alternatif)** *(tidak diimplementasikan penuh dalam proyek ini, namun dipertimbangkan)*  
   → Berdasarkan kesamaan atribut buku seperti genre, penulis, atau tahun terbit.

---

## Data Understanding

### 📚 Dataset

Dataset berasal dari **Amazon Books Dataset** (via Kaggle), terdiri dari tiga file utama:

- `books.csv`: berisi data buku
- `ratings.csv`: berisi data penilaian pengguna
- `users.csv`: berisi data pengguna

📎 **Sumber Data**:  
[🔗 Kaggle - Book Recommendation Dataset](https://www.kaggle.com/datasets/saurabhbagchi/books-dataset)

### Variabel Penting

- **books.csv**
  - ISBN (string): kode unik buku
  - Book-Title (string)
  - Book-Author (string)
  - Year-Of-Publication (integer)
  - Publisher (string)

- **ratings.csv**
  - User-ID (integer)
  - ISBN (string)
  - Book-Rating (integer, rentang 0–10)

- **users.csv**
  - User-ID (integer)
  - Location (string)
  - Age (float)

---

## 🔍 Pemeriksaan Struktur Data

- **books.csv**: 271.360 baris, sebagian kecil kolom `Book-Author` dan `Publisher` memiliki missing values.
- **users.csv**: 278.858 pengguna, kolom `Age` memiliki banyak missing dan outlier.
- **ratings.csv**: 1.149.780 data rating, lengkap dan konsisten.

### 📊 Statistik Jumlah Data

| Jenis Data                | Jumlah    |
|---------------------------|-----------|
| Buku unik                 | 271.360   |
| Pengguna                  | 278.858   |
| Pengguna memberi rating   | ±105.000  |
| Buku yang dinilai         | ±200.000  |

---

## 🧹 Data Preparation

### Pembersihan Data

- Hapus kolom gambar (`Image-URL`)
- Isi nilai kosong pada kolom `Book-Author` dan `Publisher`
  - Untuk kolom `Publisher`, dilakukan pencarian manual ke Amazon.
  - Ditemukan dua ISBN dengan data penerbit sebagai berikut:
    - `193169656X` → Penerbit: **NovelBooks, Inc.**
    - `1931696993` → Penerbit: **CreateSpace Independent Publishing Platform**
- Normalisasi nilai pada kolom `Year-Of-Publication`:
  - Buang nilai yang tidak logis (`<1000` atau `>2025`)
  - Gantikan nilai NaN dengan median, lalu ubah ke tipe data `int`
- Normalisasi kolom `Age` (umur pengguna):
  - Angka yang tidak realistis (`Age < 5` atau `Age > 90`) diubah menjadi `NaN`
  - Nilai kosong diisi dengan rata-rata, lalu dikonversi menjadi integer

---

### 🔗 Merge Dataset

Gabungkan `ratings.csv`, `books.csv`, dan `users.csv` menjadi satu dataframe:

```python
data_merged = ratings.merge(books, on='ISBN').merge(users, on='User-ID')
```

---

### 🔍 Filter & Encoding

- Normalisasi rating ke rentang 0–1
- Split data menjadi `train` dan `validation`

---

## 🧠 Modeling

### Arsitektur Model RecommenderNet

Model terdiri dari:

- Embedding layer untuk user dan book (ukuran 50)
- Penjumlahan bias dan dot product dari embedding user–book
- Output diproses melalui fungsi aktivasi sigmoid untuk menghasilkan rating prediksi dalam rentang 0–1

```python
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_books, embedding_size=50, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.user_embedding = layers.Embedding(num_users, embedding_size)
        self.user_bias = layers.Embedding(num_users, 1)
        self.book_embedding = layers.Embedding(num_books, embedding_size)
        self.book_bias = layers.Embedding(num_books, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        book_vector = self.book_embedding(inputs[:, 1])
        book_bias = self.book_bias(inputs[:, 1])
        dot_user_book = tf.reduce_sum(user_vector * book_vector, axis=1, keepdims=True)
        x = dot_user_book + user_bias + book_bias
        return tf.nn.sigmoid(x)
```

---

## ⚙️ Kompilasi & Pelatihan

- **Loss Function**: `BinaryCrossentropy`  
- **Optimizer**: `Adam`  
- **Metrik**: `RootMeanSquaredError`  
- **Epoch**: 30  
- **Batch size**: 64

```python
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=30,
    batch_size=64
)
```

---

## 📈 Evaluation

### Visualisasi Hasil

```python
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model_metrics')
plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
```
![image](https://github.com/user-attachments/assets/21364825-443a-41ea-a2cc-635aabc42e3a)


---

## Penjelasan RMSE

RMSE (Root Mean Squared Error) dihitung dengan rumus:

\[
\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 }
\]

RMSE digunakan untuk mengukur seberapa besar perbedaan antara prediksi model dengan nilai rating aktual.

---

## 📊 Interpretasi Evaluasi

| Metrik           | Nilai   | Penjelasan                               |
|------------------|---------|------------------------------------------|
| RMSE (Training)  | ~0.019  | Cukup rendah (rating distandarisasi 0–1)|
| RMSE (Validation)| ~0.310  | Konsisten, tidak overfitting            |

Model stabil dan mampu mempelajari preferensi pengguna dengan cukup baik.

---

## 🔮 Prediksi & Rekomendasi

### 📖 Cara Kerja Rekomendasi

1. Pilih user secara acak  
2. Identifikasi buku yang belum dirating  
3. Prediksi kemungkinan user menyukai buku  
4. Pilih 10 buku teratas dengan prediksi tertinggi  

```python
ratings = model.predict(user_book_array)
top_ratings_indices = ratings.argsort()[-10:][::-1]
```

---

## 📘 Contoh Hasil Rekomendasi

**📚 Menampilkan Rekomendasi untuk User ID: 92979**  
===================================

### 📘 Buku yang sebelumnya diberi rating tinggi:
-----------------------------------  
- *A Yellow Raft in Blue Water* — oleh Michael Dorris  
- *More Headlines* — oleh Jay Leno  
-----------------------------------

### 📗 10 Rekomendasi Buku Terbaik:
-----------------------------------  
- *The Boy Next Door* — oleh Meggin Cabot  
- *Harold and the Purple Crayon 50th Anniversary Edition (Purple Crayon Books)* — oleh Crockett Johnson  
- *I Am Legend* — oleh Richard Matheson  
- *Secrets of the Vine Devotional (The Breakthrough Series)* — oleh Bruce Wilkinson  
- *The Door into Summer* — oleh Robert A. Heinlein  
- *Live Albom: The Best of Detroit Free Press Sports Columnist Mitch Albom (Live Albom)* — oleh Mitch Albom  
- *The Vampire Lestat (Vampire Chronicles, Book II)* — oleh ANNE RICE  
- *Flashback* — oleh Nevada Barr  
- *The Collected Stories of Isaac Bashevis Singer* — oleh Isaac Bashevis Singer  
- *Curanderismo: Mexican American Folk Healing* — oleh Robert T., Ii Trotter  
-----------------------------------

---

## ✅ Kesimpulan

Proyek ini berhasil membangun sistem rekomendasi buku berbasis **Collaborative Filtering Neural Network** menggunakan data rating buku dari Amazon.

### 🔧 Model:
- Stabil dan akurat dalam memprediksi preferensi pengguna  
- Mampu menyarankan buku yang belum pernah dibaca user tetapi berpotensi disukai  

### 💡 Dampak Bisnis:
- Membantu pengguna menemukan buku relevan lebih cepat  
- Berpotensi meningkatkan interaksi, retensi, dan pengalaman pengguna dalam platform e-book atau e-commerce

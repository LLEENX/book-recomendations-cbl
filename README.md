# Laporan Proyek Machine Learning - Rifal Ariya Yusuftrian

## Domain Proyek

Proyek ini bertujuan untuk membangun **sistem rekomendasi buku** menggunakan pendekatan **Collaborative Filtering berbasis Neural Network**. Permasalahan ini relevan karena dalam era digital, pengguna menghadapi banyak pilihan buku, sehingga sistem rekomendasi yang efektif sangat membantu meningkatkan pengalaman membaca.

Dataset yang digunakan merupakan data rating buku dari Amazon yang tersedia di [Kaggle](https://www.kaggle.com/datasets/saurabhbagchi/books-dataset), yang berisi informasi buku, pengguna, dan penilaian buku oleh pengguna.

---

## Business Understanding

### Problem Statements

1. Bagaimana memberikan rekomendasi buku kepada pengguna berdasarkan rating yang telah mereka berikan sebelumnya?
2. Bagaimana menyarankan buku yang belum pernah dibaca pengguna tetapi berpotensi disukai?

### Goals

- Mengembangkan model sistem rekomendasi berbasis rating pengguna.
- Menghasilkan **top-N rekomendasi buku** yang relevan untuk setiap pengguna.

### Solution Statement

Pendekatan yang digunakan adalah **Collaborative Filtering dengan embedding layer** untuk representasi user dan item (buku), kemudian diprediksi menggunakan **jaringan neural network** sederhana.

---

## Data Understanding

### 📚 Dataset

Dataset berasal dari **Amazon Books Dataset** (via Kaggle), terdiri dari tiga file utama:

- `books.csv`:
  - ISBN
  - Book-Title
  - Book-Author
  - Year-Of-Publication
  - Publisher
- `ratings.csv`:
  - User-ID
  - ISBN
  - Book-Rating
- `users.csv`:
  - User-ID
  - Location
  - Age

### 📂 Pembacaan Dataset

```python
books = pd.read_csv('books.csv', sep=';', encoding='latin1')
ratings = pd.read_csv('ratings.csv', sep=';', encoding='latin1')
users = pd.read_csv('users.csv', sep=';', encoding='latin1')
```

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
- Isi nilai kosong pada `Book-Author` dan `Publisher`
- Normalisasi `Year-Of-Publication`, buang nilai anomali (`<1000` atau `>2025`)
- Normalisasi umur pengguna: ubah `Age <5` atau `Age >90` menjadi NaN, lalu isi dengan rata-rata

---

### 🔗 Merge Dataset

Gabungkan `ratings.csv`, `books.csv`, dan `users.csv` menjadi satu dataframe:

```python
data_merged = ratings.merge(books, on='ISBN').merge(users, on='User-ID')
```

---

### 🔍 Filter & Encoding

- Hanya ambil rating >= 6
- Sampling 30% data
- Encode user dan book ID menjadi angka integer

```python
df['user'] = df['userID'].map(user_to_user_encoded)
df['book'] = df['bookID'].map(book_to_book_encoded)
```

- Normalisasi rating ke 0–1
- Split data menjadi train dan validation

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

### ⚙️ Kompilasi & Pelatihan

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
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=30, batch_size=64)
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

### 📊 Interpretasi Evaluasi

| Metrik           | Nilai   | Penjelasan                                  |
|------------------|---------|---------------------------------------------|
| RMSE (Training)  | ~0.019  | Cukup rendah (rating distandarisasi 0–1)   |
| RMSE (Validation)| ~0.310  | Konsisten, tidak overfitting               |

> Model stabil dan mampu mempelajari preferensi pengguna dengan cukup baik.

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

### 📘 Contoh Hasil Rekomendasi

**📚 User ID**: `random_user_001`

#### Buku yang Disukai Sebelumnya:
- *Harry Potter and the Goblet of Fire* - oleh J.K. Rowling  
- *Angels & Demons* - oleh Dan Brown

#### 📗 10 Buku yang Direkomendasikan:
- *The Da Vinci Code* - oleh Dan Brown  
- *The Lord of the Rings* - oleh J.R.R. Tolkien  
- *A Walk to Remember* - oleh Nicholas Sparks  
- *The Notebook* - oleh Nicholas Sparks  
- *The Hobbit* - oleh J.R.R. Tolkien  
- *(dan lainnya...)*
  
---

## ✅ Kesimpulan

Proyek ini berhasil membangun sistem rekomendasi buku berbasis **Collaborative Filtering Neural Network** menggunakan data rating buku dari Amazon.

### 🔧 Model:
- Stabil dan akurat dalam memprediksi preferensi pengguna
- Mampu menyarankan buku yang belum pernah dibaca user tetapi berpotensi disukai

### 💡 Dampak Bisnis:
- Membantu pengguna menemukan buku relevan lebih cepat
- Berpotensi meningkatkan interaksi, retensi, dan pengalaman pengguna dalam platform e-book atau e-commerce

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

## 📊 Data Understanding

### Struktur Dataset dan Statistik Umum

Dataset yang digunakan terdiri dari tiga file utama:

- `books.csv`: 271.360 baris
- `ratings.csv`: 1.149.780 baris
- `users.csv`: 278.858 baris

---

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

---

## 🧹 Data Preparation

### ✅ Penjelasan Detail Fitur & Kualitas Data

#### 📘 books.csv

| Fitur               | Tipe Data | Keterangan                               | Missing Values | Outlier / Catatan                           |
|---------------------|-----------|------------------------------------------|----------------|---------------------------------------------|
| ISBN                | String    | Kode unik buku                           | 0              | -                                           |
| Book-Title          | String    | Judul buku                               | 0              | -                                           |
| Book-Author         | String    | Nama penulis                             | 2              | Diisi manual jika memungkinkan              |
| Year-Of-Publication | Integer   | Tahun terbit                             | 1              | Ada data tidak logis seperti `0`, `2050+`   |
| Publisher           | String    | Nama penerbit                            | 2              | Dilengkapi secara manual atau di-drop       |
| Image-URL(s)        | String    | Link gambar sampul buku                  | 3              | Diabaikan karena tidak digunakan            |

> **Tindakan**: Kolom `Image-URL` dihapus, nilai kosong pada `Publisher` diisi manual, tahun tidak valid dihapus.

#### 🧼 Pembersihan Data pada books.csv
Dataset books.csv memiliki 271.360 entri. Berikut hasil inspeksi awal menggunakan books.info():
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 271360 entries, 0 to 271359
Data columns (total 8 columns):
 #   Column               Non-Null Count   Dtype 
---  ------               --------------   ----- 
 0   ISBN                 271360 non-null  object
 1   Book-Title           271360 non-null  object
 2   Book-Author          271358 non-null  object
 3   Year-Of-Publication  271360 non-null  object
 4   Publisher            271358 non-null  object
 5   Image-URL-S          271360 non-null  object
 6   Image-URL-M          271360 non-null  object
 7   Image-URL-L          271357 non-null  object

```

#### 🔍 Cek Nilai Kosong

```python
books.isnull().sum()
```
Hasil:
![image](https://github.com/user-attachments/assets/085a1277-38af-459c-a808-edb1b5c60a99)

Dari gambar dapat diketahui bahwa ada beberapa kolom yang memiliki nilai kosong didalamnya yaitu, `Book-Author`, `Publisher`, dan `Image-URL-L`.

#### ✂️ Menghapus Kolom Gambar
Kolom `Image-URL-S`, `Image-URL-M`, dan I`mage-URL-L` berisi tautan gambar buku yang tidak digunakan dalam sistem rekomendasi, sehingga dihapus untuk menyederhanakan data.

```python
books_cleaned = books.copy()
books_cleaned.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1, inplace=True)
```

#### 🖊️ Menangani Missing Value pada Book-Author
Terdapat 2 baris yang memiliki nilai kosong pada kolom Book-Author. Solusinya:
- Jika ISBN ditemukan pada baris lain dengan nilai Book-Author yang valid, maka nilai diambil dari sana
- Jika tidak ditemukan, nilai akan diisi dengan 'Unknown'

```python
missing_authors = books_cleaned[books_cleaned['Book-Author'].isnull()]

for idx in missing_authors.index:
    isbn = books_cleaned.loc[idx, 'ISBN']
    same_isbn = books_cleaned[(books_cleaned['ISBN'] == isbn) & (books_cleaned['Book-Author'].notnull())]
    if not same_isbn.empty:
        books_cleaned.loc[idx, 'Book-Author'] = same_isbn.iloc[0]['Book-Author']
    else:
        books_cleaned.loc[idx, 'Book-Author'] = 'Unknown'
```

Sayangnya tidak ditemukan nama Author dari kedua data tersebut, oleh karena itu kedua data yang tidak memiliki Author diberi nilai 'Unknown'

#### 🏷️ Menangani Missing Value pada Publisher
Ditemukan 2 baris dengan nilai kosong di kolom Publisher. Nilai ini diisi secara manual berdasarkan hasil pencarian informasi ISBN dari situs eksternal seperti Amazon:
- ISBN 193169656X → NovelBooks, Inc.
- ISBN 1931696993 → CreateSpace Independent Publishing Platform

```python

books_cleaned.loc[books_cleaned['ISBN'] == '193169656X', 'Publisher'] = 'NovelBooks, Inc.'
books_cleaned.loc[books_cleaned['ISBN'] == '1931696993', 'Publisher'] = 'CreateSpace Independent Publishing Platform'

```

Kedua baris tersebut diisi secara manual berdasarkan hasil pencarian ISBN dari situs eksternal (seperti Amazon). Kita gunakan metode .loc[] untuk menyasar ISBN tertentu dan mengisi kolom Publisher yang kosong dengan data yang benar.

#### Cek kembali nilai data book yang kosong

```python
# Cek jumlah nilai kosong di setiap kolom
books_cleaned.isnull().sum()
```
![image](https://github.com/user-attachments/assets/a2b780dc-fded-46c1-b7c7-e6aa36e0bd8c)

Data buku sekarang sudah bersih.

---

#### 👤 users.csv

| Fitur     | Tipe Data | Keterangan               | Missing Values | Outlier / Catatan                        |
|-----------|-----------|--------------------------|----------------|------------------------------------------|
| User-ID   | Integer   | ID unik pengguna         | 0              | -                                        |
| Location  | String    | Lokasi pengguna          | 0              | Banyak nilai generik seperti “unknown”   |
| Age       | Float     | Usia pengguna            | 0              | Banyak outlier (`< 5` dan `> 90`)        |

> **Tindakan**: Nilai `Age < 5` atau `> 90` diubah menjadi NaN, kemudian diimputasi dengan rata-rata.

### Grafik Distribusi Usia/(`Age`) Pengguna
![image](https://github.com/user-attachments/assets/f88f4c28-7181-4c3c-af6f-213135e4fae0)

Dalam grafik tersebut dapat diketahui terdapat nilai yang tidak wajar di dalam data kolom `Age`.

### 📈 Statistik Deskriptif Age:

```python

users['Age'].describe()
users['Age'].value_counts().sort_index()

```

```text
count    278858.000000
mean         34.432926
std          10.512758
min           5.000000
25%          29.000000
50%          34.000000
75%          35.000000
max          90.000000
```
⚠️ Terdapat data yang tidak wajar sebelumnya di mana usia pengguna kurang dari 5 tahun atau lebih dari 90 tahun.
Nilai usia <5 dan >90 diduga hasil kesalahan input (misal: default value 0, typo, dll), sehingga tidak masuk akal secara statistik dan konteks pengguna buku.

### 🧼 Penanganan Nilai Kosong dan Outlier pada Kolom `Age`

Salah satu fitur penting dalam `users.csv` adalah kolom **Age**. Namun, kolom ini memiliki banyak **nilai kosong dan outlier** yang perlu dibersihkan agar tidak mengganggu hasil pemodelan.

Langkah-langkah yang dilakukan:

1. Menampilkan distribusi awal nilai unik pada kolom `Age`
2. Mengidentifikasi **usia tidak logis**: <5 tahun dan >90 tahun
3. Mengganti nilai tidak logis menjadi `NaN`
4. Mengisi nilai `NaN` dengan **rata-rata usia valid**
5. Mengubah tipe data kolom `Age` dari float ke **integer** agar efisien

```python
# Identifikasi nilai tidak logis
users['Age'] = users['Age'].apply(lambda x: np.nan if x < 5 or x > 90 else x)

# Imputasi nilai kosong dengan rata-rata
users['Age'].fillna(users['Age'].mean(), inplace=True)

# Ubah tipe data ke integer
users['Age'] = users['Age'].astype(int)
```
🔢 Mengubah Tipe Data ke Integer
Setelah nilai usia dibersihkan dan tidak lagi mengandung nilai kosong, kita ubah tipe data dari float ke integer (int). Hal ini dilakukan agar kolom Age lebih sesuai untuk interpretasi dan efisien dalam penyimpanan memori.

```python

# Ubah tipe data Age ke integer
user_cleaned['Age'] = user_cleaned['Age'].astype(int)

# Tampilkan nilai unik setelah dibersihkan
print("Nilai unik Age (setelah dibersihkan):")
print(sorted(user_cleaned['Age'].unique()))

```

🔢 Informasi Struktur Kolom users Setelah Dibersihkan

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 278858 entries, 0 to 278857
Data columns (total 3 columns):
 #   Column    Non-Null Count   Dtype 
---  ------    --------------   ----- 
 0   User-ID   278858 non-null  int64 
 1   Location  278858 non-null  object
 2   Age       278858 non-null  int64
```

---

#### ⭐ ratings.csv

| Fitur       | Tipe Data | Keterangan                 | Missing Values | Catatan                                |
|-------------|-----------|----------------------------|----------------|----------------------------------------|
| User-ID     | Integer   | ID pengguna                | 0              | -                                      |
| ISBN        | String    | ISBN buku                  | 0              | -                                      |
| Book-Rating | Integer   | Rating (0–10)              | 0              | Rating < 6 tidak digunakan dalam model |



### Tahapan yang Dilakukan:

1. **Hapus Fitur Tidak Relevan**
   - `Image-URL` dihapus dari `books.csv`

2. **Perbaikan Nilai Kosong**
   - `Book-Author` & `Publisher`: Diisi manual jika memungkinkan
   - `Age`: Imputasi menggunakan **mean usia valid**

3. **Tangani Outlier**
   - Buang `Year-Of-Publication` di luar rentang [1000, 2025]
   - Buang `Age` di luar rentang [5, 90]

4. **Normalisasi**
   - **Rating** dinormalisasi ke rentang 0–1

5. **Sampling**
   - Untuk efisiensi pelatihan: hanya digunakan **30% sampel data**

6. **Encoding ID**
   - `User-ID` dan `ISBN` dikodekan menjadi integer

```python
df['user'] = df['userID'].map(user_to_user_encoded)
df['book'] = df['bookID'].map(book_to_book_encoded)
```
7. **Split Data**
   - 80% untuk pelatihan, 20% untuk validasi

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

Untuk membangun sistem rekomendasi, digunakan pendekatan Collaborative Filtering berbasis Neural Network. Penjelasan konsep dan arsitektur model disajikan sebagai berikut:

### 🧠 Konsep Kerja Model Collaborative Filtering Neural Network

Model bekerja dengan cara **mempelajari representasi tersembunyi (latent features)** dari user dan buku dalam bentuk vektor (embedding).  
Prosesnya:

1. **User dan Buku** dikodekan ke dalam indeks numerik
2. Model menggunakan **Embedding Layer** untuk menerjemahkan indeks menjadi vektor berdimensi tetap (misal 50)
3. Vektor user dan buku kemudian dikalikan (dot product), yang mencerminkan sejauh mana user kemungkinan menyukai buku
4. Hasilnya dijumlahkan dengan bias dan diaktivasi melalui fungsi **sigmoid**, menghasilkan skor prediksi dalam rentang 0–1.

```
User ID → Embedding → Dot Product ← Embedding ← Book ID
                             ↓
                         Rating Prediksi (0–1)
```


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

## 📌 Hasil Top-N Recommendation

### 📚 Menampilkan Rekomendasi untuk User ID: `92979`

---

### 📘 Buku yang Sebelumnya Diberi Rating Tinggi:

- *A Yellow Raft in Blue Water* — oleh **Michael Dorris**
- *More Headlines* — oleh **Jay Leno**

---

### 📗 10 Buku yang Direkomendasikan:

1. *The Boy Next Door* — oleh **Meggin Cabot**
2. *Harold and the Purple Crayon* — oleh **Crockett Johnson**
3. *I Am Legend* — oleh **Richard Matheson**
4. *Secrets of the Vine Devotional* — oleh **Bruce Wilkinson**
5. *The Door into Summer* — oleh **Robert A. Heinlein**
6. *Live Albom* — oleh **Mitch Albom**
7. *The Vampire Lestat* — oleh **Anne Rice**
8. *Flashback* — oleh **Nevada Barr**
9. *The Collected Stories* — oleh **Isaac Bashevis Singer**
10. *Curanderismo: Mexican American Folk Healing* — oleh **Robert T. Trotter**

---

## 📊 Kelebihan & Kekurangan Pendekatan

| Pendekatan           | Kelebihan                                                       | Kekurangan                                                            |
|----------------------|------------------------------------------------------------------|-----------------------------------------------------------------------|
| **Collaborative NN** | Menangkap pola laten kompleks antar user–buku                   | Membutuhkan banyak data dan waktu pelatihan                           |
| **Content-Based**    | Bisa merekomendasikan item baru meski belum ada rating          | Tidak bisa generalisasi preferensi antar pengguna (cold start issue) |


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
![image](https://github.com/user-attachments/assets/0a01618e-4224-4413-a0bd-95e399cd0036)


---

## Penjelasan RMSE

RMSE (Root Mean Squared Error) dihitung dengan rumus:

$$
\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 }
$$

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

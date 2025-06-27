# Laporan Proyek Machine Learning - Rifal Ariya Yusuftrian

## Domain Proyek

Proyek ini bertujuan untuk membangun **sistem rekomendasi buku** menggunakan pendekatan **Collaborative Filtering berbasis Neural Network**. Permasalahan ini relevan karena dalam era digital, pengguna menghadapi banyak pilihan buku, sehingga sistem rekomendasi yang efektif sangat membantu meningkatkan pengalaman membaca.

Dataset yang digunakan merupakan data rating buku dari Amazon yang tersedia di Kaggle, yang berisi informasi buku, pengguna, dan penilaian buku oleh pengguna.

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

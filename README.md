# 🦁 Deep Learning – Animal Image Classification

> **Đề tài:** Ứng dụng học sâu cho bài toán phân loại động vật  
> **Ngôn ngữ:** Python 3.10  
> **Framework chính:** TensorFlow 2.10 (CPU) + DirectML Plugin  

---

## 📋 Mục lục

1. [Giới thiệu dự án](#1-giới-thiệu-dự-án)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Dataset](#3-dataset)
4. [Kiến trúc mô hình](#4-kiến-trúc-mô-hình)
5. [Pipeline xử lý](#5-pipeline-xử-lý)
6. [Kết quả thực nghiệm](#6-kết-quả-thực-nghiệm)
7. [Cài đặt môi trường](#7-cài-đặt-môi-trường)
8. [Cài đặt thư viện](#8-cài-đặt-thư-viện)
9. [Hướng dẫn chạy dự án](#9-hướng-dẫn-chạy-dự-án)
10. [Giải thích source-code chi tiết](#10-giải-thích-source-code-chi-tiết)

---

## 1. Giới thiệu dự án

Dự án xây dựng và so sánh hai mô hình học sâu (**AlexNet** và **VGG11**) trên bài toán **phân loại hình ảnh động vật** gồm 6 lớp. Mục tiêu:

- Huấn luyện và đánh giá hai kiến trúc CNN phổ biến.
- So sánh hiệu năng theo các chỉ số: Accuracy, Precision, Recall, F1-score, số tham số, thời gian huấn luyện.
- Xuất báo cáo chi tiết, biểu đồ hội tụ (loss/accuracy) và confusion matrix tự động.

---

## 2. Cấu trúc thư mục

```
ML-project-final/
│
├── data/
│   ├── train/                  # Dữ liệu huấn luyện (80% train / 20% val tự động)
│   │   ├── buffalo/
│   │   ├── cats/
│   │   ├── dogs/
│   │   ├── elephant/
│   │   ├── rhino/
│   │   └── zebra/
│   └── test/                   # Dữ liệu kiểm thử
│       ├── buffalo/
│       ├── cats/
│       ├── dogs/
│       ├── elephant/
│       ├── rhino/
│       └── zebra/
│
├── src/
│   ├── main.py                 # Pipeline huấn luyện & đánh giá chính
│   └── test.py                 # Script kiểm tra GPU/DirectML khả dụng
│
├── results/
│   ├── image/
│   │   ├── alexnet/            # Biểu đồ loss, accuracy, confusion matrix AlexNet
│   │   └── vgg11/              # Biểu đồ loss, accuracy, confusion matrix VGG11
│   └── reports/
│       ├── AlexNet_report.txt  # Báo cáo đầy đủ AlexNet
│       ├── VGG11_report.txt    # Báo cáo đầy đủ VGG11
│       └── Comparison_report.txt # Bảng so sánh & kết luận
│
└── .venv/                      # Virtual environment (Python 3.10)
```

---

## 3. Dataset

| Thuộc tính | Giá trị |
|---|---|
| Số lớp | **6** |
| Nhãn lớp | `buffalo`, `cats`, `dogs`, `elephant`, `rhino`, `zebra` |
| Cấu trúc | Thư mục theo tên lớp (ImageFolder-style) |
| Tập train | Lấy từ `data/train/`, chia tự động 80/20 bằng `validation_split` |
| Tập test | Lấy từ `data/test/`, **không shuffle** để đảm bảo tính nhất quán |
| Tổng mẫu test | **2.061 ảnh** (376 buffalo, 279 cats, 278 dogs, 376 elephant, 376 rhino, 376 zebra) |

> **Cách tổ chức dữ liệu chuẩn:** Mỗi thư mục con trong `train/` và `test/` mang tên nhãn lớp, bên trong chứa các file ảnh (`.jpg`, `.png`, ...). TensorFlow sẽ tự động đọc nhãn từ tên thư mục (`labels='inferred'`).

---

## 4. Kiến trúc mô hình

### 4.1 AlexNet (Model 1)

> Phiên bản AlexNet được điều chỉnh cho ảnh đầu vào **64×64**.

| Tham số | Giá trị |
|---|---|
| Input size | `64 × 64 × 3` |
| Batch size | `32` |
| Epochs | `30` |
| Optimizer | `RMSprop` |
| Loss | `Categorical Crossentropy` |
| Tổng tham số | **21,605,766** (~21.6 M) |

**Kiến trúc chi tiết:**

```
Input (64×64×3)
  │
  ├─ Conv2D(96, 11×11, stride=4, padding='same', ReLU)  → (16×16×96)
  ├─ MaxPool2D(3×3, stride=2)                            → (7×7×96)
  │
  ├─ Conv2D(256, 5×5, padding='same', ReLU)              → (7×7×256)
  ├─ MaxPool2D(3×3, stride=2)                            → (3×3×256)
  │
  ├─ Conv2D(384, 3×3, padding='same', ReLU)              → (3×3×384)
  ├─ Conv2D(384, 3×3, padding='same', ReLU)              → (3×3×384)
  ├─ Conv2D(256, 3×3, padding='same', ReLU)              → (3×3×256)
  ├─ MaxPool2D(3×3, stride=2)                            → (1×1×256)
  │
  ├─ Flatten()                                           → (256,)
  ├─ Dense(4096, ReLU) → Dropout(0.5)
  ├─ Dense(4096, ReLU) → Dropout(0.5)
  └─ Dense(6, Softmax)                                   → output
```

---

### 4.2 VGG11 (Model 2)

> Phiên bản VGG11 xử lý ảnh đầu vào **256×256**.

| Tham số | Giá trị |
|---|---|
| Input size | `256 × 256 × 3` |
| Batch size | `32` |
| Epochs | `25` |
| Optimizer | `RMSprop` |
| Loss | `Categorical Crossentropy` |
| Tổng tham số | **160,248,198** (~160.2 M) |

**Kiến trúc chi tiết:**

```
Input (256×256×3)
  │
  ├─ Block 1: Conv2D(64, 3×3, ReLU) → MaxPool2D(2×2)    → (128×128×64)
  ├─ Block 2: Conv2D(128, 3×3, ReLU) → MaxPool2D(2×2)   → (64×64×128)
  ├─ Block 3: Conv2D(256, 3×3, ReLU) × 2 → MaxPool2D    → (32×32×256)
  ├─ Block 4: Conv2D(512, 3×3, ReLU) × 2 → MaxPool2D    → (16×16×512)
  ├─ Block 5: Conv2D(512, 3×3, ReLU) × 2 → MaxPool2D    → (8×8×512)
  │
  ├─ Flatten()                                           → (32768,)
  ├─ Dense(4096, ReLU)
  ├─ Dense(4096, ReLU)
  └─ Dense(6, Softmax)                                   → output
```

> **Lưu ý:** VGG11 trong dự án này **không có Dropout** ở các FC layers (khác với AlexNet), điều này có thể là nguyên nhân gây overfitting nặng trên tập test.

---

## 5. Pipeline xử lý

```
Dữ liệu thô (ảnh)
    │
    ▼
[1] Đọc dữ liệu & tách train/val/test
    (tf.keras.utils.image_dataset_from_directory)
    │
    ▼
[2] Chuẩn hóa pixel về [0, 1]
    (Rescaling 1/255)
    │
    ▼
[3] Tối ưu I/O bằng prefetch + AUTOTUNE
    │
    ▼
[4] Xây dựng mô hình CNN
    (AlexNet / VGG11)
    │
    ▼
[5] Compile: optimizer=RMSprop, loss=CategoricalCrossentropy
    │
    ▼
[6] Huấn luyện (model.fit) + đo thời gian
    │
    ▼
[7] Đánh giá trên test set (model.evaluate)
    │
    ▼
[8] Dự đoán & tính chỉ số: Accuracy, Precision, Recall, F1
    │
    ▼
[9] Vẽ & lưu biểu đồ (Loss curve, Accuracy curve, Confusion Matrix)
    │
    ▼
[10] Xuất báo cáo text (.txt) & so sánh hai mô hình
```

---

## 6. Kết quả thực nghiệm

### 6.1 Bảng so sánh tổng hợp

| Mô hình | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Params | Thời gian |
|---|---|---|---|---|---|---|
| **AlexNet** | **70.79%** | **73.61%** | **71.04%** | **71.28%** | 21.61 M | 85.58 s |
| VGG11 | 29.89% | 34.80% | 27.41% | 20.72% | 160.25 M | 392.10 s |

### 6.2 Báo cáo chi tiết AlexNet (Per-class)

| Lớp | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| buffalo | 0.9356 | 0.7340 | 0.8227 | 376 |
| cats | 0.6608 | 0.6703 | 0.6655 | 279 |
| dogs | 0.6799 | 0.8022 | 0.7360 | 278 |
| elephant | 0.5620 | 0.7713 | 0.6502 | 376 |
| rhino | 0.6214 | 0.6330 | 0.6271 | 376 |
| zebra | 0.9570 | 0.6516 | 0.7753 | 376 |
| **macro avg** | **0.7361** | **0.7104** | **0.7128** | 2061 |

### 6.3 Chi phí tính toán

| Chỉ số | AlexNet | VGG11 |
|---|---|---|
| Tổng tham số | 21,605,766 | 160,248,198 |
| Thời gian huấn luyện | 82.61 s | 385.17 s |
| Thời gian inference / mẫu | 0.001440 s | 0.003360 s |
| Epoch val_loss tốt nhất | Epoch 17 | Epoch 24 |

### 6.4 Kết luận

> ✅ **AlexNet được đề xuất** cho bài toán này vì:
> - Accuracy cao hơn **gần gấp 2.4 lần** so với VGG11 (70.79% vs 29.89%)
> - Số tham số ít hơn **7.4 lần** (~21.6M vs ~160.2M)
> - Thời gian huấn luyện nhanh hơn **4.6 lần** (85.58s vs 392.10s)
> - VGG11 bị **underfitting/overfitting nghiêm trọng** do thiếu Dropout và độ phức tạp mô hình lớn hơn nhiều so với kích thước dataset.

---

## 7. Cài đặt môi trường

### ⚠️ Yêu cầu bắt buộc: Python 3.10

> TensorFlow 2.10 chỉ hỗ trợ chính thức Python **3.7 – 3.10**. Python 3.11+ sẽ gây lỗi cài đặt.

### 7.1 Kiểm tra Python 3.10 đã có chưa

```powershell
python --version
# hoặc
py -3.10 --version
```

### 7.2 Tải Python 3.10 (nếu chưa có)

Truy cập: https://www.python.org/downloads/release/python-31011/  
→ Tải `Windows installer (64-bit)` → Cài đặt với tùy chọn **"Add Python to PATH"**.

### 7.3 Tạo Virtual Environment với Python 3.10

```powershell
# Di chuyển đến thư mục dự án
cd d:\projects\ML-project-final

# Tạo .venv với đúng Python 3.10
py -3.10 -m venv .venv

# Kích hoạt môi trường (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Nếu bị lỗi policy, chạy lệnh này trước:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

> **Kiểm tra đã kích hoạt thành công:** Terminal sẽ hiện tiền tố `(.venv)` ở đầu dòng.

```powershell
# Verify Python version trong venv
python --version
# Kết quả phải là: Python 3.10.x
```

---

## 8. Cài đặt thư viện

> **Thực hiện SAU KHI đã kích hoạt `.venv`**

### 8.1 Nâng cấp pip

```powershell
python -m pip install --upgrade pip
```

### 8.2 Cài đặt NumPy (phiên bản cố định)

```powershell
pip install numpy==1.26.4
```

### 8.3 Cài đặt TensorFlow CPU

```powershell
pip install tensorflow-cpu==2.10
```

> TensorFlow 2.10 là phiên bản **cuối cùng hỗ trợ GPU native trên Windows** (và CPU build này tối ưu cho máy không có NVIDIA GPU).

### 8.4 Cài đặt DirectML Plugin (tăng tốc GPU AMD/Intel trên Windows)

```powershell
pip install tensorflow-directml-plugin
```

> DirectML cho phép TensorFlow tận dụng GPU thông qua DirectX 12 API — hỗ trợ cả GPU AMD, Intel, và NVIDIA trên Windows.

### 8.5 Cài đặt các thư viện còn lại

```powershell
pip install matplotlib scikit-learn
```

### 8.6 Tóm tắt lệnh cài đặt (chạy lần lượt)

```powershell
python -m pip install --upgrade pip
pip install numpy==1.26.4
pip install tensorflow-cpu==2.10
pip install tensorflow-directml-plugin
pip install matplotlib scikit-learn
```

### 8.7 Danh sách thư viện & vai trò

| Thư viện | Version | Vai trò |
|---|---|---|
| `numpy` | `1.26.4` | Xử lý mảng số học, argmax, array operations |
| `tensorflow-cpu` | `2.10` | Framework học sâu: build model, train, evaluate |
| `tensorflow-directml-plugin` | latest | Tăng tốc GPU trên Windows qua DirectX 12 |
| `matplotlib` | latest | Vẽ biểu đồ loss, accuracy, confusion matrix |
| `scikit-learn` | latest | Tính accuracy, precision, recall, F1, confusion matrix |

### 8.8 Kiểm tra cài đặt

```python
# Chạy file test.py để kiểm tra TensorFlow nhận GPU chưa
python src/test.py

# Kết quả mẫu (có DirectML):
# Num GPUs Available:  1
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

## 9. Hướng dẫn chạy dự án

### 9.1 Chuẩn bị dữ liệu

Tổ chức dataset theo cấu trúc:
```
data/
├── train/
│   ├── buffalo/   ← chứa ảnh trâu
│   ├── cats/      ← chứa ảnh mèo
│   ├── dogs/      ← chứa ảnh chó
│   ├── elephant/  ← chứa ảnh voi
│   ├── rhino/     ← chứa ảnh tê giác
│   └── zebra/     ← chứa ảnh ngựa vằn
└── test/
    └── (cấu trúc tương tự train/)
```

### 9.2 Kích hoạt venv và chạy

```powershell
# Kích hoạt môi trường
cd d:\projects\ML-project-final
.\.venv\Scripts\Activate.ps1

# Chạy pipeline chính
python src/main.py
```

### 9.3 Output sau khi chạy xong

| File output | Vị trí | Nội dung |
|---|---|---|
| `AlexNet_report.txt` | `results/reports/` | Toàn bộ metrics AlexNet |
| `VGG11_report.txt` | `results/reports/` | Toàn bộ metrics VGG11 |
| `Comparison_report.txt` | `results/reports/` | Bảng so sánh & kết luận |
| `ALEXNET_loss.png` | `results/image/alexnet/` | Biểu đồ loss AlexNet |
| `ALEXNET_accuracy.png` | `results/image/alexnet/` | Biểu đồ accuracy AlexNet |
| `ALEXNET_matrix.png` | `results/image/alexnet/` | Confusion matrix AlexNet |
| `VGG11_loss.png` | `results/image/vgg11/` | Biểu đồ loss VGG11 |
| `VGG11_accuracy.png` | `results/image/vgg11/` | Biểu đồ accuracy VGG11 |
| `VGG11_matrix.png` | `results/image/vgg11/` | Confusion matrix VGG11 |

---

## 10. Giải thích source-code chi tiết

### `src/main.py` — Pipeline chính (677 dòng)

#### Phần 1: Import & cấu hình môi trường (dòng 1–19)

```python
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'       # Tắt log verbose TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"      # Tắt oneDNN để tránh warning
```

#### Phần 2: Khai báo siêu tham số (dòng 21–57)

| Tham số | AlexNet | VGG11 |
|---|---|---|
| `img_height/width` | 64 × 64 | 256 × 256 |
| `batch_size` | 32 | 32 |
| `epochs` | 30 | 25 |
| `seed` | 123 | 123 |

#### Phần 3–4: Load dữ liệu (dòng 59–154)

Sử dụng `tf.keras.utils.image_dataset_from_directory()` với:
- `labels='inferred'` — tự đọc nhãn từ tên thư mục
- `label_mode='categorical'` — one-hot encoding
- `validation_split=0.2` — tự chia 80/20
- `shuffle=False` cho test set — giữ thứ tự để so sánh nhãn đúng

#### Phần 5: Chuẩn hóa dữ liệu (dòng 156–179)

```python
normalization_layer = layers.Rescaling(1.0 / 255)
# Chuyển pixel từ [0, 255] → [0.0, 1.0]
# Áp dụng cho train, val, test
# Dùng prefetch + AUTOTUNE để tối ưu pipeline I/O
```

#### Phần 6: Xây dựng mô hình (dòng 181–267)

- **AlexNet**: Dùng `models.Sequential` với `layers.Input` — kiến trúc 5 conv block + 2 FC + Dropout(0.5).
- **VGG11**: Dùng `models.Sequential` với `.add()` method — kiến trúc 5 conv block + 2 FC, **không có Dropout**.
- Cả hai dùng: `optimizer='rmsprop'`, `loss='categorical_crossentropy'`, `metrics=['accuracy']`

#### Phần 7–8: Huấn luyện & đánh giá (dòng 270–329)

```python
history = model.fit(train_ds, epochs=N, validation_data=val_ds)
test_loss, test_acc = model.evaluate(test_ds)
# Đo thời gian train và evaluate bằng time.time()
```

#### Phần 9: Inference & thu thập nhãn (dòng 331–367)

```python
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))
# Chuyển one-hot → class index để tính sklearn metrics
```

#### Phần 10–12: Tính chỉ số đánh giá (dòng 369–481)

Dùng `sklearn.metrics`:
- `accuracy_score` — độ chính xác tổng thể
- `precision_score(average='macro')` — precision trung bình các lớp
- `recall_score(average='macro')` — recall trung bình các lớp
- `f1_score(average='macro')` — F1 trung bình các lớp
- `classification_report` — báo cáo chi tiết từng lớp
- Tính chi phí: tổng params (`model.count_params()`), thời gian train/infer/per-sample

#### Phần 13: So sánh & kết luận (dòng 529–587)

Tự động xác định mô hình tốt nhất theo `accuracy` và ghi vào `Comparison_report.txt`.

#### Phần 14–15: Visualizations (dòng 590–677)

Vẽ và lưu 3 loại biểu đồ cho mỗi mô hình:
1. **Loss curve** — Training loss vs Validation loss theo epoch
2. **Accuracy curve** — Training accuracy vs Validation accuracy theo epoch
3. **Confusion matrix** — Hiển thị bằng `ConfusionMatrixDisplay` với colormap Blues

---

### `src/test.py` — Script kiểm tra GPU (11 dòng)

```python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))
```

> Dùng để xác minh DirectML plugin đã nhận GPU trước khi chạy `main.py`.

---

## 📝 Ghi chú kỹ thuật

> [!NOTE]
> **Về VGG11 accuracy thấp (29.89%):** Nguyên nhân có thể do (1) thiếu Dropout trong FC layers dẫn đến overfitting, (2) dataset nhỏ không đủ để train mô hình 160M params từ đầu, (3) cần augmentation mạnh hơn.

> [!TIP]
> Để cải thiện kết quả VGG11, có thể thêm `Dropout(0.5)` sau mỗi Dense layer và áp dụng data augmentation (flip, rotation, zoom) ngay trong pipeline TF Dataset.

> [!IMPORTANT]
> Phải dùng **Python 3.10** và cài đúng thứ tự: `numpy==1.26.4` → `tensorflow-cpu==2.10` → `tensorflow-directml-plugin`. Sai thứ tự hoặc version có thể gây conflict.

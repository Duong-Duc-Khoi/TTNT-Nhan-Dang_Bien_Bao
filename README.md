# 🚦 Nhận Dạng Biển Báo Giao Thông

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-CNN-red?logo=keras)
![OpenCV](https://img.shields.io/badge/OpenCV-Webcam-green?logo=opencv)
![Accuracy](https://img.shields.io/badge/Accuracy-~95%25-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-GTSRB-yellow)

Hệ thống nhận dạng biển báo giao thông sử dụng mạng nơ-ron tích chập (CNN), đạt độ chính xác **~95%** trên tập kiểm tra. Hỗ trợ hai chế độ hoạt động: **nhận dạng ảnh tĩnh qua giao diện desktop** và **nhận dạng thời gian thực qua webcam**.

---

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Dataset](#-dataset)
- [Kiến trúc mô hình](#-kiến-trúc-mô-hình)
- [Kết quả huấn luyện](#-kết-quả-huấn-luyện)
- [Cài đặt](#-cài-đặt)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [Danh sách 43 lớp biển báo](#-danh-sách-43-lớp-biển-báo)
- [Cấu hình hệ thống](#-cấu-hình-hệ-thống)
- [Xử lý lỗi thường gặp](#-xử-lý-lỗi-thường-gặp)
- [Tác giả](#-tác-giả)

---

## 🎯 Giới thiệu

Dự án xây dựng hệ thống nhận dạng tự động **43 loại biển báo giao thông** tiêu chuẩn châu Âu (GTSRB) bằng kỹ thuật học sâu. Hệ thống gồm ba thành phần chính:

| Thành phần | Mô tả |
|------------|-------|
| **Huấn luyện mô hình** | CNN được train trên Google Colab với dataset GTSRB |
| **Giao diện desktop** | Ứng dụng tkinter cho phép tải ảnh lên và nhận dạng |
| **Nhận dạng webcam** | Nhận dạng liên tục theo thời gian thực qua camera |

---

## 📁 Cấu trúc dự án

```
traffic-sign-recognition/
│
├── gui_app.py                  # Giao diện desktop (tkinter)
├── webcam_detection.py         # Nhận dạng thời gian thực (OpenCV)
├── cnn_training.ipynb          # Notebook huấn luyện mô hình (Google Colab)
│
├── traffic_classifier.h5       # Model dùng cho ứng dụng GUI
├── model.h5                    # Model dùng cho webcam
│
├── all.jpg                     # Ảnh banner hiển thị trên giao diện
└── README.md
```

---

## 🗄️ Dataset

**German Traffic Sign Recognition Benchmark (GTSRB)**

| Thông tin | Chi tiết |
|-----------|---------|
| Nguồn | [GTSRB – Kaggle](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) |
| Số lớp | 43 loại biển báo |
| Tổng số ảnh | ~51.000 ảnh (train + valid + test) |
| Kích thước ảnh gốc | Đa dạng, resize về 32×32 khi train |
| Định dạng | `.p` (pickle) — train / valid / test |

Dataset được tải tự động trong notebook:

```bash
https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip
```

**Tiền xử lý dữ liệu:**
- Chuẩn hóa pixel về đoạn `[0, 1]` bằng phép chia `/ 255.0`
- Nhãn được mã hóa one-hot bằng `LabelBinarizer`
- Dữ liệu train được xáo trộn ngẫu nhiên (`shuffle`) trước khi huấn luyện

---

## 🧠 Kiến trúc mô hình

Mô hình sử dụng **Keras Sequential API** với kiến trúc CNN gồm các khối tích chập đôi:

```
Input (32×32×3)
    │
    ▼
┌─────────────────────────────┐
│  Conv2D(32, 3×3) + ReLU    │
│  BatchNormalization         │  ← Khối 1
│  Conv2D(32, 3×3) + ReLU    │
│  BatchNormalization         │
│  MaxPooling2D(2×2)          │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Conv2D(64, 3×3) + ReLU    │
│  BatchNormalization         │  ← Khối 2
│  Conv2D(64, 3×3) + ReLU    │
│  BatchNormalization         │
│  MaxPooling2D(2×2)          │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Flatten                    │
│  Dense(512) + ReLU          │  ← Fully Connected
│  BatchNormalization         │
│  Dense(43) + Softmax        │  ← Output (43 lớp)
└─────────────────────────────┘
```

**Thông số huấn luyện:**

| Tham số | Giá trị |
|---------|---------|
| Optimizer | SGD (lr = 0.01, momentum = 0.9) |
| Loss function | Categorical Crossentropy |
| Số epochs | 10 |
| Batch size | 64 |
| Input shape | 32 × 32 × 3 |
| Số lớp đầu ra | 43 |

**Data Augmentation** (dùng `ImageDataGenerator`):

| Kỹ thuật | Giá trị |
|----------|---------|
| Rotation | ±0.18 độ |
| Zoom | ±15% |
| Width shift | ±20% |
| Height shift | ±20% |
| Horizontal flip | Có |

---

## 📊 Kết quả huấn luyện

Mô hình được huấn luyện **10 epochs** trên Google Colab với dataset GTSRB:

| Chỉ số | Giá trị |
|--------|---------|
| Validation accuracy | **~95%** |
| Ngưỡng tin cậy (webcam) | **75%** |
| Số lớp phân loại | 43 |

> 💡 **Gợi ý cải thiện:** Tăng số epochs lên 20–30, hoặc thay SGD bằng Adam để hội tụ nhanh hơn. Bổ sung thêm lớp Dropout để giảm overfitting.

---

## 📦 Cài đặt

### Yêu cầu hệ thống

- Python 3.8 trở lên
- Webcam (nếu dùng chế độ nhận dạng thời gian thực)
- GPU (không bắt buộc, nhưng giúp tăng tốc huấn luyện)

### Cài đặt thư viện

```bash
pip install tensorflow keras opencv-python pillow numpy scikit-learn
```

Hoặc tạo môi trường ảo trước:

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

pip install tensorflow keras opencv-python pillow numpy scikit-learn
```

---

## 🚀 Hướng dẫn sử dụng

### 1. Huấn luyện mô hình (Google Colab)

1. Mở file `cnn_training.ipynb` trên [Google Colab](https://colab.research.google.com/)
2. Mount Google Drive của bạn
3. Chạy toàn bộ các cell theo thứ tự
4. Sau khi train xong, file `CNN_for_TSR.h5` sẽ được lưu vào Drive
5. Đổi tên thành `traffic_classifier.h5` (dùng cho GUI) hoặc `model.h5` (dùng cho webcam) rồi tải về máy

---

### 2. Ứng dụng giao diện desktop

```bash
python gui_app.py
```

**Các bước sử dụng:**

1. Ứng dụng khởi động với cửa sổ **"Kiểm Tra Biển Báo"**
2. Nhấn nút **"Tải Ảnh Lên"** → chọn ảnh biển báo từ máy tính
3. Ảnh được hiển thị ở kích thước 300×300
4. Nhấn nút **"Nhận Dạng"** → kết quả xuất hiện bên dưới ảnh

> ⚠️ Yêu cầu: `traffic_classifier.h5` và `all.jpg` phải nằm cùng thư mục với `gui_app.py`.

---

### 3. Nhận dạng thời gian thực qua webcam

```bash
python webcam_detection.py
```

**Cách hoạt động:**

- Webcam đọc liên tục từng frame (640×480)
- Mỗi frame được resize về 32×32 và chuyển sang ảnh xám (grayscale)
- Áp dụng cân bằng histogram (`equalizeHist`) và chuẩn hóa về `[0, 1]`
- Nếu xác suất dự đoán vượt ngưỡng **75%**, tên biển báo và % độ tin cậy được hiển thị lên màn hình
- Nhấn **`q`** để thoát

> ⚠️ Yêu cầu: `model.h5` phải nằm cùng thư mục với `webcam_detection.py`.

---

## 🗂️ Danh sách 43 lớp biển báo

| ID | Tiếng Việt | English |
|----|------------|---------|
| 0 | Giới hạn tốc độ 20 km/h | Speed limit (20km/h) |
| 1 | Giới hạn tốc độ 30 km/h | Speed limit (30km/h) |
| 2 | Giới hạn tốc độ 50 km/h | Speed limit (50km/h) |
| 3 | Giới hạn tốc độ 60 km/h | Speed limit (60km/h) |
| 4 | Giới hạn tốc độ 70 km/h | Speed limit (70km/h) |
| 5 | Giới hạn tốc độ 80 km/h | Speed limit (80km/h) |
| 6 | Kết thúc giới hạn tốc độ 80 km/h | End of speed limit (80km/h) |
| 7 | Giới hạn tốc độ 100 km/h | Speed limit (100km/h) |
| 8 | Giới hạn tốc độ 120 km/h | Speed limit (120km/h) |
| 9 | Cấm vượt | No passing |
| 10 | Cấm vượt xe trên 3.5 tấn | No passing for vehicles over 3.5 metric tons |
| 11 | Ưu tiên tại ngã tư | Right-of-way at the next intersection |
| 12 | Đường ưu tiên | Priority road |
| 13 | Nhường đường | Yield |
| 14 | Dừng lại | Stop |
| 15 | Cấm các loại phương tiện | No vehicles |
| 16 | Cấm xe trên 3.5 tấn | Vehicles over 3.5 metric tons prohibited |
| 17 | Cấm vào | No entry |
| 18 | Chú ý chung | General caution |
| 19 | Đường cong nguy hiểm bên trái | Dangerous curve to the left |
| 20 | Đường cong nguy hiểm bên phải | Dangerous curve to the right |
| 21 | Đường cong kép | Double curve |
| 22 | Đường gồ ghề | Bumpy road |
| 23 | Đường trơn | Slippery road |
| 24 | Đường hẹp bên phải | Road narrows on the right |
| 25 | Công trình đường bộ | Road work |
| 26 | Tín hiệu giao thông | Traffic signals |
| 27 | Người đi bộ | Pedestrians |
| 28 | Trẻ em băng qua đường | Children crossing |
| 29 | Xe đạp băng qua | Bicycles crossing |
| 30 | Cẩn thận băng tuyết | Beware of ice/snow |
| 31 | Động vật hoang dã băng qua | Wild animals crossing |
| 32 | Kết thúc mọi giới hạn tốc độ và cấm vượt | End of all speed and passing limits |
| 33 | Rẽ phải phía trước | Turn right ahead |
| 34 | Rẽ trái phía trước | Turn left ahead |
| 35 | Chỉ đi thẳng | Ahead only |
| 36 | Đi thẳng hoặc rẽ phải | Go straight or right |
| 37 | Đi thẳng hoặc rẽ trái | Go straight or left |
| 38 | Đi về phía bên phải | Keep right |
| 39 | Đi về phía bên trái | Keep left |
| 40 | Bắt buộc đi theo vòng xuyến | Roundabout mandatory |
| 41 | Kết thúc lệnh cấm vượt | End of no passing |
| 42 | Kết thúc cấm vượt xe trên 3.5 tấn | End of no passing by vehicles over 3.5 metric tons |

---

## ⚙️ Cấu hình hệ thống

Các tham số quan trọng có thể điều chỉnh trực tiếp trong code:

| Tham số | Giá trị mặc định | File | Mô tả |
|---------|-----------------|------|-------|
| `threshold` | `0.75` | `webcam_detection.py` | Ngưỡng xác suất tối thiểu để hiển thị kết quả |
| `cap.set(3, 640)` | `640` | `webcam_detection.py` | Chiều rộng khung hình webcam |
| `cap.set(4, 480)` | `480` | `webcam_detection.py` | Chiều cao khung hình webcam |
| `cap.set(10, 180)` | `180` | `webcam_detection.py` | Độ sáng webcam |
| `specific_size` | `(300, 300)` | `gui_app.py` | Kích thước hiển thị ảnh trong GUI |
| `epochs` | `10` | `cnn_training.ipynb` | Số vòng lặp huấn luyện |
| `batch_size` | `64` | `cnn_training.ipynb` | Kích thước batch |
| `learning_rate` | `0.01` | `cnn_training.ipynb` | Tốc độ học |

---

## 🛠️ Xử lý lỗi thường gặp

**`ModuleNotFoundError: No module named 'keras'`**
```bash
pip install tensorflow   # Keras đã được tích hợp sẵn trong TensorFlow 2.x
```

**Webcam không mở được**
```python
# Thay đổi chỉ số camera trong webcam_detection.py
cap = cv2.VideoCapture(1)  # Thử 1, 2, ... nếu 0 không hoạt động
```

**Lỗi `FileNotFoundError` khi load model**
- Kiểm tra `traffic_classifier.h5` / `model.h5` đã nằm đúng thư mục chưa
- Đảm bảo tên file trong code khớp với tên file thực tế

**Kết quả nhận dạng không chính xác trên ảnh thực tế**
- Đảm bảo ảnh chụp rõ nét, đủ sáng
- Biển báo nên chiếm phần lớn diện tích ảnh
- Với webcam: giữ biển báo ngay giữa khung hình, tránh nhiễu nền phức tạp

**Lỗi `all.jpg not found` khi chạy GUI**
- Cần có file `all.jpg` trong cùng thư mục với `gui_app.py`
- Có thể dùng bất kỳ ảnh `.jpg` nào, đổi tên thành `all.jpg` là được

---

## 📄 Giấy phép

Dự án được thực hiện cho mục đích học thuật và nghiên cứu.  
Dataset GTSRB: [benchmark.ini.rub.de](https://benchmark.ini.rub.de/gtsrb_dataset.html)

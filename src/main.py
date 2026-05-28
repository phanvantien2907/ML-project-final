import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# =========================
# 1. KHAI BÁO ĐƯỜNG DẪN
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE_DIR, "../results/image/alexnet"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "../results/image/vgg11"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "../results/reports"),exist_ok=True)
train_dir = os.path.join(BASE_DIR, '../data/train')
test_dir = os.path.join(BASE_DIR, '../data/test')

# =========================
# 2.1. THAM SỐ
# =========================
img_height_model_1 = 64
img_width_model_1 = 64
batch_size_model_1 = 32
epochs_model_1 = 30
seed_model_1 = 123
model_name_1 = "AlexNet"

# =========================
# 2.2. THAM SỐ
# =========================
img_height_model_2 = 256
img_width_model_2 = 256
batch_size_model_2 = 32
epochs_model_2 = 25
seed_model_2 = 123
model_name_2 = "VGG11"

# File lưu toàn bộ kết quả report
report_txt_path_model_1 = os.path.join(BASE_DIR, "../results/reports/AlexNet_report.txt")
report_txt_path_model_2 = os.path.join(BASE_DIR, "../results/reports/VGG11_report.txt")
report_txt_path_compare = os.path.join(BASE_DIR, "../results/reports/Comparison_report.txt")
f_model_1 = open(report_txt_path_model_1, "w", encoding="utf-8")
f_model_2 = open(report_txt_path_model_2, "w", encoding="utf-8")
f_compare = open(report_txt_path_compare, "w", encoding="utf-8")

# =========================
# 3. KIỂM TRA THƯ MỤC DỮ LIỆU
# =========================
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Không tìm thấy thư mục train: {train_dir}")

if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Không tìm thấy thư mục test: {test_dir}")

# =========================
# 4. ĐỌC DỮ LIỆU TỪ THƯ MỤC
# =========================

# BỘ DỮ LIỆU CHO MÔ HÌNH ALEXNET
train_ds_model_1 = tf.keras.utils.image_dataset_from_directory(
    train_dir, # đọc dữ liệu
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size_model_1,
    image_size=(img_height_model_1, img_width_model_1),
    shuffle=True,
    seed=seed_model_1,
    validation_split=0.2,
    subset='training'
)

val_ds_model_1 = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size_model_1,
    image_size=(img_height_model_1, img_width_model_1),
    shuffle=True,
    seed=seed_model_1,
    validation_split=0.2,
    subset='validation'
)

test_ds_model_1 = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size_model_1,
    image_size=(img_height_model_1, img_width_model_1),
    shuffle=False
)

# BỘ DỮ LIỆU CHO MÔ HÌNH VGG11
train_ds_model_2 = tf.keras.utils.image_dataset_from_directory(
    train_dir, # đọc dữ liệu
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size_model_2,
    image_size=(img_height_model_2, img_width_model_2),
    shuffle=True,
    seed=seed_model_1,
    validation_split=0.2,
    subset='training'
)

val_ds_model_2 = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size_model_2,
    image_size=(img_height_model_2, img_width_model_2),
    shuffle=True,
    seed=seed_model_2,
    validation_split=0.2,
    subset='validation'
)

test_ds_model_2 = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size_model_2,
    image_size=(img_height_model_2, img_width_model_2),
    shuffle=False
)

# Tên lớp và số lớp thực tế mô hình ALEXNET
class_names_model_1 = train_ds_model_1.class_names
num_classes_model_1 = len(class_names_model_1)

# Tên lớp và số lớp thực tế mô hình VGG11
class_names_model_2 = train_ds_model_1.class_names
num_classes_model_2 = len(class_names_model_1)

# Tên lớp model ALEXNET
f_model_1.write(f"Class names model ALEXNET: {class_names_model_1}\n")
f_model_1.write(f"Number of classes ALEXNET: {num_classes_model_1}\n")

# Tên lớp model VGG11
f_model_2.write(f"Class names VGG11: {class_names_model_2}\n")
f_model_2.write(f"Number of classes VGG11: {num_classes_model_2}\n")

# =========================
# 5. CHUẨN HÓA DỮ LIỆU
# =========================
normalization_layer = layers.Rescaling(1.0 / 255)

# CHUẨN HOÁ DỮ LIỆU MÔ HÌNH ALEXNET
train_ds_model_1 = train_ds_model_1.map(lambda x, y: (normalization_layer(x), y))
val_ds_model_1 = val_ds_model_1.map(lambda x, y: (normalization_layer(x), y))
test_ds_model_1 = test_ds_model_1.map(lambda x, y: (normalization_layer(x), y))

AUTOTUNE_model_1 = tf.data.AUTOTUNE
train_ds_model_1 = train_ds_model_1.prefetch(buffer_size=AUTOTUNE_model_1)
val_ds_model_1 = val_ds_model_1.prefetch(buffer_size=AUTOTUNE_model_1)
test_ds_model_1 = test_ds_model_1.prefetch(buffer_size=AUTOTUNE_model_1)

# CHUẨN HOÁ DỮ LIỆU MÔ HÌNH VGG11
train_ds_model_2 = train_ds_model_2.map(lambda x, y: (normalization_layer(x), y))
val_ds_model_2 = val_ds_model_2.map(lambda x, y: (normalization_layer(x), y))
test_ds_model_2 = test_ds_model_2.map(lambda x, y: (normalization_layer(x), y))

AUTOTUNE_model_2 = tf.data.AUTOTUNE
train_ds_model_2 = train_ds_model_2.prefetch(buffer_size=AUTOTUNE_model_2)
val_ds_model_2 = val_ds_model_2.prefetch(buffer_size=AUTOTUNE_model_2)
test_ds_model_2 = test_ds_model_2.prefetch(buffer_size=AUTOTUNE_model_2)

# =========================
# 6.1. TẠO MÔ HÌNH AlexNet
# =========================
model_1 = models.Sequential([
    layers.Input(shape=(img_height_model_1, img_width_model_1, 3)),

    # Block 1
    layers.Conv2D(96, (11, 11), strides=4, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(3, 3), strides=2),

    # Block 2
    layers.Conv2D(256, (5, 5), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(3, 3), strides=2),

    # Block 3
    layers.Conv2D(384, (3, 3), padding='same', activation='relu'),

    # Block 4
    layers.Conv2D(384, (3, 3), padding='same', activation='relu'),

    # Block 5
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(3, 3), strides=2),

    # Fully-connected layers
    layers.Flatten(),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(num_classes_model_1, activation='softmax')
])

model_1.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model_1.summary(print_fn=lambda x: f_model_1.write(x + "\n"))

# =========================
# 6.2. TẠO MÔ HÌNH VGG11
# =========================

model_2 = models.Sequential()

# Block 1
model_2.add(layers.Conv2D(64, kernel_size=(3, 3), padding='same',
    activation='relu', input_shape=(img_height_model_2, img_width_model_2, 3)
))
model_2.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Block 2
model_2.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model_2.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Block 3
model_2.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
model_2.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
model_2.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Block 4
model_2.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
model_2.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
model_2.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Block 5
model_2.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
model_2.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
model_2.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Fully connected
model_2.add(layers.Flatten())
model_2.add(layers.Dense(4096, activation='relu'))
model_2.add(layers.Dense(4096, activation='relu'))
model_2.add(layers.Dense(num_classes_model_2, activation='softmax'))

model_2.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model_2.summary(print_fn=lambda x: f_model_2.write(x + "\n"))


# =========================
# 7.1. HUẤN LUYỆN MÔ HÌNH ALEXNET
# =========================
start_train_time_model_1 = time.time()

history_model_1 = model_1.fit(
    train_ds_model_1,
    epochs=epochs_model_1,
    validation_data=val_ds_model_1,
    verbose=1
)

end_train_time_model_1 = time.time()
training_time_model_1 = end_train_time_model_1 - start_train_time_model_1

# =========================
# 7.2. ĐÁNH GIÁ MÔ HÌNH ALEXNET
# =========================
start_test_time_model_1 = time.time()
test_loss_model_1 , test_acc_model_1  = model_1.evaluate(test_ds_model_1, verbose=1)
end_test_time_model_1  = time.time()
testing_time_model_1  = end_test_time_model_1 - start_test_time_model_1

f_model_1.write("\n===== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ALEXNET =====\n")
f_model_1.write(f"Test loss     : {test_loss_model_1:.4f}\n")
f_model_1.write(f"Test accuracy : {test_acc_model_1 * 100:.2f}%\n")

best_epoch = np.argmin(history_model_1.history['val_loss']) + 1
f_model_1.write(f"Epoch có val_loss nhỏ nhất: {best_epoch}\n")


# =========================
# 8.1. HUẤN LUYỆN MÔ HÌNH VGG11
# =========================
start_train_time_model_2 = time.time()

history_model_2 = model_2.fit(
    train_ds_model_2,
    epochs=epochs_model_2,
    validation_data=val_ds_model_2,
    verbose=1
)

end_train_time_model_2 = time.time()
training_time_model_2 = end_train_time_model_2 - start_train_time_model_2

# =========================
# 8.2. ĐÁNH GIÁ MÔ HÌNH VGG11
# =========================
start_test_time_model_2 = time.time()
test_loss_model_2 , test_acc_model_2  = model_2.evaluate(test_ds_model_2, verbose=1)
end_test_time_model_2  = time.time()
testing_time_model_2  = end_test_time_model_2 - start_test_time_model_2

f_model_2.write("\n===== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH VGG11 =====\n")
f_model_2.write(f"Test loss     : {test_loss_model_2:.4f}\n")
f_model_2.write(f"Test accuracy : {test_acc_model_2 * 100:.2f}%\n")

best_epoch = np.argmin(history_model_2.history['val_loss']) + 1
f_model_2.write(f"Epoch có val_loss nhỏ nhất: {best_epoch}\n")

# =========================
# 9.1. DỰ ĐOÁN TRÊN TEST SET MÔ HÌNH ALEXNET
# =========================
y_true_model_1 = []
y_pred_model_1 = []

start_infer_time_model_1 = time.time()

for images, labels in test_ds_model_1:
    preds = model_1.predict(images, verbose=0)
    y_true_model_1.extend(np.argmax(labels.numpy(), axis=1))
    y_pred_model_1.extend(np.argmax(preds, axis=1))

end_infer_time_model_1 = time.time()
inference_time_model_1 = end_infer_time_model_1 - start_infer_time_model_1

y_true_model_1 = np.array(y_true_model_1)
y_pred_model_1 = np.array(y_pred_model_1)

# =========================
# 9.2. DỰ ĐOÁN TRÊN TEST SET MÔ HÌNH VGG11
# =========================
y_true_model_2 = []
y_pred_model_2 = []

start_infer_time_model_2 = time.time()

for images, labels in test_ds_model_2:
    preds = model_2.predict(images, verbose=0)
    y_true_model_2.extend(np.argmax(labels.numpy(), axis=1))
    y_pred_model_2.extend(np.argmax(preds, axis=1))

end_infer_time_model_2 = time.time()
inference_time_model_2 = end_infer_time_model_2 - start_infer_time_model_2

y_true_model_2 = np.array(y_true_model_2)
y_pred_model_2 = np.array(y_pred_model_2)

# =========================
# 10.1. CÁC CHỈ SỐ ĐÁNH GIÁ MÔ HÌNH  ALEXNET
# =========================
acc_model_1 = accuracy_score(y_true_model_1, y_pred_model_1)
precision_macro_model_1 = precision_score(y_true_model_1, y_pred_model_1, average='macro', zero_division=0)
recall_macro_model_1 = recall_score(y_true_model_1, y_pred_model_1, average='macro', zero_division=0)
f1_macro_model_1 = f1_score(y_true_model_1, y_pred_model_1, average='macro', zero_division=0)

f_model_1.write("\n===== KẾT QUẢ ĐÁNH GIÁ (MACRO) MÔ HÌNH ALEXNET =====\n")
f_model_1.write(f"Accuracy        : {acc_model_1 * 100:.2f}%\n")
f_model_1.write(f"Precision_macro : {precision_macro_model_1 * 100:.2f}%\n")
f_model_1.write(f"Recall_macro    : {recall_macro_model_1 * 100:.2f}%\n")
f_model_1.write(f"F1_macro        : {f1_macro_model_1 * 100:.2f}%\n")

f_model_1.write("\n===== BÁO CÁO CHI TIẾT TỪNG LỚP MÔ HÌNH ALEXNET =====\n")
class_report_model_1 = classification_report(
    y_true_model_1,
    y_pred_model_1,
    target_names=class_names_model_1,
    digits=4,
    zero_division=0
)
f_model_1.write(class_report_model_1 + "\n")

# =========================
# 10.2. CÁC CHỈ SỐ ĐÁNH GIÁ MÔ HÌNH  VGG11
# =========================
acc_model_2 = accuracy_score(y_true_model_2, y_pred_model_2)
precision_macro_model_2 = precision_score(y_true_model_2, y_pred_model_2, average='macro', zero_division=0)
recall_macro_model_2 = recall_score(y_true_model_2, y_pred_model_2, average='macro', zero_division=0)
f1_macro_model_2 = f1_score(y_true_model_2, y_pred_model_2, average='macro', zero_division=0)

f_model_2.write("\n===== KẾT QUẢ ĐÁNH GIÁ (MACRO) MÔ HÌNH  VGG11 =====\n")
f_model_2.write(f"Accuracy        : {acc_model_2 * 100:.2f}%\n")
f_model_2.write(f"Precision_macro : {precision_macro_model_2 * 100:.2f}%\n")
f_model_2.write(f"Recall_macro    : {recall_macro_model_2 * 100:.2f}%\n")
f_model_2.write(f"F1_macro        : {f1_macro_model_2 * 100:.2f}%\n")

f_model_2.write("\n===== BÁO CÁO CHI TIẾT TỪNG LỚP MÔ HÌNH VGG11 =====\n")
class_report_model_2 = classification_report(
    y_true_model_2,
    y_pred_model_2,
    target_names=class_names_model_2,
    digits=4,
    zero_division=0
)
f_model_2.write(class_report_model_2 + "\n")

# =========================
# 11.1. MA TRẬN NHẦM LẪN MÔ HÌNH ALEXNET
# =========================
cm_model_1 = confusion_matrix(y_true_model_1, y_pred_model_1)

fig_cm_model_1, ax_cm_model_1 = plt.subplots(figsize=(10, 8))
disp_model_1 = ConfusionMatrixDisplay(confusion_matrix=cm_model_1, display_labels=class_names_model_1)
disp_model_1.plot(cmap='Blues', xticks_rotation=45, values_format='d', ax=ax_cm_model_1, colorbar=False)
plt.title("Confusion Matrix MÔ HÌNH ALEXNET")
plt.tight_layout()


# =========================
# 11.2. MA TRẬN NHẦM LẪN MÔ HÌNH VGG11
# =========================
cm_model_2 = confusion_matrix(y_true_model_2, y_pred_model_2)

fig_cm_model_2, ax_cm_model_2 = plt.subplots(figsize=(10, 8))
disp_model_2 = ConfusionMatrixDisplay(confusion_matrix=cm_model_2, display_labels=class_names_model_2)
disp_model_2.plot(cmap='Blues', xticks_rotation=45, values_format='d', ax=ax_cm_model_2, colorbar=False)
plt.title("Confusion Matrix MÔ HÌNH VGG11")
plt.tight_layout()


# =========================
# 12.1. CHI PHÍ TÍNH TOÁN MÔ HÌNH ALEXNET
# =========================
total_params_model_1 = model_1.count_params()
params_million_model_1 = total_params_model_1 / 1e6

# Dùng train + infer để biểu diễn chi phí thực nghiệm tổng quát
total_time_model_1 = training_time_model_1 + inference_time_model_1

num_test_samples_model_1 = len(y_true_model_1)
avg_inference_time_per_sample_model_1 = inference_time_model_1 / num_test_samples_model_1 if num_test_samples_model_1 > 0 else 0.0

f_model_1.write("\n===== CHI PHÍ TÍNH TOÁN MÔ HÌNH ALEXNET =====\n")
f_model_1.write(f"Tổng số tham số                 : {total_params_model_1:,}\n")
f_model_1.write(f"Số tham số (triệu)             : {params_million_model_1:.4f} M\n")
f_model_1.write(f"Thời gian huấn luyện           : {training_time_model_1:.4f} giây\n")
f_model_1.write(f"Thời gian evaluate             : {testing_time_model_1:.4f} giây\n")
f_model_1.write(f"Thời gian suy luận test set    : {inference_time_model_1:.4f} giây\n")
f_model_1.write(f"Tổng thời gian (train+infer)   : {total_time_model_1:.4f} giây\n")
f_model_1.write(f"Thời gian suy luận / mẫu       : {avg_inference_time_per_sample_model_1:.6f} giây/mẫu\n")

# =========================
# 12.2. CHI PHÍ TÍNH TOÁN MÔ HÌNH VGG11
# =========================
total_params_model_2 = model_2.count_params()
params_million_model_2 = total_params_model_2 / 1e6

# Dùng train + infer để biểu diễn chi phí thực nghiệm tổng quát
total_time_model_2 = training_time_model_2 + inference_time_model_2

num_test_samples_model_2 = len(y_true_model_2)
avg_inference_time_per_sample_model_2 = inference_time_model_2 / num_test_samples_model_2 if num_test_samples_model_2 > 0 else 0.0

f_model_2.write("\n===== CHI PHÍ TÍNH TOÁN MÔ HÌNH VGG11 =====\n")
f_model_2.write(f"Tổng số tham số                 : {total_params_model_2:,}\n")
f_model_2.write(f"Số tham số (triệu)             : {params_million_model_2:.4f} M\n")
f_model_2.write(f"Thời gian huấn luyện           : {training_time_model_2:.4f} giây\n")
f_model_2.write(f"Thời gian evaluate             : {testing_time_model_2:.4f} giây\n")
f_model_2.write(f"Thời gian suy luận test set    : {inference_time_model_2:.4f} giây\n")
f_model_2.write(f"Tổng thời gian (train+infer)   : {total_time_model_2:.4f} giây\n")
f_model_2.write(f"Thời gian suy luận / mẫu       : {avg_inference_time_per_sample_model_2:.6f} giây/mẫu\n")

# =========================
# 13.1. IN BẢNG KẾT QUẢ MÔ HÌNH ALEXNET
# =========================
f_model_1.write("\n===== BẢNG KẾT QUẢ MÔ HÌNH ALEXNET  =====\n")

f_model_1.write(f"{'Mô hình':<20}"
        f"{'Accuracy (%)':<15}"
        f"{'Precision (%)':<18}"
        f"{'Recall (%)':<15}"
        f"{'F1_macro (%)':<15}"
        f"{'Params (M)':<15}"
        f"{'Time (s)':<12}\n")

f_model_1.write("-" * 110 + "\n")

f_model_1.write(f"{model_name_1:<20}"
        f"{acc_model_1 * 100:<15.2f}"
        f"{precision_macro_model_1 * 100:<18.2f}"
        f"{recall_macro_model_1 * 100:<15.2f}"
        f"{f1_macro_model_1 * 100:<15.2f}"
        f"{params_million_model_1:<15.4f}"
        f"{total_time_model_1:<12.2f}\n")

# =========================
# 13.2. IN BẢNG KẾT QUẢ MÔ HÌNH VGG11
# =========================
f_model_2.write("\n===== BẢNG KẾT QUẢ MÔ HÌNH VGG11  =====\n")

f_model_2.write(f"{'Mô hình':<20}"
        f"{'Accuracy (%)':<15}"
        f"{'Precision (%)':<18}"
        f"{'Recall (%)':<15}"
        f"{'F1_macro (%)':<15}"
        f"{'Params (M)':<15}"
        f"{'Time (s)':<12}\n")

f_model_2.write("-" * 110 + "\n")

f_model_2.write(f"{model_name_2:<20}"
        f"{acc_model_2 * 100:<15.2f}"
        f"{precision_macro_model_2 * 100:<18.2f}"
        f"{recall_macro_model_2 * 100:<15.2f}"
        f"{f1_macro_model_2 * 100:<15.2f}"
        f"{params_million_model_2:<15.4f}"
        f"{total_time_model_2:<12.2f}\n")

# =========================
# 13.3. SO SÁNH ALEXNET VÀ VGG11
# =========================

f_model_2.write("\n")
f_model_2.write("=" * 120 + "\n")
f_model_2.write("SO SÁNH ALEXNET VÀ VGG11\n")
f_model_2.write("=" * 120 + "\n")

f_model_2.write(f"{'Model':<15}"
        f"{'Accuracy':<15}"
        f"{'Precision':<15}"
        f"{'Recall':<15}"
        f"{'F1':<15}"
        f"{'Params(M)':<15}"
        f"{'Time(s)':<15}\n")

f_model_2.write("-"*120 + "\n")

f_model_2.write(f"{model_name_1:<15}"
        f"{acc_model_1*100:<15.2f}"
        f"{precision_macro_model_1*100:<15.2f}"
        f"{recall_macro_model_1*100:<15.2f}"
        f"{f1_macro_model_1*100:<15.2f}"
        f"{params_million_model_1:<15.2f}"
        f"{total_time_model_1:<15.2f}\n")

f_model_2.write(f"{model_name_2:<15}"
        f"{acc_model_2*100:<15.2f}"
        f"{precision_macro_model_2*100:<15.2f}"
        f"{recall_macro_model_2*100:<15.2f}"
        f"{f1_macro_model_2*100:<15.2f}"
        f"{params_million_model_2:<15.2f}"
        f"{total_time_model_2:<15.2f}\n")


# =========================
# 13.4. KẾT LUẬN
# =========================

if acc_model_1 > acc_model_2:
    best_model = model_name_1
else:
    best_model = model_name_2

f_compare.write("\n")
f_compare.write("=" * 120 + "\n")
f_compare.write("KẾT LUẬN\n")
f_compare.write("=" * 120 + "\n")

f_compare.write(f"Accuracy cao nhất : {model_name_1 if acc_model_1 > acc_model_2 else model_name_2}\n")

f_compare.write(f"F1-score cao nhất : {model_name_1 if f1_macro_model_1 > f1_macro_model_2 else model_name_2}\n")

f_compare.write(f"Mô hình it tham số nhất   : {model_name_1 if total_params_model_1 < total_params_model_2 else model_name_2}\n")

f_compare.write(f"Mô hình nhanh nhất: {model_name_1 if total_time_model_1 < total_time_model_2 else model_name_2}\n")

f_compare.write(f"\nMô hình được đề xuất: {best_model}\n")


# =========================
# 14.1. VẼ QUÁ TRÌNH HỘI TỤ MÔ HÌNH ALEXNET
# =========================
fig1_model_1 = plt.figure(figsize=(8, 5))
plt.plot(history_model_1.history['loss'], '-b^', label='Training loss')
plt.plot(history_model_1.history['val_loss'], '-rv', label='Validation loss')

plt.ylabel('Loss',fontsize=16)
plt.xlabel('Epochs',fontsize=16)
plt.legend(loc='upper right')
plt.title('Convergence Process: Training Loss vs Validation Loss')
plt.grid(True)
plt.tight_layout()

fig2_model_1 = plt.figure(figsize=(8, 5))
plt.plot(history_model_1.history['accuracy'], '-b>', label='Training accuracy')
plt.plot(history_model_1.history['val_accuracy'], '-r<', label='Validation accuracy')
plt.ylabel('Accuracy',fontsize=16)
plt.xlabel('Epochs',fontsize=16)
plt.legend(loc='upper left')
plt.title('Convergence Process: Training Accuracy vs Validation Accuracy')
plt.grid(True)
plt.tight_layout()


# =========================
# 14.2. VẼ QUÁ TRÌNH HỘI TỤ MÔ HÌNH ALEXNET
# =========================
fig1_model_2 = plt.figure(figsize=(8, 5))
plt.plot(history_model_2.history['loss'], '-b^', label='Training loss')
plt.plot(history_model_2.history['val_loss'], '-rv', label='Validation loss')

plt.ylabel('Loss',fontsize=16)
plt.xlabel('Epochs',fontsize=16)
plt.legend(loc='upper right')
plt.title('Convergence Process: Training Loss vs Validation Loss')
plt.grid(True)
plt.tight_layout()

fig2_model_2 = plt.figure(figsize=(8, 5))
plt.plot(history_model_2.history['accuracy'], '-b>', label='Training accuracy')
plt.plot(history_model_2.history['val_accuracy'], '-r<', label='Validation accuracy')
plt.ylabel('Accuracy',fontsize=16)
plt.xlabel('Epochs',fontsize=16)
plt.legend(loc='upper left')
plt.title('Convergence Process: Training Accuracy vs Validation Accuracy')
plt.grid(True)
plt.tight_layout()

# =========================
# 15.1. LƯU HÌNH RA FILE MÔ HÌNH ALEXNET
# =========================

loss_plot_path_model_1 = os.path.join(BASE_DIR, "../results/image/alexnet/ALEXNET_loss.png")
acc_plot_path_model_1 = os.path.join(BASE_DIR, "../results/image/alexnet/ALEXNET_accuracy.png")
cm_plot_path_model_1 = os.path.join(BASE_DIR, "../results/image/alexnet/ALEXNET_matrix.png")

fig1_model_1.savefig(loss_plot_path_model_1, dpi=300, bbox_inches='tight')
fig2_model_1.savefig(acc_plot_path_model_1, dpi=300, bbox_inches='tight')
fig_cm_model_1.savefig(cm_plot_path_model_1, dpi=300, bbox_inches='tight')


print(f"Đã lưu các kết quả vào        : {report_txt_path_model_1}")
print(f"Đã lưu đồ thị loss vào        : {loss_plot_path_model_1}")
print(f"Đã lưu đồ thị accuracy vào    : {acc_plot_path_model_1}")
print(f"Đã lưu confusion matrix vào   : {cm_plot_path_model_1}")

# =========================
# 15.2. LƯU HÌNH RA FILE MÔ HÌNH VGG11
# =========================
loss_plot_path_model_2 = os.path.join(BASE_DIR, "../results/image/vgg11/VGG11_loss.png")
acc_plot_path_model_2 = os.path.join(BASE_DIR, "../results/image/vgg11/VGG11_accuracy.png")
cm_plot_path_model_2 = os.path.join(BASE_DIR, "../results/image/vgg11/VGG11_matrix.png")

fig1_model_2.savefig(loss_plot_path_model_2, dpi=300, bbox_inches='tight')
fig2_model_2.savefig(acc_plot_path_model_2, dpi=300, bbox_inches='tight')
fig_cm_model_2.savefig(cm_plot_path_model_2, dpi=300, bbox_inches='tight')


print(f"Đã lưu các kết quả vào        : {report_txt_path_model_2}")
print(f"Đã lưu đồ thị loss vào        : {loss_plot_path_model_2}")
print(f"Đã lưu đồ thị accuracy vào    : {acc_plot_path_model_2}")
print(f"Đã lưu confusion matrix vào   : {cm_plot_path_model_2}")

f_model_1.close()
f_model_2.close()
f_compare.close()
plt.show()
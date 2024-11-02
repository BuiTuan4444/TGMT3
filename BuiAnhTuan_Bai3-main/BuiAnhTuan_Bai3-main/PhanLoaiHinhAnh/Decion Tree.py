import cv2
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Hàm tải và tiền xử lý ảnh
def load_and_preprocess_image(image_path, size=(64, 64)):
    # Đọc ảnh từ đường dẫn
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Cannot load image at {image_path}")
        return None, None
    # Thay đổi kích thước ảnh
    resized_image = cv2.resize(image, size)
    # Chuyển ảnh sang màu xám
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # Làm phẳng ảnh
    return image, gray.flatten()

# Tạo tập dữ liệu
image_paths = ["cho1.jpg", "cho2.jpg", "meo1.jpg", "meo2.jpg"]
labels = ["dog", "dog", "cat", "cat"]

data, valid_labels, original_dogs, original_cats = [], [], [], []

for img, label in zip(image_paths, labels):
    original, processed_image = load_and_preprocess_image(img)
    if processed_image is not None:
        data.append(processed_image)  # Lưu ảnh đã xử lý
        valid_labels.append(label)

        # Phân loại ảnh gốc vào danh sách chó và mèo
        if label == "dog":
            original_dogs.append(original)
        else:
            original_cats.append(original)

# Kiểm tra kích thước của tất cả các ảnh trong data
for i, img in enumerate(data):
    print(f"Image {i} shape: {img.shape}")

# Mã hóa nhãn (Label Encoding)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(valid_labels)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(data, encoded_labels, test_size=0.3, random_state=42)

# Huấn luyện và đánh giá mô hình với Decision Tree
decision_tree = DecisionTreeClassifier()  # Sử dụng Decision Tree
start_time = time.time()
decision_tree.fit(X_train, y_train)
training_time = time.time() - start_time

# Thực hiện dự đoán trên tập kiểm tra
y_pred = decision_tree.predict(X_test)

# Tính toán các độ đo hiệu suất
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
recall = recall_score(y_test, y_pred, average='macro', zero_division=1)

# Báo cáo chi tiết hơn về hiệu suất
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=1))

# Hiển thị các kết quả
print("Hiệu suất của mô hình Decision Tree:")
print(f"Thời gian huấn luyện (s): {training_time:.4f}")
print(f"Độ chính xác: {accuracy:.4f}")
print(f"Độ chính xác (Precision): {precision:.4f}")
print(f"Độ thu hồi (Recall): {recall:.4f}")

# Hiển thị các ảnh gốc chó và mèo
plt.figure(figsize=(10, 5))

# Hiển thị ảnh chó
for i, original in enumerate(original_dogs):
    plt.subplot(2, len(original_dogs), i + 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title(f"Dog {i + 1}")
    plt.axis('off')

# Hiển thị ảnh mèo
for i, original in enumerate(original_cats):
    plt.subplot(2, len(original_cats), i + 1 + len(original_dogs))
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title(f"Cat {i + 1}")
    plt.axis('off')

plt.tight_layout()
plt.show()

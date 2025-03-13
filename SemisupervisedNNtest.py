import numpy as np
import struct
import os

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
import streamlit as st
import matplotlib.pyplot as plt

# Cấu Hình Streamlit
@st.cache_data  # Lưu cache để tránh load lại dữ liệu mỗi lần chạy lại Streamlit
def get_sampled_pixels(images, sample_size=100_000):
    return np.random.choice(images.flatten(), sample_size, replace=False)

@st.cache_data  # Cache danh sách ảnh ngẫu nhiên
def get_random_indices(num_images, total_images):
    return np.random.randint(0, total_images, size=num_images)
# Cấu hình Streamlit    
# st.set_page_config(page_title="Phân loại ảnh", layout="wide")
# Định nghĩa hàm để đọc file .idx
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    return images
def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels   

# Hàm để lấy 1% dữ liệu ban đầu cho mỗi class
def get_initial_train_data(x_train, y_train, percent=0.01):
    initial_x_train = []
    initial_y_train = []
    
    for class_label in range(10):
        class_indices = np.where(np.argmax(y_train, axis=1) == class_label)[0]
        num_samples = int(len(class_indices) * percent)
        selected_indices = np.random.choice(class_indices, num_samples, replace=False)
        
        initial_x_train.append(x_train[selected_indices])
        initial_y_train.append(y_train[selected_indices])
    
    initial_x_train = np.concatenate(initial_x_train)
    initial_y_train = np.concatenate(initial_y_train)
    
    return initial_x_train, initial_y_train

# Hàm để xây dựng model Neural Network
def build_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Hàm chính để thực hiện Pseudo Labelling
def pseudo_labelling(x_train, y_train, x_test, y_test, threshold, max_iter):
    # Lấy 1% dữ liệu ban đầu
    initial_x_train, initial_y_train = get_initial_train_data(x_train, y_train)
    
    # Huấn luyện model ban đầu
    model = build_model()
    history = model.fit(initial_x_train, initial_y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    
    # Hiển thị kết quả ban đầu
    st.write("### Kết quả huấn luyện ban đầu")
    st.write(f"Độ chính xác trên tập validation: {history.history['val_accuracy'][-1]:.4f}")
    
    # Lặp lại quá trình Pseudo Labelling
    for i in range(max_iter):
        st.write(f"### Lần lặp {i + 1}")
        
        # Dự đoán nhãn cho phần dữ liệu còn lại
        remaining_x_train = np.delete(x_train, np.where(np.isin(x_train, initial_x_train))[0], axis=0)
        predictions = model.predict(remaining_x_train)
        
        # Gán nhãn giả cho các mẫu có độ tin cậy cao
        pseudo_labels = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        high_confidence_indices = np.where(confidences > threshold)[0]
        
        pseudo_x_train = remaining_x_train[high_confidence_indices]
        pseudo_y_train = tf.keras.utils.to_categorical(pseudo_labels[high_confidence_indices], 10)
        
        # Cập nhật tập dữ liệu huấn luyện
        initial_x_train = np.concatenate([initial_x_train, pseudo_x_train])
        initial_y_train = np.concatenate([initial_y_train, pseudo_y_train])
        
        # Huấn luyện lại model
        model = build_model()
        history = model.fit(initial_x_train, initial_y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
        
        # Hiển thị kết quả
        st.write(f"Độ chính xác trên tập validation sau lần lặp {i + 1}: {history.history['val_accuracy'][-1]:.4f}")
        st.write(f"Số lượng mẫu được gán nhãn giả: {len(pseudo_x_train)}")
        
        # Hiển thị một số ảnh được gán nhãn giả
        st.write("Một số ảnh được gán nhãn giả:")
        fig, axes = plt.subplots(1, 5, figsize=(10, 2))
        for idx, ax in enumerate(axes):
            ax.imshow(pseudo_x_train[idx], cmap='gray')
            ax.set_title(f"Label: {np.argmax(pseudo_y_train[idx])}")
            ax.axis('off')
        st.pyplot(fig)
    
    # Đánh giá model trên tập test
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    st.write("### Kết quả cuối cùng trên tập test")
    st.write(f"Độ chính xác trên tập test: {test_acc:.4f}")

# Giao diện Streamlit
def run_PseudoLabellingt_app():
    # Định nghĩa đường dẫn đến các file MNIST
    dataset_path = os.path.dirname(os.path.abspath(__file__)) 
    train_images_path = os.path.join(dataset_path, "train-images.idx3-ubyte")
    train_labels_path = os.path.join(dataset_path, "train-labels.idx1-ubyte")
    test_images_path = os.path.join(dataset_path, "t10k-images.idx3-ubyte")
    test_labels_path = os.path.join(dataset_path, "t10k-labels.idx1-ubyte")

    # Tải dữ liệu
    train_images = load_mnist_images(train_images_path)
    train_labels = load_mnist_labels(train_labels_path)
    test_images = load_mnist_images(test_images_path)
    test_labels = load_mnist_labels(test_labels_path)

    # Chuyển đổi dữ liệu thành vector 1 chiều
    X = np.concatenate((train_images, test_images), axis=0)  # Gộp toàn bộ dữ liệu
    y = np.concatenate((train_labels, test_labels), axis=0)
    X = X.reshape(X.shape[0], -1)  # Chuyển thành vector 1 chiều

    # Cho phép người dùng chọn tỷ lệ validation và test
    test_size = st.slider("🔹 Chọn % tỷ lệ tập test", min_value=10, max_value=50, value=20, step=1) / 100
            
    x_train, y_train, x_test, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    st.title("Pseudo Labelling với Neural Network trên tập dữ liệu MNIST")
    
    # Thiết lập các tham số
    threshold = st.sidebar.slider("Ngưỡng quyết định", min_value=0.8, max_value=0.99, value=0.95, step=0.01)
    max_iter = st.sidebar.slider("Số lần lặp", min_value=1, max_value=10, value=5, step=1)
    
    # Bắt đầu quá trình Pseudo Labelling
    if st.button("Bắt đầu"):
        pseudo_labelling(x_train, y_train, x_test, y_test, threshold, max_iter)

# Chạy ứng dụng Streamlit
if __name__ == "__main__":
    run_PseudoLabellingt_app()
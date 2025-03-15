import time
import streamlit as st
import os
import numpy as np
import pandas as pd
import random
import mlflow
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
from mlflow.tracking import MlflowClient
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras import layers, optimizers
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow import keras




def preprocess_canvas_image(canvas_result):
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data[:, :, 0].astype(np.uint8))
        img = img.resize((28, 28)).convert("L")  # Resize và chuyển thành grayscale
        img = np.array(img, dtype=np.float32) / 255.0  # Chuẩn hóa về [0, 1]
        return img.reshape(1, -1)  # Chuyển thành vector 1D
    return None

def load_mnist_data():
    X = np.load("data/X_test.npy")
    y = np.load("data/y_test.npy")
    return X, y






def data_preparation():
    # Đọc dữ liệu
    X, y = load_mnist_data()
    X = X.reshape(X.shape[0], -1)  # Chuyển ảnh về vector 1D
    # total_samples = X.shape[0] 

    # Cho phép người dùng chọn tỷ lệ validation và test
    test_size = st.slider("🔹 Chọn % tỷ lệ tập test", min_value=10, max_value=50, value=20, step=1) / 100

    # Tạo vùng trống để hiển thị kết quả
    result_placeholder = st.empty()
    # Tạo nút "Lưu Dữ Liệu"
    if st.button("Xác Nhận & Lưu Dữ Liệu"):
        
        # Phân chia dữ liệu
        X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Lấy 1% số lượng ảnh cho mỗi class (0-9) để làm tập dữ liệu train ban đầu
        train_indices = []
        for i in range(10):
            class_indices = np.where(y_train_data == i)[0]
            num_samples = int(0.01 * len(class_indices))
            indices = np.random.choice(class_indices, num_samples, replace=False)
            train_indices.extend(indices)

        X_train_initial = X_train_data[train_indices]
        y_train_initial = y_train_data[train_indices]

        # Chuyển 99% còn lại sang tập val
        val_indices = np.setdiff1d(np.arange(len(X_train_data)), train_indices)
        X_val_data = X_train_data[val_indices]
        y_val_data = y_train_data[val_indices]


        # Tính tỷ lệ thực tế của từng tập
        total_samples = X.shape[0]
        test_percent = (X_test_data.shape[0] / total_samples) * 100
        train_percent = (X_train_initial.shape[0] / total_samples) * 100
        val_percent = (X_val_data.shape[0] / total_samples) * 100

        # Lưu dữ liệu vào session_state
        st.session_state["X_train"] = X_train_initial
        st.session_state["y_train"] = y_train_initial
        st.session_state["X_val"] = X_val_data
        st.session_state["y_val"] = y_val_data
        st.session_state["X_test"] = X_test_data
        st.session_state["y_test"] = y_test_data
        
        # # Ghi log cho quá trình phân chia dữ liệu
        # mlflow.log_param("test_size", test_size)
        # mlflow.log_metric("test_percent", test_percent)
        # mlflow.log_metric("train_percent", train_percent)
        # mlflow.log_metric("val_percent", val_percent)
        # with result_placeholder:
        # Hiển thị kết quả
        st.write(f"📊 **Tỷ lệ phân chia**: Test={test_percent:.0f}%, Train={train_percent:.0f}%, Val={val_percent:.0f}%")
        st.write("✅ Dữ liệu đã được xử lý và chia tách.")
        st.write(f"🔹 Kích thước tập huấn luyện ban đầu: `{X_train_initial.shape}`")
        st.write(f"🔹 Kích thước tập kiểm tra: `{X_test_data.shape}`")
        st.write(f"🔹 Kích thước tập validation: `{X_val_data.shape}`")

        # Tạo biểu đồ số lượng dữ liệu của mỗi nhãn trong tập train
        unique_labels, counts = np.unique(y_train_initial, return_counts=True)
        fig, ax = plt.subplots()
        ax.bar(unique_labels, counts)
        ax.set_xlabel('Nhãn')
        ax.set_ylabel('Số lượng')
        ax.set_title('Phân phối số lượng dữ liệu trong tập train')
        ax.set_xticks(unique_labels)
        st.pyplot(fig)





def learning_model():
    # Lấy dữ liệu từ session_state
    X_train = st.session_state["X_train"]
    X_val = st.session_state["X_val"]
    X_test = st.session_state["X_test"]
    y_train = st.session_state["y_train"]
    y_val = st.session_state["y_val"]
    y_test = st.session_state["y_test"]

    # Lựa chọn tham số huấn luyện
    num_k_folds = st.slider("Số fold cho Cross-Validation:", 3, 10, 5)
    num_layers = st.slider("Số lớp ẩn:", 1, 5, 2)
    epochs = st.slider("Số lần lặp tối đa", 2, 50, 5)
    learning_rate_init = st.slider("Tốc độ học", 0.001, 0.1, 0.01, step=0.001, format="%.3f")
    threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
    iteration = st.slider("Số lần lặp tối đa", 1, 10, 5)
    activation = st.selectbox("Hàm kích hoạt:", ["relu", "sigmoid", "tanh"])
    num_neurons = st.selectbox("Số neuron mỗi lớp:", [32, 64, 128, 256], index=0)
    optimizer = st.selectbox("Chọn hàm tối ưu", ["adam", "sgd", "lbfgs"])
    loss_fn = "sparse_categorical_crossentropy"
    run_name = st.text_input("🔹 Nhập tên Run:", "Default_Run")
    st.session_state['run_name'] = run_name

    if st.button("⏹️ Huấn luyện mô hình"):
        with st.spinner("🔄 Đang huấn luyện..."):
            mlflow.start_run(run_name=run_name)

            # Ghi log các tham số
            mlflow.log_params({
                "num_layers": num_layers,
                "num_neurons": num_neurons,
                "activation": activation,
                "optimizer": optimizer,
                "k_folds": num_k_folds,
                "epochs": epochs,
                "learning_rate": learning_rate_init,
                "threshold": threshold,
                "max_iterations": iteration
            })

            # Tạo mô hình Neural Network
            cnn = keras.Sequential([
                layers.Input(shape=(X_train.shape[1],)),
                *[layers.Dense(num_neurons, activation=activation) for _ in range(num_layers)],
                layers.Dense(10, activation="softmax")
            ])

            # Chọn optimizer
            if optimizer == "adam":
                opt = keras.optimizers.Adam(learning_rate=learning_rate_init)
            elif optimizer == "sgd":
                opt = keras.optimizers.SGD(learning_rate=learning_rate_init)
            else:
                opt = keras.optimizers.RMSprop(learning_rate=learning_rate_init)

            cnn.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])

            # Khởi tạo các biến
            accuracies, losses = [], []
            start_time = time.time()
            iteration_count = 0

            if len(X_train) == 0 or len(y_train) == 0:
                st.error("❌ Tập huấn luyện rỗng!")
                mlflow.end_run()
                return
            # Vòng lặp huấn luyện với Pseudo-Labeling
            for iter_idx in range(iteration):
                iteration_count += 1
                st.write(f"**Lần lặp thứ {iteration_count}:**")
                k_folds = min(num_k_folds, max(2, int(5000 / len(X_train))))
                progress_bar = st.progress(0)
                progress_text = st.empty()

                # Cross-validation với StratifiedKFold
                kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
                fold_count = 0
                for i, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
                    fold_count += 1
                    X_k_train, X_k_val = X_train[train_idx], X_train[val_idx]
                    y_k_train, y_k_val = y_train[train_idx], y_train[val_idx]

                    # Huấn luyện mô hình
                    history = cnn.fit(
                        X_k_train, y_k_train,
                        epochs=epochs,
                        validation_data=(X_k_val, y_k_val),
                        verbose=0,  # Tắt log chi tiết để tránh làm chậm Streamlit
                        batch_size=num_neurons  # Thêm batch_size để tối ưu hóa hiệu suất
                    )

                    accuracies.append(history.history["val_accuracy"][-1])
                    losses.append(history.history["val_loss"][-1])

                    # Cập nhật tiến trình
                    progress = i
                    progress_bar.progress(progress)
                    progress_text.text(f"️🎯 Tiến trình huấn luyện: {int(progress * 100)}%")

                # Dự đoán trên tập validation để gán nhãn giả (Pseudo-Labeling)
                if len(X_val) > 0:
                    y_pred = cnn.predict(X_val, verbose=0)
                    y_pred_class = np.argmax(y_pred, axis=1)
                    max_probs = np.max(y_pred, axis=1)

                    # Gán nhãn giả dựa trên ngưỡng
                    pseudo_labels = np.where(max_probs >= threshold, y_pred_class, -1)
                    confident_mask = max_probs >= threshold

                    X_confident = X_val[confident_mask]
                    y_confident = pseudo_labels[confident_mask]

                    # Cập nhật tập huấn luyện và tập validation
                    if len(X_confident) > 0:
                        X_train = np.concatenate((X_train, X_confident), axis=0)
                        y_train = np.concatenate((y_train, y_confident), axis=0)
                        X_val = X_val[~confident_mask]
                        y_val = y_val[~confident_mask]
                    else:
                        st.write(f"🔹 Vòng {iter_idx + 1}: Không có mẫu nào vượt ngưỡng {threshold}. Dừng lại.")
                        break

                    # Điều kiện dừng
                    if len(X_val) == 0:
                        st.write("✅ Đã gán nhãn hết tập unlabeled!")
                        break
                    elif len(X_confident) < 10:
                        st.write(f"🔹 Vòng {iter_idx + 1}: Số mẫu gán nhãn quá ít ({len(X_confident)}). Dừng lại.")
                        break
                else:
                    st.write("✅ Không còn dữ liệu unlabeled để gán nhãn!")
                    break

            # Kết thúc huấn luyện
            elapsed_time = time.time() - start_time
            avg_val_accuracy = np.mean(accuracies) if accuracies else 0
            avg_val_loss = np.mean(losses) if losses else 0

            # Đánh giá trên tập test
            test_loss, test_accuracy = cnn.evaluate(X_test, y_test, verbose=0)

            # Ghi log kết quả
            mlflow.log_metrics({
                "avg_val_accuracy": avg_val_accuracy,
                "avg_val_loss": avg_val_loss,
                "test_accuracy": test_accuracy,
                "test_loss": test_loss,
                "elapsed_time": elapsed_time
            })

            mlflow.end_run()

            # Lưu mô hình và hiển thị kết quả
            st.session_state["selected_model_type"] = "Neural Network"
            st.session_state["trained_model"] = cnn
            st.success(f"✅ Huấn luyện hoàn tất trong {elapsed_time:.2f} giây!")
            st.write(f"📊 **Độ chính xác trên tập test:** {test_accuracy:.4f}")

            # Vẽ biểu đồ Loss và Accuracy
            st.markdown("#### 📈 Biểu đồ Accuracy và Loss")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(history.history['loss'], label='Train Loss', color='blue')
            ax1.plot(history.history['val_loss'], label='Test Loss', color='orange')
            ax1.set_title('Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()

            ax2.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
            ax2.plot(history.history['val_accuracy'], label='Test Accuracy', color='orange')
            ax2.set_title('Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()

            st.pyplot(fig)








def run_PseudoLabellingt_app():
    

    # @st.cache_data  # Lưu cache để tránh load lại dữ liệu mỗi lần chạy lại Streamlit
    # def get_sampled_pixels(images, sample_size=100_000):
    #     return np.random.choice(images.flatten(), sample_size, replace=False)

    # @st.cache_data  # Cache danh sách ảnh ngẫu nhiên
    # def get_random_indices(num_images, total_images):
    #     return np.random.randint(0, total_images, size=num_images)

    # # Cấu hình Streamlit    
    # # st.set_page_config(page_title="Phân loại ảnh", layout="wide")
    # # Định nghĩa hàm để đọc file .idx
    # def load_mnist_images(filename):
    #     with open(filename, 'rb') as f:
    #         magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
    #         images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    #     return images

    # def load_mnist_labels(filename):
    #     with open(filename, 'rb') as f:
    #         magic, num = struct.unpack('>II', f.read(8))
    #         labels = np.fromfile(f, dtype=np.uint8)
    #     return labels

    mlflow_tracking_uri = st.secrets["MLFLOW_TRACKING_URI"]
    mlflow_username = st.secrets["MLFLOW_TRACKING_USERNAME"]
    mlflow_password = st.secrets["MLFLOW_TRACKING_PASSWORD"]
    
    # Thiết lập biến môi trường
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
    os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password
    
    # Thiết lập MLflow (Đặt sau khi mlflow_tracking_uri đã có giá trị)
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Đọc dữ liệu
    X_data, y_data = load_mnist_data()

    # total_samples = X.shape[0] # Tổng số mẫu

    # # Định nghĩa đường dẫn đến các file MNIST
    # dataset_path = os.path.dirname(os.path.abspath(__file__)) 
    # train_images_path = os.path.join(dataset_path, "train-images.idx3-ubyte")
    # train_labels_path = os.path.join(dataset_path, "train-labels.idx1-ubyte")
    # test_images_path = os.path.join(dataset_path, "t10k-images.idx3-ubyte")
    # test_labels_path = os.path.join(dataset_path, "t10k-labels.idx1-ubyte")

    # # Tải dữ liệu
    # train_images = load_mnist_images(train_images_path)
    # train_labels = load_mnist_labels(train_labels_path)
    # test_images = load_mnist_images(test_images_path)
    # test_labels = load_mnist_labels(test_labels_path)

    # Giao diện Streamlit
    st.title("📸 Phân loại ảnh MNIST với Streamlit")
    tabs = st.tabs([
        "Thông tin dữ liệu",
        "Thông tin",
        "Chuẩn bị dữ liệu",
        "Huấn luyện mô hình",
        "Demo dự đoán file ảnh",
        "Demo dự đoán Viết Tay",
        "Thông tin & Mlflow",
    ])
    # tab_info, tab_load, tab_preprocess, tab_split,  tab_demo, tab_log_info = tabs
    tab_info,tab_note, tab_data ,tab_preprocess,  tab_demo, tab_demo_2 ,tab_mlflow= tabs






    # with st.expander("🖼️ Dữ liệu ban đầu", expanded=True):
    with tab_info:
        with st.expander("**Thông tin dữ liệu**", expanded=True):
            st.markdown(
                '''
                **MNIST** là phiên bản được chỉnh sửa từ bộ dữ liệu NIST gốc của Viện Tiêu chuẩn và Công nghệ Quốc gia Hoa Kỳ.  
                Bộ dữ liệu ban đầu gồm các chữ số viết tay từ nhân viên bưu điện và học sinh trung học.  

                Các nhà nghiên cứu **Yann LeCun, Corinna Cortes, và Christopher Burges** đã xử lý, chuẩn hóa và chuyển đổi bộ dữ liệu này thành **MNIST** để dễ dàng sử dụng hơn cho các bài toán nhận dạng chữ số viết tay.
                '''
            )
            # image = Image.open(r'C:\Users\Dell\OneDrive\Pictures\Documents\Code\python\OpenCV\HMVPYTHON\App\image.png')

            # Gắn ảnh vào Streamlit và chỉnh kích thước
            # st.image(image, caption='Mô tả ảnh', width=600) 
            # Đặc điểm của bộ dữ liệu
        with st.expander("**Đặc điểm của bộ dữ liệu**", expanded=True):
            st.markdown(
                '''
                - **Số lượng ảnh:** 70.000 ảnh chữ số viết tay  
                - **Kích thước ảnh:** Mỗi ảnh có kích thước 28x28 pixel  
                - **Cường độ điểm ảnh:** Từ 0 (màu đen) đến 255 (màu trắng)  
                - **Dữ liệu nhãn:** Mỗi ảnh đi kèm với một nhãn số từ 0 đến 9  
                '''
            )
            st.write(f"🔍 Số lượng ảnh huấn luyện: `{X_data.shape[0]}`")
            st.write(f"🔍 Số lượng ảnh kiểm tra: `{y_data.shape[0]}`")

        with st.expander("**Hiển thị số lượng mẫu của từng chữ số từ 0 đến 9 trong tập huấn luyện**", expanded=True):
            label_counts = pd.Series(y_data).value_counts().sort_index()

            # # Hiển thị biểu đồ cột
            # st.subheader("📊 Biểu đồ số lượng mẫu của từng chữ số")
            # st.bar_chart(label_counts)

            # Hiển thị bảng dữ liệu dưới biểu đồ
            st.subheader("📋 Số lượng mẫu cho từng chữ số")
            df_counts = pd.DataFrame({"Chữ số": label_counts.index, "Số lượng mẫu": label_counts.values})
            st.dataframe(df_counts)


            st.subheader("Chọn ngẫu nhiên 10 ảnh từ tập huấn luyện để hiển thị")
            num_images = 10
            random_indices = random.sample(range(len(X_data)), num_images)
            fig, axes = plt.subplots(1, num_images, figsize=(10, 5))

            for ax, idx in zip(axes, random_indices):
                ax.imshow(X_data[idx], cmap='gray')
                ax.axis("off")
                ax.set_title(f"Label: {y_data[idx]}")

            st.pyplot(fig)
        with st.expander("**Kiểm tra hình dạng của tập dữ liệu**", expanded=True):    
            # Kiểm tra hình dạng của tập dữ liệu
            st.write("🔍 Hình dạng tập dữ liệu:", X_data.shape)


    with tab_note:
        with st.expander("**Thông tin mô hình**", expanded=True):
        # Assume model_option1 is selected from somewhere in the app
            st.markdown("""
                    ### Neural Network (NN)
                    """) 
            st.markdown("---")        
            st.markdown("""            
            ### Khái Niệm:  
            **Neural Network (NN)**:
            - Là một mô hình tính toán lấy cảm hứng từ cấu trúc và chức năng của mạng lưới thần kinh sinh học. Nó được tạo thành từ các nút kết nối với nhau, hay còn gọi là nơ-ron nhân tạo, được sắp xếp thành các lớp.
            - Ý tưởng chính của **Neural Network** là tạo ra một mô hình tính toán có khả năng học hỏi và xử lý thông tin giống như bộ não con người.
            """)

            st.markdown("---")        
                       
            st.write("### Mô Hình Tổng Quát:")   
            st.image("imgnn/modelnn.png",use_container_width ="auto")
            st.markdown(""" 
            - Layer đầu tiên là input layer, các layer ở giữa được gọi là hidden layer, layer cuối cùng được gọi là output layer. Các hình tròn được gọi là node.
            - Mỗi mô hình luôn có 1 input layer, 1 output layer, có thể có hoặc không các hidden layer. Tổng số layer trong mô hình được quy ước là số layer - 1 (Không tính input layer).
            - Mỗi node trong hidden layer và output layer :
                - Liên kết với tất cả các node ở layer trước đó với các hệ số w riêng.
                - Mỗi node có 1 hệ số bias b riêng.
                - Diễn ra 2 bước: tính tổng linear và áp dụng activation function.
            """)

            st.markdown("---")          
            st.markdown("""
            ### Nguyên lý hoạt động:  
            - Dữ liệu đầu vào được đưa vào lớp đầu vào.
            - Mỗi nơ-ron trong lớp ẩn nhận tín hiệu từ các nơ-ron ở lớp trước đó, xử lý tín hiệu và chuyển tiếp kết quả đến các nơ-ron ở lớp tiếp theo.
            - Quá trình này tiếp tục cho đến khi dữ liệu đến lớp đầu ra.
            - Kết quả đầu ra được tạo ra dựa trên các tín hiệu nhận được từ lớp ẩn cuối cùng.
            """)           
            st.markdown("---")
            st.markdown("""  
            ### Áp dụng vào ngữ cảnh Neural Network với MNIST:  
            - **MNIST (Modified National Institute of Standards and Technology database)** là một bộ dữ liệu kinh điển trong lĩnh vực học máy, đặc biệt là trong việc áp dụng mạng nơ-ron. Nó bao gồm 70.000 ảnh xám của chữ số viết tay (từ 0 đến 9), được chia thành 60.000 ảnh huấn luyện và 10.000 ảnh kiểm tra.
            - Mục tiêu của bài toán là phân loại chính xác chữ số từ 0 đến 9 dựa trên ảnh đầu vào.
            - Có nhiều cách để áp dụng mạng nơ-ron cho bài toán phân loại chữ số viết tay trên MNIST. Dưới đây là một số phương pháp phổ biến:
                - **Multi-Layer Perceptron (MLP)**: Một mô hình mạng nơ-ron sâu với nhiều lớp ẩn.
                - **Convolutional Neural Network (CNN)**: Một mô hình mạng nơ-ron sâu được thiết kế đặc biệt cho việc xử lý ảnh.
                - **Recurrent Neural Network (RNN)**: Một mô hình mạng nơ-ron sâu được thiết kế cho dữ liệu chuỗi.
            """)

    


    # Chuẩn bị dữ liệu
    with tab_data:
        with st.expander("**Chuẩn bị dữ liệu**", expanded=True):
            st.write("🔍 **Chuẩn bị dữ liệu cho mô hình**")
            data_preparation()  
        





    # 3️⃣ HUẤN LUYỆN MÔ HÌNH
    with tab_preprocess:
        with st.expander("**Huấn luyện Neural Network**", expanded=True):
        
            # Cập nhật dữ liệu
            learning_model()
                
                    





    with tab_demo:
        with st.expander("**Dự đoán kết quả**", expanded=True):
            st.write("**Dự đoán trên ảnh do người dùng tải lên**")
            if "trained_model" not in st.session_state:
                st.warning("⚠️ Chưa có mô hình nào được huấn luyện.")
            else:
                best_model = st.session_state["trained_model"]
                st.write(f"Mô hình đang sử dụng: Neural Network")

                uploaded_file = st.file_uploader("📂 Chọn một ảnh", type=["png", "jpg", "jpeg"])
                if uploaded_file:
                    image = Image.open(uploaded_file).convert("L")
                    image = np.array(image.resize((28, 28)), dtype=np.float32) / 255.0  # Normalize to [0, 1]
                    image = image.reshape(1, -1)  # Shape: (1, 784)

                    prediction = best_model.predict(image)[0]
                    st.image(uploaded_file, caption="📷 Ảnh đã tải lên", width=150)
                    st.success(f"Dự đoán: {np.argmax(prediction)} với xác suất {np.max(prediction):.2f}")







    with tab_demo_2:   
        with st.expander("**Dự đoán kết quả**", expanded=True):
            st.write("**Dự đoán trên ảnh do người dùng tải lên**")

            # Kiểm tra xem mô hình đã được huấn luyện và lưu kết quả chưa
            if "selected_model_type" not in st.session_state or "trained_model" not in st.session_state:
                st.warning("⚠️ Chưa có mô hình nào được huấn luyện. Vui lòng huấn luyện mô hình trước khi dự đoán.")
            else:
                best_model_name = st.session_state.selected_model_type
                best_model = st.session_state.trained_model

                st.write(f"Mô hình đang sử dụng: `{best_model_name}`")
                # st.write(f"✅ Độ chính xác trên tập kiểm tra: `{st.session_state.get('test_accuracy', 'N/A'):.4f}`")

                # 🆕 Cập nhật key cho canvas khi nhấn "Tải lại"
                # if "key_value" not in st.session_state:
                #     st.session_state.key_value = str(random.randint(0, 1000000))

                # if st.button("🔄 Tải lại"):
                #     try:
                #         st.session_state.key_value = str(random.randint(0, 1000000))
                #     except Exception as e:
                #         st.error(f"Cập nhật key không thành công: {str(e)}")
                #         st.stop()
                st.session_state.key_value = str(random.randint(0, 1000000))

                # ✍️ Vẽ dữ liệu
                canvas_result = st_canvas(
                    fill_color="black",
                    stroke_width=10,
                    stroke_color="white",
                    background_color="black",
                    height=150,
                    width=150,
                    drawing_mode="freedraw",
                    key=st.session_state.key_value,
                    update_streamlit=True
                ) 

                if st.button("Dự đoán"):
                    img = preprocess_canvas_image(canvas_result)

                    if img is not None:
                        X_train = st.session_state["X_train"]
                        # Hiển thị ảnh sau xử lý
                        st.image(Image.fromarray((img.reshape(28, 28) * 255).astype(np.uint8)), caption="Ảnh sau xử lý", width=100)

                        # Dự đoán
                        prediction = best_model.predict(img)[0]

                        st.success(f"Dự đoán: {np.argmax(prediction)} với xác suất {np.max(prediction)*100:.2f}%")
                    else:
                        st.error("⚠️ Hãy vẽ một số trước khi bấm Dự đoán!")







    with tab_mlflow:
        st.header("Thông tin Huấn luyện & MLflow UI")
        try:
            client = MlflowClient()
            experiment_name = "Classification"
    
            # Kiểm tra nếu experiment đã tồn tại
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = client.create_experiment(experiment_name)
                st.success(f"Experiment mới được tạo với ID: {experiment_id}")
            else:
                experiment_id = experiment.experiment_id
                st.info(f"Đang sử dụng experiment ID: {experiment_id}")
    
            mlflow.set_experiment(experiment_name)
    
            # Truy vấn các run trong experiment
            runs = client.search_runs(experiment_ids=[experiment_id])
    
            # 1) Chọn và đổi tên Run Name
            st.subheader("Đổi tên Run")
            if runs:
                run_options = {run.info.run_id: f"{run.data.tags.get('mlflow.runName', 'Unnamed')} - {run.info.run_id}"
                               for run in runs}
                selected_run_id_for_rename = st.selectbox("Chọn Run để đổi tên:", 
                                                          options=list(run_options.keys()), 
                                                          format_func=lambda x: run_options[x])
                new_run_name = st.text_input("Nhập tên mới cho Run:", 
                                             value=run_options[selected_run_id_for_rename].split(" - ")[0])
                if st.button("Cập nhật tên Run"):
                    if new_run_name.strip():
                        client.set_tag(selected_run_id_for_rename, "mlflow.runName", new_run_name.strip())
                        st.success(f"Đã cập nhật tên Run thành: {new_run_name.strip()}")
                    else:
                        st.warning("Vui lòng nhập tên mới cho Run.")
            else:
                st.info("Chưa có Run nào được log.")
    
            # 2) Xóa Run
            st.subheader("Danh sách Run")
            if runs:
                selected_run_id_to_delete = st.selectbox("", 
                                                         options=list(run_options.keys()), 
                                                         format_func=lambda x: run_options[x])
                if st.button("Xóa Run", key="delete_run"):
                    client.delete_run(selected_run_id_to_delete)
                    st.success(f"Đã xóa Run {run_options[selected_run_id_to_delete]} thành công!")
                    st.experimental_rerun()  # Tự động làm mới giao diện
            else:
                st.info("Chưa có Run nào để xóa.")
    
            # 3) Danh sách các thí nghiệm
            st.subheader("Danh sách các Run đã log")
            if runs:
                selected_run_id = st.selectbox("Chọn Run để xem chi tiết:", 
                                               options=list(run_options.keys()), 
                                               format_func=lambda x: run_options[x])
    
                # 4) Hiển thị thông tin chi tiết của Run được chọn
                selected_run = client.get_run(selected_run_id)
                st.write(f"**Run ID:** {selected_run_id}")
                st.write(f"**Run Name:** {selected_run.data.tags.get('mlflow.runName', 'Unnamed')}")
    
                st.markdown("### Tham số đã log")
                st.json(selected_run.data.params)
    
                st.markdown("### Chỉ số đã log")
                metrics = {
                    "mean_cv_accuracy": selected_run.data.metrics.get("mean_cv_accuracy", "N/A"),
                    "std_cv_accuracy": selected_run.data.metrics.get("std_cv_accuracy", "N/A"),
                    "accuracy": selected_run.data.metrics.get("accuracy", "N/A"),
                    "model_type": selected_run.data.metrics.get("model_type", "N/A"),
                    "kernel": selected_run.data.metrics.get("kernel", "N/A"),
                    "C_value": selected_run.data.metrics.get("C_value", "N/A")
                
                }
                st.json(metrics)
    
                # 5) Nút bấm mở MLflow UI
                st.subheader("Truy cập MLflow UI")
                mlflow_url = "https://dagshub.com/quangdinhhusc/HMVPYTHON.mlflow"
                if st.button("Mở MLflow UI"):
                    st.markdown(f'**[Click để mở MLflow UI]({mlflow_url})**')
            else:
                st.info("Chưa có Run nào được log. Vui lòng huấn luyện mô hình trước.")
    
        except Exception as e:
            st.error(f"Không thể kết nối với MLflow: {e}")

    




if __name__ == "__main__":
    run_PseudoLabellingt_app()
    
    
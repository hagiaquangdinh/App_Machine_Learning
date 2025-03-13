import streamlit as st

# Import các hàm từ ứng dụng con
from ClusteringMinsttest import run_ClusteringMinst_app
from ClassificationMinsttest import run_ClassificationMinst_app
from LinearRegressiontest import run_LinearRegression_app  
from PcaTSNEMinsttest import run_PcaTSNEMinst_app
from NeuralNetworkMNSITtest import run_NeuralNetwork_app
from SemisupervisedNNtest import run_PseudoLabellingt_app

# Cấu hình trang chính - phải được gọi ngay đầu file
st.set_page_config(page_title="App Machine Learning", page_icon="💻", layout="wide")

# Sidebar chứa menu ứng dụng
st.sidebar.title("Home page")
app_choice = st.sidebar.radio(
    "Chọn ứng dụng:",
    ["Linear Regression", "Classification", "Clustering","PCA_T-SNE", "Neural Network", "Pseudo Labelling"]
)


# Nội dung chính của trang
st.title("Chương Trình Ứng Dụng")

# Điều hướng đến ứng dụng được chọn
if app_choice == "Linear Regression":
    run_LinearRegression_app()
elif app_choice == "Classification":
    run_ClassificationMinst_app()
elif app_choice == "Clustering":
    run_ClusteringMinst_app()
elif app_choice == "PCA_T-SNE":
    run_PcaTSNEMinst_app()
elif app_choice == "Neural Network":
    run_NeuralNetwork_app()
elif app_choice == "Pseudo Labelling":
    run_PseudoLabellingt_app()

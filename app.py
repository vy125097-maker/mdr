import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. CẤU HÌNH TRANG WEB
# ==========================================
st.set_page_config(
    page_title="Hệ thống Chẩn đoán Lao & Kháng thuốc MDR-TB",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Ứng Dụng AI Dự Đoán Chủng Vi Khuẩn Lao Kháng Đa Thuốc (MDR-TB)")
st.markdown("---")

# ==========================================
# 2. LOAD MÔ HÌNH (BẢN TẠM MẪU CHẠY NGAY)
# ==========================================
@st.cache_resource
def load_resnet_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.eval()
    return model

@st.cache_resource
def load_rf_model():
    model = RandomForestClassifier()
    # Khởi tạo dữ liệu mẫu để mô hình chạy demo
    X_dummy = np.random.rand(10, 7)
    y_dummy = np.random.randint(0, 2, 10)
    model.fit(X_dummy, y_dummy)
    return model

resnet_model = load_resnet_model()
rf_model = load_rf_model()
st.sidebar.success("Đã khởi tạo mô hình AI thành công!")

# ==========================================
# 3. GIAO DIỆN NHẬP DỮ LIỆU
# ==========================================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Ảnh X-quang Phổi")
    uploaded_file = st.file_uploader("Chọn ảnh X-quang (PNG, JPG, JPEG)...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Ảnh X-quang đã tải lên", use_container_width=True)

with col2:
    st.subheader("Thông Tin Bệnh Nhân & Lâm Sàng")
    with st.form("patient_info_form"):
        age = st.number_input("Tuổi bệnh nhân:", min_value=1, max_value=120, value=45)
        gender = st.selectbox("Giới tính:", ["Nam", "Nữ"])
        previous_treatment = st.selectbox("Tiền sử điều trị lao trước đây:", ["Chưa từng", "Đã từng điều trị"])
        bmi = st.number_input("Chỉ số BMI:", min_value=10.0, max_value=50.0, value=18.5)
        smoking = st.selectbox("Tiền sử hút thuốc:", ["Không", "Có"])
        hiv_status = st.selectbox("Tình trạng HIV:", ["Âm tính", "Dương tính"])
        
        gender_num = 1 if gender == "Nam" else 0
        prev_treat_num = 1 if previous_treatment == "Đã từng điều trị" else 0
        smoking_num = 1 if smoking == "Có" else 0
        hiv_num = 1 if hiv_status == "Dương tính" else 0

        submit_btn = st.form_submit_button("🚀 Tiến Hành Dự Đoán")

# ==========================================
# 4. XỬ LÝ & DỰ ĐOÁN
# ==========================================
if submit_btn:
    if uploaded_file is None:
        st.warning("⚠️ Vui lòng upload ảnh X-quang trước khi dự đoán!")
    else:
        st.markdown("---")
        st.subheader("Kết Quả Chẩn Đoán Tích Hợp AI")
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = resnet_model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            tb_prob = probabilities[0][1].item()

        features = np.array([[age, gender_num, prev_treat_num, bmi, smoking_num, hiv_num, tb_prob]])
        
        mdr_prediction = rf_model.predict(features)[0]
        mdr_prob = rf_model.predict_proba(features)[0][1]

        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric("Tỷ lệ nghi ngờ mắc Lao (ResNet18)", f"{tb_prob * 100:.2f}%")
            if tb_prob > 0.5:
                st.error("⚠️ Khả năng cao tổn thương phổi do Lao!")
            else:
                st.success("✅ Hình ảnh X-quang bình thường.")

        with res_col2:
            st.metric("Tỷ lệ Kháng Đa Thuốc MDR-TB (Random Forest)", f"{mdr_prob * 100:.2f}%")
            if mdr_prediction == 1 or mdr_prob > 0.5:
                st.error("🚨 CẢNH BÁO: Nguy cơ cao kháng thuốc MDR-TB!")
            else:
                st.success("✅ Nguy cơ kháng thuốc thấp.")

        st.info("**Lưu ý:** Kết quả dự đoán từ AI chỉ mang tính chất tham khảo hỗ trợ bác sĩ lâm sàng.")
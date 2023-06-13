import pandas as pd
import streamlit as st
import gdown
import io
from surprise import Dataset, Reader, NMF

# Link tải dữ liệu từ Google Drive
data_url = 'https://drive.google.com/uc?id=1MHLvwXQMgRKz9BMYqNE-NxPVUfoEmoYJ'

# Tải dữ liệu từ link Google Drive trực tiếp vào DataFrame
file_id = data_url.split('/')[-2]
csv_url = f'https://drive.google.com/uc?id={file_id}'
csv_file = gdown.download(csv_url, quiet=False)

# Đọc dữ liệu từ tệp CSV
data = pd.read_csv(csv_file)

# Tạo một đối tượng Reader để định dạng dữ liệu
reader = Reader(rating_scale=(1, 5))

# Tạo một đối tượng Dataset từ DataFrame
dataset = Dataset.load_from_df(data[['reviewerID', 'asin', 'overall']], reader)

# Xây dựng mô hình NMF với số lượng yếu tố latents = 10
model = NMF(n_factors=10)

# Đào tạo mô hình trên toàn bộ dữ liệu
trainset = dataset.build_full_trainset()
model.fit(trainset)

# Lấy ID người dùng đầu vào từ người dùng
user_id = st.text_input("Nhập ID người dùng:")
k = st.number_input("Nhập số lượng sản phẩm khuyến nghị:", min_value=1, step=1)

if st.button("Khuyến nghị"):
    # Lấy danh sách sản phẩm chưa được người dùng đánh giá
    user = trainset.to_inner_uid(user_id)
    items_to_recommend = [(trainset.to_raw_iid(iid), model.predict(uid=user, iid=iid).est) for iid in trainset.all_items() if iid not in trainset.ur[user]]

    # Sắp xếp dự đoán theo xếp hạng giảm dần
    top_k_predictions = sorted(items_to_recommend, key=lambda x: x[1], reverse=True)[:k]

    # Hiển thị danh sách sản phẩm được khuyến nghị cùng với điểm số
    recommended_df = pd.DataFrame(top_k_predictions, columns=['asin', 'rating'])
    recommended_df = recommended_df.merge(data[['asin', 'reviewerID']], on='asin', how='left')

    st.write("Top", k, "sản phẩm được khuyến nghị:")
    st.table(recommended_df[['asin', 'rating']])

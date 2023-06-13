# Import các thư viện cần thiết
import streamlit as st
import gdown
import pandas as pd
from surprise import Dataset, Reader, NMF
import io

# Link tải dữ liệu
data_url = 'https://drive.google.com/uc?id=1351xVuTBwyqnKVzpHbN5tbKD1rBeMpgQ'

# Tải dữ liệu trực tiếp vào DataFrame
url = gdown.download(data_url, quiet=False)
data = pd.read_csv(io.StringIO(url))
st.dataframe(data)

# # Xử lý dữ liệu
# ratings_df = data[['reviewerID', 'asin', 'overall']]
# reader = Reader(rating_scale=(1, 5))
# data = Dataset.load_from_df(ratings_df, reader)

# # Tạo tập huấn luyện
# trainset = data.build_full_trainset()

# # Xây dựng mô hình NMF
# model = NMF()
# model.fit(trainset)

# # Streamlit UI
# st.title('Hệ thống khuyến nghị sản phẩm')

# user_id = st.text_input("Nhập mã người dùng:")
# k = st.slider("Số sản phẩm khuyến nghị:", min_value=1, max_value=20, value=5)

# if st.button("Khuyến nghị"):
#     if user_id:
#         # Lấy danh sách các sản phẩm chưa được người dùng này đánh giá
#         iids = data['asin'].unique()
#         iids_unrated = [iid for iid in iids if iid not in [x for x, _ in trainset.ir[trainset.to_inner_uid(user_id)]]]

#         # Dự đoán điểm đánh giá cho các sản phẩm chưa được đánh giá
#         predictions = [model.predict(user_id, iid) for iid in iids_unrated]

#         # Sắp xếp các sản phẩm theo điểm đánh giá dự đoán
#         predictions.sort(key=lambda x: x.est, reverse=True)

#         # Hiển thị top k sản phẩm được khuyến nghị
#         recommended_items = [pred.iid for pred in predictions[:k]]
#         st.write("Sản phẩm được khuyến nghị:", recommended_items)

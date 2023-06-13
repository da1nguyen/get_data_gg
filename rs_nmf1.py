import pandas as pd
import streamlit as st
import requests
import io
from surprise import Dataset, Reader, NMF
from surprise.model_selection import cross_validate

# Khai báo URL dữ liệu
data_url = 'https://drive.google.com/uc?id=1IXbptj9A5VD-yHh8I_70SZcv2hi8NY2e'

# Yêu cầu dữ liệu từ URL
response = requests.get(data_url)

# Kiểm tra xem có lỗi trong quá trình tải dữ liệu không
assert response.status_code == 200, 'Could not download the data'

# Đọc dữ liệu vào DataFrame
data = pd.read_csv(io.StringIO(response.content.decode('utf-8')))

# Hiển thị DataFrame
st.dataframe(data)

# Xây dựng mô hình khuyến nghị
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(data[['reviewerID', 'asin', 'overall']], reader)
model = NMF(n_factors=10)
cross_validate(model, dataset, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Nhập thông tin từ người dùng
user_id = st.text_input("Nhập ID người dùng:")
k = st.text_input("Nhập số lượng sản phẩm khuyến nghị:")
k = int(k) if k.isnumeric() else 0

# Đào tạo mô hình trên toàn bộ dữ liệu
trainset = dataset.build_full_trainset()
model.fit(trainset)

# Lấy danh sách sản phẩm chưa được người dùng đánh giá
items_to_recommend = [iid for iid in trainset.all_items() if iid not in trainset.ur[user_id]]

# Dự đoán xếp hạng cho sản phẩm chưa được đánh giá
predictions = [model.predict(user_id, iid) for iid in items_to_recommend]

# Sắp xếp dự đoán theo xếp hạng giảm dần
top_k_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:k]

# Hiển thị danh sách sản phẩm được khuyến nghị
recommended_items = [pred.iid for pred in top_k_predictions]
recommended_df = data[data['asin'].isin(recommended_items)]
recommended_df = recommended_df[['asin', 'overall']]
recommended_df = recommended_df.drop_duplicates()

st.write(f"Top {k} sản phẩm được khuyến nghị:")
st.dataframe(recommended_df)

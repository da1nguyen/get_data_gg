# Khai báo các thư viện
import pandas as pd
import streamlit as st
import requests
import io

# Chỉ ra dữ liệu (ở đây chứa một file csv)
data_url = 'https://drive.google.com/uc?id=1MHLvwXQMgRKz9BMYqNE-NxPVUfoEmoYJ'

# Yêu cầu dữ liệu từ link kết url trên
response = requests.get(data_url)

# Kiểm tra xem link có thể nhận về trực tiếp hay không
assert response.status_code == 200, 'Could not download the data'

# Đọc dữ liệu vào DataFrame
data = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
pd.set_option('display.max_colwidth', None)

# Chọn chỉ mục của các cột muốn hiển thị
selected_columns = [1, 2, 4, 5, 7]  

# Hiển thị DataFrame với các cột đã chọn
st.dataframe(data.iloc[:, selected_columns])

from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
from surprise import NMF

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

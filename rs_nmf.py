import requests
import os
import tempfile

def download_file(url):
    local_filename = url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return f.name

data_path = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Musical_Instruments_5.json.gz'

# Tải file về máy
local_path = download_file(data_path)

# Tải dữ liệu
df = get_df(local_path)

# Xóa file tạm sau khi đã sử dụng
os.remove(local_path)

# Tải dữ liệu
df = get_df(data_path)
import streamlit as st
from surprise import Dataset, Reader, NMF
from surprise.model_selection import train_test_split

# Xử lý dữ liệu
ratings_df = df[['reviewerID', 'asin', 'overall']]
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df, reader)

# Tạo tập huấn luyện
trainset = data.build_full_trainset()

# Xây dựng mô hình NMF
model = NMF()
model.fit(trainset)

# Streamlit UI
st.title('Hệ thống khuyến nghị sản phẩm')

user_id = st.text_input("Nhập mã người dùng:")
k = st.slider("Số sản phẩm khuyến nghị:", min_value=1, max_value=20, value=5)

if st.button("Khuyến nghị"):
    if user_id:
        # Lấy danh sách các sản phẩm chưa được người dùng này đánh giá
        iids = df['asin'].unique()
        iids_unrated = [iid for iid in iids if not trainset.knows_item(iid)]

        # Dự đoán điểm đánh giá cho các sản phẩm chưa được đánh giá
        predictions = [model.predict(user_id, iid) for iid in iids_unrated]

        # Sắp xếp các sản phẩm theo điểm đánh giá dự đoán
        predictions.sort(key=lambda x: x.est, reverse=True)

        # Hiển thị top k sản phẩm được khuyến nghị
        recommended_items = [pred.iid for pred in predictions[:k]]
        st.write("Sản phẩm được khuyến nghị:", recommended_items)
    else:
        st.write("Vui lòng nhập mã người dùng.")

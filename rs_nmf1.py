import pandas as pd
import streamlit as st
import requests
import io
from surprise import Dataset, Reader, NMF

# Chỉ ra dữ liệu (ở đây chứa một file csv)
data_url = 'https://drive.google.com/uc?id=1MHLvwXQMgRKz9BMYqNE-NxPVUfoEmoYJ'

# Yêu cầu dữ liệu từ link kết url trên
response = requests.get(data_url)

# Kiểm tra xem link có thể nhận về trực tiếp hay không
assert response.status_code == 200, 'Could not download the data'

# Đọc dữ liệu vào DataFrame
data = pd.read_csv(io.StringIO(response.content.decode('utf-8')))

# Khai báo các tên cột trong DataFrame
column_mapping = {
    'reviewerID': 'userID',
    'asin': 'ProductID',
    'overall': 'rating'
}

# Đổi tên các cột trong DataFrame
data = data.rename(columns=column_mapping)

# Tạo một đối tượng Reader để định dạng dữ liệu
reader = Reader(rating_scale=(1, 5))

# Tạo một đối tượng Dataset từ DataFrame
dataset = Dataset.load_from_df(data[['userID', 'ProductID', 'rating']], reader)

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
    items_to_recommend = [iid for uid, iid, _ in trainset.build_anti_testset() if uid == user_id]

    # Dự đoán xếp hạng cho sản phẩm chưa được đánh giá
    predictions = model.test([(user_id, iid, 0) for iid in items_to_recommend])

    # Sắp xếp dự đoán theo xếp hạng giảm dần
    top_k_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:k]

    # Hiển thị danh sách sản phẩm được khuyến nghị cùng với điểm số
    recommended_items = [(pred.iid, pred.est) for pred in top_k_predictions]
    recommended_df = pd.DataFrame(recommended_items, columns=['ProductID', 'Score'])
    recommended_df = recommended_df.merge(data[['ProductID', 'itemName']], on='ProductID', how='left')
    st.write("Top", k, "sản phẩm được khuyến nghị:")
    st.dataframe(recommended_df)

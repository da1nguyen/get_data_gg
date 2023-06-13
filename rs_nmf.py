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

# # Tạo một đối tượng Reader để định dạng dữ liệu
# reader = Reader(rating_scale=(1, 5))

# # Tạo một đối tượng Dataset từ DataFrame
# dataset = Dataset.load_from_df(data[['userID', 'itemID', 'rating']], reader)

# # Xây dựng mô hình NMF với số lượng yếu tố latents = 10
# model = NMF(n_factors=10)

# # Đào tạo mô hình trên dữ liệu
# cross_validate(model, dataset, measures=['RMSE', 'MAE'], cv=5, verbose=True)
# # Lấy ID người dùng đầu vào từ người dùng
# user_id = st.text_input("Nhập ID người dùng:")
# k = int(st.text_input("Nhập số lượng sản phẩm khuyến nghị:"))

# # Đào tạo mô hình trên toàn bộ dữ liệu
# trainset = dataset.build_full_trainset()
# model.fit(trainset)

# # Lấy danh sách sản phẩm chưa được người dùng đánh giá
# items_to_recommend = trainset.build_anti_testset().for_user(user_id)

# # Dự đoán xếp hạng cho sản phẩm chưa được đánh giá
# predictions = model.test(items_to_recommend)

# # Sắp xếp dự đoán theo xếp hạng giảm dần
# top_k_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:k]

# # Hiển thị danh sách sản phẩm được khuyến nghị
# recommended_items = [pred.iid for pred in top_k_predictions]
# st.write("Top", k, "sản phẩm được khuyến nghị:")
# st.write(data[data['itemID'].isin(recommended_items)][['itemID', 'itemName']])

# import pandas as pd
# import streamlit as st
# from surprise import Dataset, Reader, NMF

# # Khai báo URL dữ liệu
# data_url = 'https://drive.google.com/uc?id=1IXbptj9A5VD-yHh8I_70SZcv2hi8NY2e'

# # Đọc dữ liệu từ URL
# data = pd.read_csv(data_url)

# # Chọn các cột reviewerID, asin và overall
# data = data[['reviewerID', 'asin', 'overall']]

# # Hiển thị DataFrame trên Streamlit
# st.write(data)

# # Tạo một đối tượng Reader để định dạng dữ liệu
# reader = Reader(rating_scale=(1, 5))

# # Tạo một đối tượng Dataset từ DataFrame
# dataset = Dataset.load_from_df(data, reader)

# # Xây dựng mô hình NMF với số lượng yếu tố latents = 10
# model = NMF(n_factors=10)

# # Đào tạo mô hình trên toàn bộ dữ liệu
# trainset = dataset.build_full_trainset()
# model.fit(trainset)

# # Lấy ID người dùng đầu vào từ người dùng
# user_id = st.text_input("Nhập ID người dùng:")
# k = st.number_input("Nhập số lượng sản phẩm khuyến nghị:", min_value=1, max_value=len(data), value=3)

# # Xử lý khi nút được nhấn
# if st.button("Khuyến nghị"):
#     # Kiểm tra xem ID người dùng có tồn tại trong dữ liệu không
#     if user_id not in data['reviewerID'].unique():
#         st.error("ID người dùng không hợp lệ.")
#     else:
#         # Lấy danh sách các sản phẩm mà người dùng đã đánh giá
#         rated_items = data[data['reviewerID'] == user_id]['asin'].tolist()

#         # Lấy danh sách sản phẩm chưa được người dùng đánh giá
#         items_to_recommend = list(set(data['asin'].unique()) - set(rated_items))

#         # Dự đoán xếp hạng cho sản phẩm chưa được đánh giá
#         predictions = [model.predict(user_id, item_id) for item_id in items_to_recommend]

#         # Sắp xếp dự đoán theo xếp hạng giảm dần
#         top_k_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:k]

#         # Hiển thị danh sách sản phẩm được khuyến nghị
#         recommended_items = [pred.iid for pred in top_k_predictions]
#         recommended_df = data[data['asin'].isin(recommended_items)]
#         st.write("Top", k, "sản phẩm được khuyến nghị:")
#         st.write(recommended_df)
import pandas as pd
import streamlit as st
import requests
import io
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
from surprise import NMF
from scipy.spatial.distance import cosine

# Khai báo URL dữ liệu
data_url = 'https://drive.google.com/uc?id=1MHLvwXQMgRKz9BMYqNE-NxPVUfoEmoYJ'

# Yêu cầu dữ liệu từ URL
response = requests.get(data_url)

# Kiểm tra xem có lỗi trong quá trình tải dữ liệu không
assert response.status_code == 200, 'Could not download the data'

# Đọc dữ liệu vào DataFrame
data = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
pd.set_option('display.max_colwidth', None)

# Chọn chỉ mục của các cột muốn hiển thị
selected_columns = ['reviewerID', 'asin', 'overall']

# Hiển thị DataFrame với các cột đã chọn
st.dataframe(data[selected_columns])

# Tạo một đối tượng Reader để định dạng dữ liệu
reader = Reader(rating_scale=(1, 5))

# Tạo một đối tượng Dataset từ DataFrame
dataset = Dataset.load_from_df(data[['reviewerID', 'asin', 'overall']], reader)

# Xây dựng mô hình NMF với số lượng yếu tố latents = 10
model = NMF(n_factors=10)

# Đào tạo mô hình trên dữ liệu
cross_validate(model, dataset, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Lấy mã người dùng đầu vào từ người dùng
user_id = st.text_input("Nhập mã người dùng:")
k = int(st.text_input("Nhập số lượng sản phẩm khuyến nghị:"))

# Đào tạo mô hình trên toàn bộ dữ liệu
trainset = dataset.build_full_trainset()
model.fit(trainset)

# Tìm top k sản phẩm tương tự dựa trên mã sản phẩm đầu vào
item_id = st.text_input("Nhập mã sản phẩm:")
similar_items = []
item_index = trainset.to_inner_iid(item_id)
item_vector = model.qi[item_index]

for i in range(trainset.n_items):
    if i != item_index:
        other_item_vector = model.qi[i]
        similarity = 1 - cosine(item_vector, other_item_vector)
        similar_items.append((trainset.to_raw_iid(i), similarity))

similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)[:k]

# Hiển thị danh sách sản phẩm tương tự
st.write("Top", k, "sản phẩm tương tự cho mã sản phẩm", item_id)
for item in similar_items:
    st.write(item[0])


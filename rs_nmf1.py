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
# Khai báo các thư viện
# Khai báo các thư viện
import pandas as pd
import streamlit as st
import requests
import io
import numpy as np

# Khai báo các thư viện
import pandas as pd
import streamlit as st
import requests
import io
import numpy as np

# Khai báo URL dữ liệu
data_url = 'https://drive.google.com/uc?id=1MHLvwXQMgRKz9BMYqNE-NxPVUfoEmoYJ'

# Yêu cầu dữ liệu từ URL
response = requests.get(data_url)

# Kiểm tra xem có lỗi trong quá trình tải dữ liệu không
assert response.status_code == 200, 'Could not download the data'

# Đọc dữ liệu vào DataFrame
data = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
pd.set_option('display.max_colwidth', None)

# Lấy danh sách các sản phẩm
item_list = data['asin'].unique()

# Hiển thị danh sách các sản phẩm
st.write("Danh sách các sản phẩm:")
st.write(item_list)

# Nhập mã sản phẩm để tìm các sản phẩm tương tự
item_id = st.text_input("Nhập mã sản phẩm:")

if st.button("Xử lý"):
    if item_id in item_list:
        # Tạo một DataFrame con chỉ chứa các đánh giá của sản phẩm được chọn
        item_data = data[data['asin'] == item_id][['reviewerID', 'asin', 'overall']]

        # Tạo một ma trận đánh giá sản phẩm
        item_matrix = pd.pivot_table(item_data, values='overall', index='reviewerID', columns='asin', fill_value=0)

        # Tính độ tương tự cosine giữa các sản phẩm
        similarity_matrix = np.dot(item_matrix, item_matrix.T)

        # Lấy chỉ số của sản phẩm được chọn
        item_index = item_data.index[0]

        # Tìm top k sản phẩm tương tự
        similar_items = similarity_matrix[item_index].argsort()[::-1][1:k+1]

        # Hiển thị danh sách sản phẩm tương tự
        similar_item_data = data.loc[similar_items][['asin', 'overall']]
        st.write("Top", k, "sản phẩm tương tự:")
        st.write(similar_item_data)
    else:
        st.write("Mã sản phẩm không hợp lệ!")


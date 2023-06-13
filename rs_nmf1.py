import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Khai báo dữ liệu
data_url = 'https://drive.google.com/uc?id=1IXbptj9A5VD-yHh8I_70SZcv2hi8NY2e'

# Yêu cầu dữ liệu từ URL
data = pd.read_csv(data_url)

# Chọn cột reviewerID, asin, overall
data = data[['reviewerID', 'asin', 'overall']]

# Tạo ma trận đánh giá
rating_matrix = data.pivot_table(index='reviewerID', columns='asin', values='overall')

# Tính độ tương đồng cosine giữa các sản phẩm
similarity_matrix = cosine_similarity(rating_matrix.T)

# Tạo ứng dụng Streamlit
st.title("Hệ thống khuyến nghị sản phẩm")

# Nhập mã sản phẩm
item_id = st.text_input("Nhập mã sản phẩm:")

# Số lượng sản phẩm khuyến nghị
k = st.number_input("Nhập số lượng sản phẩm khuyến nghị:", min_value=1, max_value=10, step=1)

# Tìm index của sản phẩm trong ma trận đánh giá
item_index = data[data['asin'] == item_id].index[0]

# Tìm top k sản phẩm tương tự
similar_items = similarity_matrix[item_index].argsort()[::-1][1:k+1]

# Lấy thông tin sản phẩm khuyến nghị
recommended_items = data.loc[similar_items, 'asin']

# Hiển thị danh sách sản phẩm khuyến nghị
st.write(f"Top {k} sản phẩm tương tự:")
st.write(recommended_items)

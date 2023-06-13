import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Khai báo URL dữ liệu
data_url = 'https://drive.google.com/uc?id=1IXbptj9A5VD-yHh8I_70SZcv2hi8NY2e'

# Yêu cầu dữ liệu từ URL
data = pd.read_csv(data_url)

# Lấy danh sách mã sản phẩm
items = data['asin'].unique()

# Hiển thị danh sách mã sản phẩm để chọn
# selected_item = st.selectbox("Chọn mã sản phẩm:", items)
selected_item = st.selectbox("Chọn mã sản phẩm:", items, index=0)
# Lấy chỉ số của sản phẩm được chọn
item_index = data[data['asin'] == selected_item].index[0]

# Tính ma trận tương đồng cosine
item_features = data.pivot(index='asin', columns='reviewerID', values='overall').fillna(0)
similarity_matrix = cosine_similarity(item_features)

# Nhập số lượng sản phẩm khuyến nghị
k = st.number_input("Nhập số lượng sản phẩm khuyến nghị:", value=5, min_value=1, step=1)

# Tìm top k sản phẩm tương tự
similar_items = similarity_matrix[item_index].argsort()[::-1][1:k+1]

# Tạo DataFrame kết quả
result_df = pd.DataFrame({
'Mã sản phẩm': data[data.index.isin(similar_items)]['asin'],
'Độ tương đồng': similarity_matrix[item_index][similar_items]
})

# Hiển thị danh sách sản phẩm khuyến nghị
st.write("Danh sách sản phẩm khuyến nghị:")
st.write(result_df)

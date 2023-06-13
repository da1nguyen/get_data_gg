# Khai báo các thư viện
import pandas as pd
import streamlit as st
import requests
import io
from surprise import Dataset, Reader, NMF
from sklearn.metrics.pairwise import cosine_similarity

# Chỉ ra dữ liệu (ở đây chứa một file csv)
data_url = 'https://drive.google.com/uc?id=1MHLvwXQMgRKz9BMYqNE-NxPVUfoEmoYJ'


# Yêu cầu dữ liệu từ link kết url trên
response = requests.get(data_url)

# Kiểm tra xem link có thể nhận về trực tiếp hay không
assert response.status_code == 200, 'Could not download the data'

# Đọc dữ liệu vào DataFrame
data = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
pd.set_option('display.max_colwidth', None)

st.dataframe(data[['reviewerID','asin','overall','reviewText']].head())

# Lấy danh sách mã sản phẩm
items = data['asin'].unique()

# Hiển thị danh sách mã sản phẩm để chọn
selected_item = st.selectbox("Chọn mã sản phẩm:", items, index=0, format_func=lambda item: item[:10])

# Lấy chỉ số của sản phẩm được chọn
item_index = data[data['asin'] == selected_item].index[0]

# Tính ma trận tương đồng cosine
item_features = data.pivot(index='asin', columns='reviewerID', values='overall').fillna(0)
similarity_matrix = cosine_similarity(item_features)

# Nhập số lượng sản phẩm khuyến nghị
k = st.number_input("Nhập số lượng sản phẩm khuyến nghị:", value=5, min_value=1, step=1)

# Tìm top k sản phẩm tương tự

import matplotlib.pyplot as plt

# Tạo DataFrame kết quả
result_df = pd.DataFrame({
    'Mã sản phẩm': data[data.index.isin(similar_items)]['asin'],
    'Độ tương đồng': similarity_matrix[item_index][similar_items]
}).set_index('Mã sản phẩm')

# Hiển thị danh sách sản phẩm khuyến nghị
st.write("Danh sách sản phẩm khuyến nghị:")
st.write(result_df)

# Vẽ biểu đồ
st.write("Biểu đồ độ tương đồng:")
fig, ax = plt.subplots()
result_df.plot(kind='barh', legend=False, ax=ax)
plt.xlabel('Độ tương đồng')
plt.tight_layout()
st.pyplot(fig)

result_df = pd.DataFrame({
    'Mã sản phẩm': data[data.index.isin(similar_items)]['asin'],
    'Độ tương đồng': similarity_matrix[item_index][similar_items]
})

# Hiển thị danh sách sản phẩm khuyến nghị
st.write("Danh sách sản phẩm khuyến nghị:")
st.write(result_df)

import matplotlib.pyplot as plt

# ...

# Tạo DataFrame kết quả
result_df = pd.DataFrame({
    'Mã sản phẩm': data[data.index.isin(similar_items)]['asin'],
    'Độ tương đồng': similarity_matrix[item_index][similar_items]
}).set_index('Mã sản phẩm')

# Hiển thị danh sách sản phẩm khuyến nghị
st.write("Danh sách sản phẩm khuyến nghị:")
st.write(result_df)

# Vẽ biểu đồ
st.write("Biểu đồ độ tương đồng:")
fig, ax = plt.subplots()
result_df.plot(kind='barh', legend=False, ax=ax)
plt.xlabel('Độ tương đồng')
plt.tight_layout()
st.pyplot(fig)

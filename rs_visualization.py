import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
import gdown

import pandas as pd
import streamlit as st
import requests
import io
from surprise import Dataset, Reader, NMF

# Khai báo các thư viện

# Chỉ ra dữ liệu (ở đây chứa một file csv)
data_url = 'https://drive.google.com/uc?id=1MHLvwXQMgRKz9BMYqNE-NxPVUfoEmoYJ'

# Yêu cầu dữ liệu từ link kết url trên
response = requests.get(data_url)

# Kiểm tra xem link có thể nhận về trực tiếp hay không
assert response.status_code == 200, 'Could not download the data'

# Đọc dữ liệu vào DataFrame
data = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
pd.set_option('display.max_colwidth', None)

# Tạo ma trận đánh giá
R = data.values

# Thực hiện NMF
model = NMF(n_components=5, init='random', random_state=0)
W = model.fit_transform(R)
H = model.components_

# Trực quan hóa ma trận W và H
st.title('NMF Decomposition')
st.write('W Matrix')
st.dataframe(pd.DataFrame(W))
st.write('H Matrix')
st.dataframe(pd.DataFrame(H))

# Xác định danh sách người dùng và sản phẩm
users = df.iloc[:, 0]  # Giả sử cột đầu tiên là ID người dùng
products = df.columns[1:]  # Giả sử các cột tiếp theo là ID sản phẩm

# Dự đoán đánh giá
predicted_ratings = np.dot(W, H)
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=products, index=users)

# Chọn người dùng từ thanh bên
st.sidebar.title('Recommendation')
selected_user = st.sidebar.selectbox('Select a User', users)

# Khuyến nghị sản phẩm dựa trên phương pháp item-item
similarities = np.dot(H.T, H)
item_similarity_df = pd.DataFrame(similarities, index=products, columns=products)

selected_user_ratings = predicted_ratings_df.loc[selected_user]
similar_items = item_similarity_df[selected_user_ratings.idxmax()].sort_values(ascending=False).head(6)

st.write('Top 5 Similar Products for User\'s Highest Rated Product')
st.dataframe(similar_items)

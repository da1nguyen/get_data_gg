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
from sklearn.metrics.pairwise import cosine_similarity

# Khai báo URL dữ liệu
data_url = 'https://drive.google.com/uc?id=1MHLvwXQMgRKz9BMYqNE-NxPVUfoEmoYJ'

# Yêu cầu dữ liệu từ URL
data = pd.read_csv(data_url)

# Hiển thị danh sách mã sản phẩm để người dùng chọn
product_ids = data['asin'].unique()
selected_product_id = st.selectbox("Chọn mã sản phẩm:", product_ids)

# Lọc dữ liệu dựa trên mã sản phẩm được chọn
selected_product_data = data[data['asin'] == selected_product_id]

# Tạo DataFrame chứa thông tin các sản phẩm tương tự
similar_products_df = pd.DataFrame(columns=['asin', 'title', 'similarity'])

# Tính độ tương tự cosine giữa sản phẩm được chọn và các sản phẩm khác
for index, row in data.iterrows():
    if row['asin'] != selected_product_id:
        similarity = cosine_similarity(selected_product_data['features'].values.reshape(1, -1), row['features'].reshape(1, -1))[0][0]
        similar_products_df = similar_products_df.append({'asin': row['asin'], 'title': row['title'], 'similarity': similarity}, ignore_index=True)

# Sắp xếp các sản phẩm tương tự theo độ tương tự giảm dần
similar_products_df = similar_products_df.sort_values(by='similarity', ascending=False)

# Lấy top k sản phẩm tương tự
k = st.slider("Chọn số lượng sản phẩm tương tự:", min_value=1, max_value=len(similar_products_df), value=5)
top_k_similar_products = similar_products_df.head(k)

# Hiển thị danh sách sản phẩm tương tự
st.write("Top", k, "sản phẩm tương tự cho mã sản phẩm", selected_product_id)
st.dataframe(top_k_similar_products[['asin', 'title', 'similarity']])


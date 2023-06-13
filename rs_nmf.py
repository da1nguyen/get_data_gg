import pandas as pd
import streamlit as st
import requests
import io
import json

# ID của file trên Google Drive
file_id = '1351xVuTBwyqnKVzpHbN5tbKD1rBeMpgQ'

# Link tải dữ liệu
data_url = f'https://drive.google.com/uc?export=download&id={file_id}'

# Tải dữ liệu từ URL
response = requests.get(data_url)

# Đảm bảo rằng tải dữ liệu thành công
assert response.status_code == 200, 'Could not download the data'

# Phân tích dữ liệu JSON
data = json.loads(response.content.decode('utf-8'))

# Chuyển đổi dữ liệu JSON thành DataFrame
df = pd.json_normalize(data)

# Hiển thị DataFrame
st.dataframe(df)

import pandas as pd
import streamlit as st
import requests
import io

# ID của file trên Google Drive
file_id = '1351xVuTBwyqnKVzpHbN5tbKD1rBeMpgQ'

# Link tải dữ liệu
data_url = f'https://drive.google.com/uc?export=download&id={file_id}'

# Tải dữ liệu từ URL
response = requests.get(data_url)

# Đảm bảo rằng tải dữ liệu thành công
assert response.status_code == 200, 'Could not download the data'

# Tạo một đối tượng file-like từ dữ liệu thu được
data = pd.read_json(io.StringIO(response.content.decode('utf-8')))

# Hiển thị DataFrame
st.dataframe(data)

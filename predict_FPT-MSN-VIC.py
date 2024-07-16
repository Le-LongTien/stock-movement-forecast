import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from matplotlib.dates import YearLocator, DateFormatter, MonthLocator

# Bước 1: Đọc và xử lý dữ liệu
df = pd.read_csv('FPT.csv')

# Chuyển đổi cột Date/Time về định dạng datetime
df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%m/%d/%Y %H:%M')

# Tạo cột biến động giá cổ phiếu sau 1 giờ (60 phút)
df = df.sort_values(by='Date/Time').reset_index(drop=True)
df['Close_N'] = df['Close'].shift(-60)
df['Price_Change'] = df['Close_N'] - df['Close']

# Loại bỏ các hàng có giá trị NaN
df = df.dropna(subset=['Price_Change'])

# Chỉ sử dụng các cột cần thiết cho mô hình
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change']
df_features = df[features]
df_price_change = df[['Price_Change']]

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(df_features)
scaled_price_change = scaler.fit_transform(df_price_change)

# Tạo dữ liệu train và test
lookback = 50  # Số lượng thời gian quá khứ được sử dụng để dự đoán tương lai
X, y = [], []
for i in range(lookback, len(scaled_features) - 60):
    X.append(scaled_features[i-lookback:i])
    y.append(scaled_price_change[i])  # "Price_Change" là biến phụ thuộc

X = np.array(X)
y = np.array(y)

# Chia thành train và test set
split_ratio = 0.6
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Bước 2: Xây dựng và huấn luyện mô hình CNN + LSTM
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(loss='mean_squared_error', optimizer='adam')

save_model = "best_model.keras"
best_model = ModelCheckpoint(save_model, monitor='loss', verbose=2, save_best_only=True, mode='auto')
model.fit(X_train, y_train, epochs=3, batch_size=50, verbose=2, callbacks=[best_model])

#Bước 3: Dự đoán và đánh giá mô hình
final_model = load_model(save_model)
y_train_predict = final_model.predict(X_train)
y_test_predict = final_model.predict(X_test)

#inverse transform cho Price_Change
scaler_price = MinMaxScaler(feature_range=(0, 1))
scaler_price.fit(df[['Price_Change']])
y_train = scaler_price.inverse_transform(y_train)
y_train_predict = scaler_price.inverse_transform(y_train_predict)
y_test = scaler_price.inverse_transform(y_test)
y_test_predict = scaler_price.inverse_transform(y_test_predict)
# R2
print('Độ phù hợp tập train:', r2_score(y_train, y_train_predict))
print('Độ phù hợp tập test:', r2_score(y_test, y_test_predict))
# MAE
print('Sai số tuyệt đối trung bình trên tập train (VNĐ):', mean_absolute_error(y_train, y_train_predict))
print('Sai số tuyệt đối trung bình trên tập test (VNĐ):', mean_absolute_error(y_test, y_test_predict))
# MAPE
print('Phần trăm sai số tuyệt đối trung bình tập train:', mean_absolute_percentage_error(y_train, y_train_predict))
print('Phần trăm sai số tuyệt đối trung bình tập test:', mean_absolute_percentage_error(y_test, y_test_predict))

# Vẽ biểu đồ so sánh kết quả
# Biểu đồ biến động trên tập huấn luyện
train_data_index = df['Date/Time'].iloc[lookback:split_index+lookback]
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(train_data_index, y_train, label='Biến động thực tế (Train)', color='blue')
ax.plot(train_data_index, y_train_predict, label='Biến động dự đoán (Train)', color='green')
ax.set_title('Biến động Thực tế và Dự đoán - Tập Huấn luyện')
ax.set_xlabel('Thời gian')
ax.set_ylabel('Biến động')
ax.legend()
plt.show()

# Biểu đồ biến động trên tập kiểm tra
test_data_index = df['Date/Time'].iloc[split_index+lookback:split_index+lookback+len(y_test_predict)]
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(test_data_index, y_test, label='Biến động thực tế (Test)', color='blue')
ax.plot(test_data_index, y_test_predict, label='Biến động dự đoán (Test)', color='red')
ax.set_title('Biến động Thực tế và Dự đoán - Tập Kiểm tra')
ax.set_xlabel('Thời gian')
ax.set_ylabel('Biến động')
ax.legend()
plt.show()

# Biểu đồ lỗi dự đoán trên tập train
train_errors = y_train - y_train_predict
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(train_data_index, train_errors, color='blue')
ax.set_title('Lỗi dự đoán trên tập huấn luyện')
ax.set_xlabel('Thời gian')
ax.set_ylabel('Lỗi')
plt.show()

# Biểu đồ lỗi dự đoán trên tập test
test_errors = y_test - y_test_predict
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(test_data_index, test_errors, color='red')
ax.set_title('Lỗi dự đoán trên tập kiểm tra')
ax.set_xlabel('Thời gian')
ax.set_ylabel('Lỗi')
plt.show()

# Biểu đồ phân phối lỗi trên tập train
fig, ax = plt.subplots(figsize=(14, 8))
sns.histplot(train_errors, bins=50, kde=True, color='blue', ax=ax)
ax.set_title('Phân phối của lỗi trên tập huấn luyện')
ax.set_xlabel('Lỗi')
ax.set_ylabel('Tần suất')
plt.show()

# Biểu đồ phân phối lỗi trên tập test
fig, ax = plt.subplots(figsize=(14, 8))
sns.histplot(test_errors, bins=50, kde=True, color='red', ax=ax)
ax.set_title('Phân phối của lỗi trên tập kiểm tra')
ax.set_xlabel('Lỗi')
ax.set_ylabel('Tần suất')
plt.show()


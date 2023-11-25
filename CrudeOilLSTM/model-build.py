# Load in libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

# Load in the data
price_data = pd.read_csv('newpricedata.csv')
forecast_data = pd.read_csv('newforecastdata.csv')

# Extract and preprocess data
price_dates = price_data.iloc[:, 0].values
price_prices = price_data.iloc[:, 1].values

# create a list of the dates in forecast_data to be predicted
forecast_dates = forecast_data.iloc[:,0].values

# make the dates in price_dates and forecast_dates datetime objects using the datetime library in month/day/year format
price_dates = [datetime.strptime(date, '%m/%d/%Y') for date in price_dates]
forecast_dates = [datetime.strptime(date, '%m/%d/%Y') for date in forecast_dates]

# convert the datetime objects in price_dates and forecast_dates to ordinal numbers
price_dates = [date.toordinal() for date in price_dates]
forecast_dates = [date.toordinal() for date in forecast_dates]

# convert the lists to numpy arrays
price_dates = np.array(price_dates)
price_prices = np.array(price_prices)

# reshape the numpy arrays to be 2D
price_dates = price_dates.reshape(-1, 1)
price_prices = price_prices.reshape(-1, 1)

# convert the lists to numpy arrays
forecast_dates = np.array(forecast_dates)

# reshape the numpy arrays to be 2D
forecast_dates = forecast_dates.reshape(-1, 1)

# split the data into training and testing data
price_dates_train, price_dates_test, price_prices_train, price_prices_test = train_test_split(price_dates, price_prices, test_size=0.2, random_state=0)

# scale the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
price_dates_train_scaled = scaler.fit_transform(price_dates_train)
price_prices_train_scaled = scaler.fit_transform(price_prices_train)
price_dates_test_scaled = scaler.fit_transform(price_dates_test)
price_prices_test_scaled = scaler.fit_transform(price_prices_test)

# reshape the data to be 3D
price_dates_train_scaled = np.reshape(price_dates_train_scaled, (price_dates_train_scaled.shape[0], price_dates_train_scaled.shape[1], 1))
price_dates_test_scaled = np.reshape(price_dates_test_scaled, (price_dates_test_scaled.shape[0], price_dates_test_scaled.shape[1], 1))

# Model configuration
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(price_dates_train_scaled.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Training the model and saving loss and RMSE values
epochs = 1000
loss_values = []
rmse_values = []

for epoch in range(epochs):
    history = model.fit(price_dates_train_scaled, price_prices_train_scaled, epochs=1, batch_size=32, verbose=0)
    loss = history.history['loss'][0]
    loss_values.append(loss)

    if (epoch + 1) % 10 == 0:  # Calculate RMSE every 10 epochs
        predicted_prices = model.predict(price_dates_test_scaled)
        predicted_prices = scaler.inverse_transform(predicted_prices)
        rmse = sqrt(mean_squared_error(price_prices_test_scaled, predicted_prices))
        rmse_values.append(rmse)

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}')

# Plot the loss and RMSE values
plt.figure(figsize=(11,8))
plt.plot(range(1, epochs + 1), loss_values, color='green', label='Loss')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Loss (MSE) vs. Epoch', fontsize=18)
plt.legend()

# plot the real prices and the predicted prices
plt.figure(figsize=(11,8))
plt.plot(price_prices_test, color='red', label='Real Prices')
plt.plot(predicted_prices, color='blue', label='Predicted Prices')

# save the model
model.save('oil_price_model.h5')


# calculate the r2 score
from sklearn.metrics import r2_score
r2 = r2_score(price_prices_test, predicted_prices)

# remove the brackets from the lists
# price_prices_test = [i[0] for i in price_prices_test]
# predicted_prices = [i[0] for i in predicted_prices]

plt.figure(figsize=(11,8))
plt.plot(price_prices_test, predicted_prices, 'o', color='blue', label='$R^2$ = {}'.format(np.round(r2,3)))

plt.xlabel('Real Prices', fontsize=15)
plt.ylabel('Predicted Prices',  fontsize=15)  
plt.title('Real vs. Predicted Prices')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend()
plt.savefig('realvspredictedr2.png')

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# ================================
# Page config
# ================================
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

# ================================
# LSTM Model
# ================================
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True,
                            dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# ================================
# Train and cache model
# ================================
@st.cache_resource
def train_model(ticker, start, end):
    stock = yf.download(ticker, start=start, end=end)
    data = stock['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Create sequences
    X, y = [], []
    for i in range(60, len(data_scaled)):
        X.append(data_scaled[i-60:i, 0])
        y.append(data_scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y)

    # Train model
    model = StockLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        pred = model(X_t).squeeze()
        loss = criterion(pred, y_t)
        loss.backward()
        optimizer.step()

    return model, scaler, stock, data_scaled

# ================================
# App UI
# ================================
st.title("📈 Stock Price Predictor")
st.write("Predict future stock prices using LSTM Neural Network!")

# Sidebar
st.sidebar.header("Settings")
ticker = st.sidebar.selectbox("Select Stock",
    ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META"])
start_date = st.sidebar.date_input("Start Date",
    value=pd.to_datetime("2019-01-01"))
end_date = st.sidebar.date_input("End Date",
    value=pd.to_datetime("2024-01-01"))
forecast_days = st.sidebar.slider("Forecast Days", 7, 60, 30)

if st.button("🔮 Train & Predict", type="primary"):
    with st.spinner(f'Training LSTM on {ticker} data... (1-2 minutes)'):
        model, scaler, stock, data_scaled = train_model(
            ticker, str(start_date), str(end_date))

    st.success("Model trained! ✅")

    # Stock stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price",
                  f"${stock['Close'].iloc[-1].values[0]:.2f}")
    with col2:
        st.metric("Highest Price",
                  f"${stock['Close'].max().values[0]:.2f}")
    with col3:
        st.metric("Lowest Price",
                  f"${stock['Close'].min().values[0]:.2f}")
    with col4:
        st.metric("Average Price",
                  f"${stock['Close'].mean().values[0]:.2f}")

    # Forecast
    model.eval()
    last_60 = data_scaled[-60:]
    future = []

    for _ in range(forecast_days):
        seq = torch.FloatTensor(last_60).reshape(1, 60, 1)
        with torch.no_grad():
            next_p = model(seq).item()
        future.append(next_p)
        last_60 = np.append(last_60[1:], [[next_p]], axis=0)

    future_prices = scaler.inverse_transform(
        np.array(future).reshape(-1, 1))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Historical price
    axes[0].plot(stock['Close'].values, color='#378ADD', linewidth=1.5)
    axes[0].set_title(f'{ticker} Historical Price')
    axes[0].set_xlabel('Days')
    axes[0].set_ylabel('Price ($)')
    axes[0].grid(True, alpha=0.3)

    # Forecast
    historical = scaler.inverse_transform(data_scaled[-60:])
    axes[1].plot(range(60), historical,
                 color='#378ADD', linewidth=2, label='Historical')
    axes[1].plot(range(60, 60+forecast_days), future_prices,
                 color='#1D9E75', linewidth=2,
                 linestyle='--', label=f'{forecast_days} Day Forecast')
    axes[1].axvline(x=60, color='#E24B4A', linestyle='--', alpha=0.7)
    axes[1].set_title(f'{ticker} — {forecast_days} Day Forecast')
    axes[1].set_xlabel('Days')
    axes[1].set_ylabel('Price ($)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # Forecast table
    st.subheader(f"📊 {forecast_days} Day Price Forecast")
    forecast_df = pd.DataFrame({
        'Day': range(1, forecast_days+1),
        'Predicted Price': [f"${p[0]:.2f}" for p in future_prices]
    })
    st.dataframe(forecast_df, hide_index=True, use_container_width=True)


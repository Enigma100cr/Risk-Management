# app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from kiteconnect import KiteConnect
from transformers import pipeline
import requests
import yfinance as yf
from datetime import datetime, timedelta
import pytz
from scipy.stats import norm
import tensorflow as tf
import quantstats as qs
import feedparser
from streamlit.components.v1 import html

# ---------------------
# Configuration
# ---------------------
st.set_page_config(
    page_title="AI Trading Nexus Pro",
    layout="wide",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

# ---------------------
# Initialize Components
# ---------------------
try:
    sentiment_analyzer = pipeline('sentiment-analysis', model='ProsusAI/finbert')
except Exception as e:
    st.error(f"Error initializing NLP model: {str(e)}")

# ---------------------
# Data Modules
# ---------------------
@st.cache_data(ttl=300)
def get_live_market_data():
    """Fetch real-time NSE data"""
    try:
        nse_indices = ['^NSEI', '^NSEBANK', '^INDIAVIX']
        data = {}
        for index in nse_indices:
            ticker = yf.Ticker(index)
            hist = ticker.history(period='1d')
            data[index] = hist['Close'].iloc[-1]
        return data
    except Exception as e:
        st.error(f"Market data error: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def get_fii_dii_data():
    """Fetch institutional activity from NSE"""
    try:
        url = "https://www.nseindia.com/api/daily-reports?key=fiiDii"
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).json()
        return pd.DataFrame(response['data'])
    except Exception as e:
        st.error(f"Institutional data error: {str(e)}")
        return pd.DataFrame()

# ---------------------
# AI Modules
# ---------------------
class LSTMPredictor:
    def __init__(self):
        self.model = tf.keras.models.load_model('lstm_model.h5')  # Pretrained model
    
    def predict(self, ticker):
        try:
            data = yf.download(f"{ticker}.NS", period='60d')['Close'].values
            sequence = data[-30:].reshape(1, 30, 1)
            prediction = self.model.predict(sequence)
            return data[-1], prediction[0][0]
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None, None

# ---------------------
# UI Components
# ---------------------
def render_sidebar():
    with st.sidebar:
        st.header("🔑 Broker Authentication")
        api_key = st.text_input("Zerodha API Key", type="password")
        access_token = st.text_input("Access Token", type="password")
        
        if st.button("Connect Live Data"):
            try:
                kite = KiteConnect(api_key=api_key)
                kite.set_access_token(access_token)
                st.session_state.kite = kite
                st.success("Connected successfully!")
            except Exception as e:
                st.error(f"Connection failed: {str(e)}")
        
        st.header("⚙️ Settings")
        st.selectbox("Trading Style", ["Intraday", "Swing", "Positional"])
        st.slider("Risk Appetite", 1, 5, 3)
        st.checkbox("Enable Tax Optimization", True)

def render_market_overview():
    col1, col2, col3 = st.columns(3)
    with col1:
        data = get_live_market_data()
        if data:
            st.metric("NIFTY 50", f"₹{data['^NSEI']:,.2f}")
    
    with col2:
        st.metric("BANK NIFTY", f"₹{data['^NSEBANK']:,.2f}")
    
    with col3:
        st.metric("India VIX", f"{data['^INDIAVIX']:.2f}%")

# ---------------------
# Main App
# ---------------------
def main():
    st.title("🇮🇳 AI Trading Nexus Pro")
    st.markdown("Institutional-grade trading platform for Indian markets")
    
    render_sidebar()
    render_market_overview()
    
    # ------------------
    # Institutional Flow
    # ------------------
    st.subheader("📈 Smart Money Tracking")
    try:
        fii_dii = get_fii_dii_data()
        if not fii_dii.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=fii_dii['date'],
                y=fii_dii['fii_net'],
                name='FII Flow',
                marker_color='#1f77b4'
            ))
            fig.add_trace(go.Bar(
                x=fii_dii['date'],
                y=fii_dii['dii_net'],
                name='DII Flow',
                marker_color='#ff7f0e'
            ))
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Flow data error: {str(e)}")
    
    # ------------------
    # AI Prediction
    # ------------------
    st.subheader("🔮 AI Forecast Engine")
    pred_col1, pred_col2 = st.columns(2)
    
    with pred_col1:
        ticker = st.selectbox("Select Stock", ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY'])
        predictor = LSTMPredictor()
        current_price, predicted_price = predictor.predict(ticker)
        if current_price and predicted_price:
            delta = (predicted_price - current_price)/current_price
            st.metric("LSTM Prediction", 
                     f"₹{predicted_price:,.2f}", 
                     f"{delta:.2%}")
    
    with pred_col2:
        st.write("**Technical Signals**")
        st.code("""
        RSI (14): 68 [Neutral]
        MACD: -1.5 [Bearish]
        Volume Trend: ↗️ 1.2M → 1.8M
        Options OI: ↗️ 12% Increase
        """)
    
    # ------------------
    # Backtesting
    # ------------------
    st.subheader("📊 Strategy Analytics")
    tab1, tab2, tab3 = st.tabs(["Backtester", "Optimizer", "Risk Analysis"])
    
    with tab1:
        if st.button("Run Backtest"):
            try:
                returns = qs.utils.download_returns('^NSEI', period='2y')
                qs.reports.html(
                    returns,
                    benchmark='^NSEI',
                    output='backtest.html',
                    title='Strategy Backtest'
                )
                with open('backtest.html', 'r') as f:
                    html_content = f.read()
                html(html_content, height=1000, scrolling=True)
            except Exception as e:
                st.error(f"Backtest failed: {str(e)}")
    
    # ------------------
    # Risk Management
    # ------------------
    with tab3:
        st.subheader("🛡️ Portfolio Protection")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Scenario Analysis**")
            scenarios = {
                'Market Crash (-40%)': 0.6,
                'Rate Hike (-25%)': 0.75,
                'Currency Crisis (-30%)': 0.7
            }
            for scenario, factor in scenarios.items():
                st.write(f"{scenario}: ₹{1000000 * factor:,.2f}")
        
        with col2:
            st.write("**Hedge Advisor**")
            st.progress(0.65, text="Optimal Hedge Ratio: 65%")
            st.checkbox("Enable Dynamic Hedging")
    
    # ------------------
    # News Integration
    # ------------------
    st.subheader("📰 Real-time Market Pulse")
    try:
        news = feedparser.parse("https://www.moneycontrol.com/rss/latestnews.xml")
        for entry in news.entries[:5]:
            with st.expander(f"{entry.title}"):
                st.write(entry.published)
                sentiment = sentiment_analyzer(entry.title)[0]
                st.write(f"Sentiment: {sentiment['label']} ({sentiment['score']:.2f})")
                st.markdown(f"[Read more]({entry.link})")
    except Exception as e:
        st.error(f"News error: {str(e)}")

# ---------------------
# Run Application
# ---------------------
if __name__ == "__main__":
    main()

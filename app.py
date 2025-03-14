# app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
import yfinance as yf

# Configure page
st.set_page_config(
    page_title="AI Trading Analytics Suite",
    layout="wide",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .metric-box {padding: 20px; border-radius: 10px; margin: 10px 0;}
    .positive {background-color: #e6f4ea; color: #137333;}
    .negative {background-color: #fce8e6; color: #a50e0e;}
    .header {color: #1a73e8; font-weight: bold;}
    .stExpander {border: 1px solid #e0e0e0; border-radius: 8px; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)

def main():
    st.title("📈 AI-Powered Trading Analytics Platform")
    
    # Navigation
    menu = st.sidebar.radio("Navigation", [
        "Risk Metrics", 
        "Strategy Analysis", 
        "Technical Analysis", 
        "Portfolio Management"
    ])

    if menu == "Risk Metrics":
        show_risk_metrics()
    elif menu == "Strategy Analysis":
        show_strategy_analysis()
    elif menu == "Technical Analysis":
        show_technical_analysis()
    elif menu == "Portfolio Management":
        show_portfolio_management()

def show_risk_metrics():
    st.header("📉 Advanced Risk Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("Sharpe Ratio", expanded=True):
            st.markdown('<div class="header">Risk-Adjusted Returns</div>', unsafe_allow_html=True)
            rp = st.number_input("Portfolio Return (%)", min_value=0.0, key="sr_rp") / 100
            rf = st.number_input("Risk-Free Rate (%)", min_value=0.0, key="sr_rf") / 100
            sigma_p = st.number_input("Portfolio Volatility (%)", min_value=0.0, key="sr_vol") / 100
            
            if sigma_p != 0:
                sharpe = (rp - rf) / sigma_p
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            else:
                st.error("Volatility cannot be zero!")

        with st.expander("Value at Risk (VaR)"):
            st.markdown('<div class="header">Potential Maximum Loss</div>', unsafe_allow_html=True)
            portfolio_value = st.number_input("Portfolio Value ($)", min_value=0.0, key="var_val")
            confidence = st.selectbox("Confidence Level", [95, 99], key="var_conf")
            volatility = st.number_input("Annual Volatility (%)", min_value=0.0, key="var_vol") / 100
            days = st.number_input("Time Horizon (Days)", min_value=1, key="var_days")
            
            z_score = norm.ppf(confidence/100)
            daily_vol = volatility / np.sqrt(252)
            var = portfolio_value * z_score * daily_vol * np.sqrt(days)
            st.metric(f"{confidence}% VaR", f"${abs(var):,.2f}")

    with col2:
        with st.expander("Sortino Ratio"):
            st.markdown('<div class="header">Downside Risk Adjustment</div>', unsafe_allow_html=True)
            sor_rp = st.number_input("Portfolio Return (%)", key="sor_rp") / 100
            sor_rf = st.number_input("Risk-Free Rate (%)", key="sor_rf") / 100
            sigma_d = st.number_input("Downside Volatility (%)", min_value=0.0, key="sor_vol") / 100
            
            if sigma_d != 0:
                sortino = (sor_rp - sor_rf) / sigma_d
                st.metric("Sortino Ratio", f"{sortino:.2f}")
            else:
                st.error("Downside Volatility cannot be zero!")

        with st.expander("Conditional VaR (CVaR)"):
            st.markdown('<div class="header">Expected Tail Loss</div>', unsafe_allow_html=True)
            cvar_conf = st.selectbox("Confidence Level", [95, 99], key="cvar_conf")
            cvar_val = st.number_input("Portfolio Value ($)", min_value=0.0, key="cvar_val")
            cvar_vol = st.number_input("Annual Volatility (%)", key="cvar_vol") / 100
            cvar_days = st.number_input("Time Horizon (Days)", key="cvar_days")
            
            alpha = cvar_conf/100
            z = norm.ppf(alpha)
            daily_vol = cvar_vol / np.sqrt(252)
            cvar = cvar_val * (norm.pdf(z)/(1-alpha)) * daily_vol * np.sqrt(cvar_days)
            st.metric(f"{cvar_conf}% CVaR", f"${abs(cvar):,.2f}")

    # Additional Risk Metrics
    with st.expander("Advanced Risk Analysis"):
        col1, col2 = st.columns(2)
        
        with col1:
            peak = st.number_input("Portfolio Peak Value ($)", min_value=0.0)
            trough = st.number_input("Portfolio Trough Value ($)", min_value=0.0)
            if peak > 0:
                drawdown = (peak - trough)/peak
                st.metric("Maximum Drawdown", f"{drawdown:.2%}")
        
        with col2:
            cagr = st.number_input("Annual Return (CAGR %)", min_value=0.0) / 100
            max_dd = st.number_input("Max Drawdown (%)", min_value=0.0, max_value=100.0) / 100
            if max_dd != 0:
                calmar = cagr / max_dd
                st.metric("Calmar Ratio", f"{calmar:.2f}")

def show_strategy_analysis():
    st.header("📈 Trading Strategy Analysis")
    
    strategy = st.selectbox("Select Strategy", [
        "Momentum Trading", 
        "Mean Reversion", 
        "Breakout Strategy",
        "Scalping"
    ])
    
    with st.expander(f"{strategy} Backtester"):
        col1, col2 = st.columns(2)
        
        with col1:
            capital = st.number_input("Initial Capital ($)", value=10000.0)
            risk_per_trade = st.slider("Risk per Trade (%)", 0.1, 5.0, 2.0)
            win_rate = st.slider("Win Rate (%)", 1, 100, 60)
        
        with col2:
            avg_win = st.number_input("Average Win (%)", value=15.0)
            avg_loss = st.number_input("Average Loss (%)", value=10.0)
            num_trades = st.number_input("Trades/Month", value=20)
        
        # Monte Carlo simulation
        monthly_returns = []
        for _ in range(1000):
            monthly_pnl = 0
            for _ in range(num_trades):
                if np.random.rand() < win_rate/100:
                    monthly_pnl += capital * (risk_per_trade/100) * (avg_win/100)
                else:
                    monthly_pnl -= capital * (risk_per_trade/100) * (avg_loss/100)
            monthly_returns.append(monthly_pnl)
        
        # Display results
        fig = px.histogram(monthly_returns, nbins=50, 
                          title="Expected Monthly Returns Distribution")
        st.plotly_chart(fig, use_container_width=True)

def show_technical_analysis():
    st.header("📊 Technical Analysis Toolkit")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        symbol = st.text_input("Stock Symbol", "AAPL")
        period = st.selectbox("Time Frame", ["1d", "5d", "1mo", "6mo", "1y"])
        indicator = st.selectbox("Technical Indicator", [
            "RSI", "MACD", 
            "Bollinger Bands", 
            "Moving Averages"
        ])
    
    with col2:
        data = yf.download(symbol, period=period)
        
        if not data.empty:
            fig = go.Figure()
            
            # Price chart
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ))
            
            # Add selected indicator
            if indicator == "Moving Averages":
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'].rolling(20).mean(),
                    name='20-day MA'
                ))
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'].rolling(50).mean(),
                    name='50-day MA'
                ))
            
            st.plotly_chart(fig, use_container_width=True)

def show_portfolio_management():
    st.header("💰 Portfolio Optimization")
    
    with st.expander("Modern Portfolio Theory"):
        assets = st.multiselect("Select Assets", ["SPY", "GLD", "BTC-USD", "TLT"])
        
        if len(assets) >= 2:
            returns = pd.DataFrame()
            for asset in assets:
                data = yf.download(asset, period="1y")['Close'].pct_change().dropna()
                returns[asset] = data
            
            # Calculate efficient frontier
            cov_matrix = returns.cov()
            expected_returns = returns.mean()
            
            # Generate random portfolios
            num_portfolios = 10000
            results = np.zeros((3, num_portfolios))
            
            for i in range(num_portfolios):
                weights = np.random.random(len(assets))
                weights /= np.sum(weights)
                
                portfolio_return = np.sum(weights * expected_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                results[0,i] = portfolio_return
                results[1,i] = portfolio_vol
                results[2,i] = results[0,i] / results[1,i]  # Sharpe Ratio
            
            # Plot efficient frontier
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results[1,:], y=results[0,:], 
                mode='markers',
                marker=dict(
                    color=results[2,:],
                    colorscale='Viridis',
                    size=8
                )
            ))
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()

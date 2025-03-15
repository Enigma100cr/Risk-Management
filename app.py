# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import norm

# Configure stable page settings
st.set_page_config(
    page_title="Trade Fitness Pro",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .metric-card {padding: 20px; border-radius: 10px; margin: 10px 0; border: 1px solid #e0e0e0;}
    .positive {color: #2e7d32;} .negative {color: #c62828;}
    .exercise-box {border-left: 5px solid #4caf50; padding-left: 1rem;}
    .stProgress > div > div {background: #4caf50;}
    .stButton>button {background-color: #4caf50; color: white;}
</style>
""", unsafe_allow_html=True)

# Cache data processing for performance
@st.cache_data(ttl=3600)
def process_trades(uploaded_file):
    """Process uploaded trade data with comprehensive error handling"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        
        # Standardize columns
        column_map = {
            'Symbol': 'symbol', 'Qty': 'quantity',
            'Buy Price': 'entry_price', 'Sell Price': 'exit_price',
            'Date': 'trade_date', 'P&L': 'pnl'
        }
        df = df.rename(columns={k:v for k,v in column_map.items() if k in df.columns})
        
        # Calculate essential metrics
        if all(col in df.columns for col in ['entry_price', 'exit_price', 'quantity']):
            df['pnl'] = (df['exit_price'] - df['entry_price']) * df['quantity']
            df['return_pct'] = (df['exit_price'] / df['entry_price'] - 1) * 100
            df['result'] = np.where(df['pnl'] > 0, 'Win', 'Loss')
            df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
        
        return df.dropna(subset=['pnl'])
    
    except Exception as e:
        st.error(f"🚨 File Error: {str(e)}")
        return pd.DataFrame()

def show_fitness_dashboard(df):
    """Main trading fitness interface"""
    st.title("💰 Trading Fitness Dashboard")
    
    # Key metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        total = df['pnl'].sum()
        st.metric("Total P&L", f"₹{total:+,.2f}", 
                 help="Your financial push-ups - total profit/loss")
    
    with col2:
        win_rate = len(df[df['pnl'] > 0])/len(df)*100
        st.metric("Win Rate", f"{win_rate:.1f}%", 
                 help="Your trading accuracy score")
    
    with col3:
        avg_return = df['return_pct'].mean()
        st.metric("Avg Return", f"{avg_return:+.1f}%", 
                 help="Average gain per trade")
    
    # Visualization section
    with st.expander("📈 Performance Analysis", expanded=True):
        tab1, tab2, tab3 = st.tabs(["Wealth Journey", "Asset Report Card", "Risk Analysis"])
        
        with tab1:
            df = df.sort_values('trade_date')
            df['cumulative_pnl'] = df['pnl'].cumsum()
            fig = px.area(df, x='trade_date', y='cumulative_pnl', 
                          title="Your Financial Fitness Progress")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            asset_perf = df.groupby('symbol')['pnl'].sum().reset_index()
            fig = px.bar(asset_perf, x='symbol', y='pnl', color='pnl',
                        title="Stock Performance Report Card")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            daily_returns = df['return_pct']/100
            fig = px.histogram(daily_returns, nbins=50,
                             title="Daily Returns Distribution")
            st.plotly_chart(fig, use_container_width=True)

def show_trading_school():
    """Interactive trading education section"""
    st.title("🏫 Trading Fitness Academy")
    
    lessons = {
        "💪 Profit & Loss": {
            "analogy": "Financial Push-Ups",
            "formula": "P&L = (Sell Price - Buy Price) × Quantity",
            "example": "Buy 100 Reliance @ ₹2500 → Sell @ ₹2600\nP&L = (2600-2500) × 100 = ₹10,000 Profit"
        },
        "🎯 Risk-Reward Ratio": {
            "analogy": "Cricket Batting Strategy",
            "formula": "Risk ₹1 to Make ₹2 = 1:2 Ratio",
            "example": "Stop Loss @ ₹2450 (Risk ₹50)\nTarget @ ₹2600 (Reward ₹100)"
        },
        "🛡️ Value at Risk (VaR)": {
            "analogy": "Financial Seatbelt",
            "formula": "Worst-case loss with 95% confidence",
            "example": "₹1L portfolio → 95% VaR = ₹5,000\nMeans 5% chance to lose >₹5k"
        }
    }
    
    for title, content in lessons.items():
        with st.expander(title, expanded=True):
            st.markdown(f"""
            **{content['analogy']}**  
            *{content['formula']}*
            
            📚 **Real Example:**  
            {content['example']}
            """)
            st.markdown("---")

def show_advanced_metrics(df):
    """Professional risk analysis tools"""
    st.title("🔍 Advanced Fitness Metrics")
    
    with st.expander("Risk Analysis Gym"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sharpe Ratio")
            returns = df['return_pct']/100
            rf = st.number_input("Risk-Free Rate (%)", 5.0)/100
            sharpe = (returns.mean() - rf)/returns.std() * np.sqrt(252)
            st.metric("Score", f"{sharpe:.2f}", 
                     help="Higher = Better risk-adjusted returns")
        
        with col2:
            st.subheader("Sortino Ratio")
            downside_returns = returns[returns < 0]
            sortino = (returns.mean() - rf)/downside_returns.std() * np.sqrt(252)
            st.metric("Score", f"{sortino:.2f}", 
                     help="Focuses on downside risk only")
    
    with st.expander("Safety Equipment Check"):
        col1, col2 = st.columns(2)
        
        with col1:
            confidence = st.selectbox("Confidence Level", [95, 99])
            var = df['pnl'].quantile(1 - confidence/100)
            st.metric(f"{confidence}% VaR", f"₹{var:,.2f}")
        
        with col2:
            cvar = df[df['pnl'] <= var]['pnl'].mean()
            st.metric(f"{confidence}% CVaR", f"₹{cvar:,.2f}")

def main():
    # Navigation sidebar
    st.sidebar.title("Trade Fitness Coach")
    page = st.sidebar.radio("Menu", ["Dashboard", "Academy", "Advanced Gym"])
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Trade History", 
                                            type=['csv','xlsx'],
                                            help="Supports common Indian broker formats")
    
    df = process_trades(uploaded_file) if uploaded_file else pd.DataFrame()
    
    if page == "Dashboard":
        if not df.empty:
            show_fitness_dashboard(df)
        else:
            st.info("📤 Upload your trade history to begin analysis")
    
    elif page == "Academy":
        show_trading_school()
    
    elif page == "Advanced Gym":
        st.title("🏋️♂️ Advanced Trading Gym")
        if not df.empty:
            show_advanced_metrics(df)
        else:
            st.warning("Upload data to access advanced metrics")

if __name__ == "__main__":
    main()

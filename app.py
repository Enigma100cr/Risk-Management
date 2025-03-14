# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import norm

# Configure page
st.set_page_config(
    page_title="Trade Fitness Coach",
    layout="wide",
    page_icon="🏋️♂️",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .metric-card {padding: 20px; border-radius: 10px; margin: 10px 0; border: 1px solid #e0e0e0;}
    .fitness-score {background: #e3f2fd; color: #0d47a1;}
    .exercise-box {border-left: 5px solid #4caf50; padding-left: 1rem;}
    .good {color: #2e7d32;} .bad {color: #c62828;}
    .stProgress > div > div {background: #4caf50;}
</style>
""", unsafe_allow_html=True)

def process_indian_trades(uploaded_file):
    """Process Indian broker files with safety checks"""
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        
        # Standardize Indian broker columns
        column_map = {
            'Symbol': 'symbol', 'Scrip': 'symbol', 'Instrument': 'symbol',
            'Qty': 'quantity', 'Buy Price': 'entry_price', 
            'Sell Price': 'exit_price', 'Trade Date': 'trade_date',
            'P&L': 'pnl', 'Charges': 'charges'
        }
        df = df.rename(columns={k:v for k,v in column_map.items() if k in df.columns})
        
        # Calculate essential metrics
        if all(col in df.columns for col in ['entry_price', 'exit_price', 'quantity']):
            df['pnl'] = (df['exit_price'] - df['entry_price']) * df['quantity']
            df['return_pct'] = (df['exit_price'] / df['entry_price'] - 1) * 100
            df['result'] = np.where(df['pnl'] > 0, 'Win', 'Loss')
        
        return df
    except Exception as e:
        st.error(f"🚨 File Error: {str(e)}")
        return pd.DataFrame()

def show_fitness_dashboard(df):
    """Main trading fitness dashboard"""
    st.title("🏋️♂️ Your Trading Fitness Report")
    
    # Key metrics cards
    total_pnl = df['pnl'].sum()
    win_rate = len(df[df['pnl'] > 0])/len(df)*100 if len(df) > 0 else 0
    avg_return = df['return_pct'].mean() if len(df) > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card fitness-score"><h3>💪 Trading Strength</h3>'
                    f'<h2 class="{"good" if total_pnl > 0 else "bad"}">₹{total_pnl:+,.2f}</h2>'
                    '<p>Total Profit/Loss</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="metric-card fitness-score"><h3>🎯 Accuracy Score</h3>'
                    f'<h2>{win_rate:.1f}%</h2><p>Win Rate</p></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="metric-card fitness-score"><h3>⚡ Power Returns</h3>'
                    f'<h2 class="{"good" if avg_return > 0 else "bad"}">{avg_return:+.1f}%</h2>'
                    '<p>Avg Return/Trade</p></div>', unsafe_allow_html=True)

    # Visual fitness tracker
    with st.expander("📈 Progress Charts", expanded=True):
        tab1, tab2 = st.tabs(["Profit Journey", "Stock Performance"])
        
        with tab1:
            if 'trade_date' in df.columns:
                df = df.sort_values('trade_date')
                df['Cumulative P&L'] = df['pnl'].cumsum()
                fig = px.area(df, x='trade_date', y='Cumulative P&L', 
                              title="Your Profit Growth Journey")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            stock_perf = df.groupby('symbol')['pnl'].sum().reset_index()
            fig = px.bar(stock_perf, x='symbol', y='pnl', color='pnl',
                        title="Stock-wise Performance")
            st.plotly_chart(fig, use_container_width=True)

def show_beginner_education():
    """Interactive trading fitness tutorial"""
    st.title("📚 Trading Fitness 101")
    
    lessons = {
        "💪 Profit/Loss (P&L)": {
            "analogy": "Your Financial Push-Up",
            "desc": "Measures your trading strength - money gained/lost per trade",
            "example": "Buy 10 TCS @ ₹3,500 → Sell @ ₹3,600\nP&L = (3600-3500) × 10 = ₹1,000 profit"
        },
        "🎯 Win Rate": {
            "analogy": "Accuracy Score",
            "desc": "Percentage of successful trades vs total attempts",
            "example": "7 wins out of 10 trades = 70% Win Rate\nLike hitting 7/10 shots in basketball"
        },
        "⚖️ Risk-Reward": {
            "analogy": "Financial Balance",
            "desc": "Risk ₹1 to make ₹2 = 1:2 ratio\nAlways aim for positive ratios",
            "example": "Stop Loss: ₹10 | Target: ₹20 = 1:2 ratio"
        }
    }
    
    for title, content in lessons.items():
        with st.expander(title, expanded=True):
            st.markdown(f"""
            **{content['analogy']}**  
            {content['desc']}
            
            *Example:*  
            {content['example']}
            """)
            st.markdown("---")

def main():
    st.sidebar.title("🏋️♂️ Trade Fitness Coach")
    page = st.sidebar.radio("Menu", ["Fitness Dashboard", "Beginner Tutorial", "Advanced Metrics"])
    
    # File upload in sidebar
    uploaded_file = st.sidebar.file_uploader("Upload Trade History", 
                                            type=['csv','xlsx'],
                                            help="Supports Zerodha, Upstox, Angel One formats")
    
    df = process_indian_trades(uploaded_file) if uploaded_file else pd.DataFrame()
    
    if page == "Fitness Dashboard":
        if not df.empty:
            show_fitness_dashboard(df)
        else:
            st.info("📤 Upload your trade history to begin analysis")
    
    elif page == "Beginner Tutorial":
        show_beginner_education()
    
    elif page == "Advanced Metrics":
        st.title("📊 Advanced Fitness Metrics")
        
        if not df.empty:
            with st.expander("💹 Risk-Adjusted Returns"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Sharpe Ratio")
                    rf = st.number_input("Risk-Free Rate (%)", 5.0, key="sharpe_rf")/100
                    returns = df['return_pct']/100
                    sharpe = (returns.mean() - rf)/returns.std() * np.sqrt(252)
                    st.metric("Score", f"{sharpe:.2f}", 
                             help="Higher = Better risk-adjusted returns")
                
                with col2:
                    st.subheader("Sortino Ratio")
                    downside_returns = returns[returns < 0]
                    sortino = (returns.mean() - rf)/downside_returns.std() * np.sqrt(252)
                    st.metric("Score", f"{sortino:.2f}", 
                             help="Focuses on downside risk only")
            
            with st.expander("⚠️ Risk Analysis"):
                col1, col2 = st.columns(2)
                
                with col1:
                    confidence = st.selectbox("Confidence Level", [95, 99], key="var_conf")
                    z = norm.ppf(confidence/100)
                    var = df['pnl'].quantile(1 - confidence/100)
                    st.metric(f"{confidence}% VaR", f"₹{var:,.2f}")
                
                with col2:
                    cvar = df[df['pnl'] <= var]['pnl'].mean()
                    st.metric(f"{confidence}% CVaR", f"₹{cvar:,.2f}")

if __name__ == "__main__":
    main()

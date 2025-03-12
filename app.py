import streamlit as st
import numpy as np
from scipy.stats import norm

# Page configuration
st.set_page_config(
    page_title="Trading Metrics Calculator",
    layout="wide",
    page_icon="📈"
)

# Main header
st.title("Trading Metrics Calculator")
st.markdown("""
A comprehensive tool for traders and investors to evaluate portfolio performance using key risk management metrics.
""")

# Sharpe Ratio
with st.expander("Sharpe Ratio", expanded=True):
    st.latex(r"\text{Sharpe Ratio} = \frac{R_p - R_f}{\sigma_p}")
    col1, col2, col3 = st.columns(3)
    with col1:
        rp = st.number_input("Portfolio Return (%)", min_value=0.0, key="sharpe_rp") / 100
    with col2:
        rf = st.number_input("Risk-Free Rate (%)", min_value=0.0, key="sharpe_rf") / 100
    with col3:
        sigma_p = st.number_input("Portfolio Volatility (%)", min_value=0.0, key="sharpe_vol") / 100
    
    if sigma_p != 0:
        sharpe = (rp - rf) / sigma_p
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    else:
        st.error("Volatility cannot be zero!")

# Sortino Ratio
with st.expander("Sortino Ratio"):
    st.latex(r"\text{Sortino Ratio} = \frac{R_p - R_f}{\sigma_d}")
    col1, col2, col3 = st.columns(3)
    with col1:
        rp_sortino = st.number_input("Portfolio Return (%)", key="sortino_rp") / 100
    with col2:
        rf_sortino = st.number_input("Risk-Free Rate (%)", key="sortino_rf") / 100
    with col3:
        sigma_d = st.number_input("Downside Volatility (%)", min_value=0.0, key="sortino_vol") / 100
    
    if sigma_d != 0:
        sortino = (rp_sortino - rf_sortino) / sigma_d
        st.metric("Sortino Ratio", f"{sortino:.2f}")
    else:
        st.error("Downside Volatility cannot be zero!")

# Value at Risk (VaR)
with st.expander("Value at Risk (VaR)"):
    st.latex(r"\text{VaR}_\alpha = P \, (\Delta P \leq -V) = \alpha")
    portfolio_value = st.number_input("Portfolio Value ($)", min_value=0.0, key="var_value")
    confidence = st.selectbox("Confidence Level", [95, 99], key="var_conf")
    volatility = st.number_input("Annual Volatility (%)", min_value=0.0, key="var_vol") / 100
    days = st.number_input("Time Horizon (Days)", min_value=1, key="var_days")
    
    z_score = norm.ppf(confidence/100)
    daily_vol = volatility / np.sqrt(252)
    var = portfolio_value * z_score * daily_vol * np.sqrt(days)
    st.metric(f"{confidence}% VaR", f"${abs(var):,.2f}")

# Conditional VaR
with st.expander("Conditional VaR (CVaR)"):
    st.latex(r"C\text{VaR}_\alpha = E \, [\text{Loss} \, | \, \text{Loss} \geq \text{VaR}_\alpha]")
    portfolio_value_cvar = st.number_input("Portfolio Value ($)", key="cvar_value", min_value=0.0)
    confidence_cvar = st.selectbox("Confidence Level", [95, 99], key="cvar_conf")
    volatility_cvar = st.number_input("Annual Volatility (%)", key="cvar_vol", min_value=0.0) / 100
    days_cvar = st.number_input("Time Horizon (Days)", key="cvar_days", min_value=1)
    
    alpha = confidence_cvar/100
    z = norm.ppf(alpha)
    daily_vol = volatility_cvar / np.sqrt(252)
    cvar = portfolio_value_cvar * (norm.pdf(z) / (1 - alpha)) * daily_vol * np.sqrt(days_cvar)
    st.metric(f"{confidence_cvar}% CVaR", f"${abs(cvar):,.2f}")

# Expectancy
with st.expander("Trading Expectancy"):
    st.latex(r"\text{Expectancy} = \left( \frac{W}{T} \times AWR \right) - \left( \frac{L}{T} \times ALR \right)")
    col1, col2 = st.columns(2)
    with col1:
        wins = st.number_input("Winning Trades", min_value=0, key="win_trades")
        awr = st.number_input("Avg Win Size (%)", min_value=0.0, key="awr") / 100
    with col2:
        losses = st.number_input("Losing Trades", min_value=0, key="loss_trades")
        alr = st.number_input("Avg Loss Size (%)", min_value=0.0, key="alr") / 100
    
    total = wins + losses
    if total > 0:
        expectancy = ((wins/total)*awr) - ((losses/total)*alr)
        st.metric("Expectancy", f"{expectancy:.2%}")
    else:
        st.error("Total trades cannot be zero!")

# MAE
with st.expander("Maximum Adverse Excursion (MAE)"):
    st.latex(r"\text{MAE} = \max(P_{\text{entry}} - P_{\text{min}})")
    entry = st.number_input("Entry Price", min_value=0.0, key="mae_entry")
    low = st.number_input("Lowest Price", min_value=0.0, key="mae_low")
    mae = entry - low
    st.metric("MAE", f"{mae:.2f}")

# Drawdown
with st.expander("Drawdown Calculator"):
    st.latex(r"\text{Drawdown} = \frac{\text{Peak} - \text{Trough}}{\text{Peak}}")
    peak = st.number_input("Peak Value ($)", min_value=0.0, key="dd_peak")
    trough = st.number_input("Trough Value ($)", min_value=0.0, key="dd_trough")
    
    if peak > 0:
        drawdown = (peak - trough)/peak
        st.metric("Drawdown", f"{drawdown:.2%}")
    else:
        st.error("Peak value must be positive!")

# Calmar Ratio
with st.expander("Calmar Ratio"):
    st.latex(r"\text{Calmar Ratio} = \frac{R_p}{\text{Max Drawdown}}")
    cagr = st.number_input("Annual Return (%)", min_value=0.0, key="calmar_return") / 100
    max_dd = st.number_input("Max Drawdown (%)", min_value=0.0, max_value=100.0, key="calmar_dd") / 100
    
    if max_dd != 0:
        calmar = cagr / max_dd
        st.metric("Calmar Ratio", f"{calmar:.2f}")
    else:
        st.error("Max Drawdown cannot be zero!")

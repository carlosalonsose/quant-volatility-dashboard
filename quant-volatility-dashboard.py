import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import numpy as np
from scipy.stats import norm
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Quant Volatility Dashboard")

# --- Custom CSS (Adaptive Dark/Light Mode) ---
st.markdown("""
    <style>
    /* --- Base Styles (Light Mode Default) --- */
    .metric-container {
        padding: 10px;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 0.75rem 1rem;
        border-radius: 0.75rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #6b7280;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 600;
        color: #111827;
    }
    .metric-sub {
        font-size: 0.9rem;
        color: #10b981;
        font-weight: 600;
    }
    
    /* Arbitrage Cards Base */
    .arb-section-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin-top: 0.5rem;
        margin-bottom: 0.35rem;
    }
    .info-card {
        background-color: #f8fafc;
        border-radius: 0.75rem;
        padding: 1rem 1.25rem;
        border: 1px solid #e5e7eb;
        margin-bottom: 0.75rem;
    }
    .card-title {
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.25rem;
        color: #111827; 
    }
    .card-subtitle {
        font-size: 0.9rem;
        color: #4b5563;
        margin-bottom: 0.35rem;
    }
    .sell-card { border-left: 4px solid #dc2626; }
    .buy-card { border-left: 4px solid #16a34a; }
    .bs-card { border-left: 4px solid #2563eb; }
    .small-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6b7280;
        margin-bottom: 0.25rem;
    }

    /* --- Quant Metrics Cards Base --- */
    .qm-row { margin-top: 0.75rem; margin-bottom: 0.75rem; }
    .qm-card {
        background-color: #ffffff;
        border-radius: 0.75rem;
        padding: 0.9rem 1rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        font-size: 0.9rem;
    }
    .qm-card-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6b7280;
    }
    .qm-card-value {
        font-size: 0.95rem;
        font-weight: 600;
        color: #111827;
    }
    
    /* Pills (Badges) */
    .qm-pill-blue { background-color: #eff6ff; color: #1d4ed8; padding: 0.1rem 0.5rem; border-radius: 999px; font-size: 0.75rem; font-weight: 500; }
    .qm-pill-green { background-color: #ecfdf3; color: #15803d; padding: 0.1rem 0.5rem; border-radius: 999px; font-size: 0.75rem; font-weight: 500; }
    .qm-pill-amber { background-color: #fffbeb; color: #b45309; padding: 0.1rem 0.5rem; border-radius: 999px; font-size: 0.75rem; font-weight: 500; }

    /* --- DARK MODE OVERRIDES (Automatic detection) --- */
    @media (prefers-color-scheme: dark) {
        /* Main Metric Cards */
        .metric-card {
            background-color: #262730; /* Streamlit Dark Grey */
            border: 1px solid #41444e;
        }
        .metric-title { color: #9ca3af !important; }
        .metric-value { color: #f3f4f6 !important; }
        
        /* Arbitrage Cards */
        .info-card {
            background-color: #262730;
            border: 1px solid #41444e;
        }
        .card-title { color: #f3f4f6 !important; }
        .card-subtitle { color: #d1d5db !important; }
        .small-label { color: #9ca3af !important; }

        /* Quant Metrics Cards */
        .qm-card {
            background-color: #262730;
            border: 1px solid #41444e;
        }
        .qm-card-label { color: #9ca3af !important; }
        .qm-card-value { color: #f3f4f6 !important; }
        
        /* Pills in Dark Mode (Desaturated backgrounds) */
        .qm-pill-blue { background-color: #1e3a8a; color: #bfdbfe; }
        .qm-pill-green { background-color: #14532d; color: #bbf7d0; }
        .qm-pill-amber { background-color: #78350f; color: #fde68a; }
    }
    </style>
    """, unsafe_allow_html=True)

# --- Caching Functions ---

@st.cache_resource(ttl=300)
def get_ticker_obj(T):
    return yf.Ticker(T)

@st.cache_data(ttl=300, hash_funcs={yf.Ticker: lambda t: t.ticker})
def get_spot_price(ticker):
    try:
        info = ticker.info
        price = info.get('currentPrice')
        if price: return float(price), info
        data = ticker.history(period='1d', interval='1m')
        if not data.empty: return float(data['Close'].iloc[-1]), info
        return 0.0, info
    except Exception: return 0.0, None

@st.cache_data(ttl=300, hash_funcs={yf.Ticker: lambda t: t.ticker})
def get_historical_data(ticker, period="1y"):
    try:
        return ticker.history(period=period)
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=300, hash_funcs={yf.Ticker: lambda t: t.ticker})
def get_option_data(ticker, expiration):
    try:
        chain = ticker.option_chain(expiration)
        current_price, _ = get_spot_price(ticker)
        
        if current_price > 0:
            chain.calls['moneyness'] = chain.calls['strike'] / current_price
            chain.puts['moneyness'] = chain.puts['strike'] / current_price
            chain.calls['log_moneyness'] = np.log(chain.calls['moneyness'].replace(0, np.nan))
            chain.puts['log_moneyness'] = np.log(chain.puts['moneyness'].replace(0, np.nan))
        else:
            chain.calls['moneyness'] = np.nan
            chain.puts['moneyness'] = np.nan
            
        return chain.calls, chain.puts
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=300, hash_funcs={yf.Ticker: lambda t: t.ticker})
def get_expirations(ticker):
    try:
        return ticker.options
    except Exception: return []

@st.cache_data(ttl=600, hash_funcs={yf.Ticker: lambda t: t.ticker})
def calculate_atm_term_structure(ticker, expirations_to_scan, current_price):
    term_iv = []
    if current_price == 0: return pd.DataFrame()
    for exp in expirations_to_scan:
        try:
            opt = ticker.option_chain(exp)
            atm = opt.calls.iloc[(opt.calls['strike'] - current_price).abs().argsort().iloc[0]]
            days = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days + 1
            if days <= 0: continue
            term_iv.append({'DTE': days, 'ATM_IV': atm['impliedVolatility']})
        except: continue
    return pd.DataFrame(term_iv)

@st.cache_data(ttl=600, hash_funcs={yf.Ticker: lambda t: t.ticker})
def calculate_skew_term_structure(ticker, expirations_to_scan, current_price):
    skew_terms = []
    if current_price == 0: return pd.DataFrame()
    for exp in expirations_to_scan:
        try:
            c = ticker.option_chain(exp).calls
            c['moneyness'] = c['strike'] / current_price
            c = c.dropna(subset=['impliedVolatility', 'moneyness'])
            c = c[c['moneyness'] > 0]
            c['log_moneyness'] = np.log(c['moneyness'])
            
            c_filtered = c.dropna(subset=['log_moneyness', 'impliedVolatility'])
            if len(c_filtered) > 3:
                coef = np.polyfit(c_filtered['log_moneyness'], c_filtered['impliedVolatility'], 2)
                slope = coef[1]
                curvature = coef[0]
                dte = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days + 1
                if dte <= 0: continue
                skew_terms.append({'DTE': dte, 'slope': slope, 'curvature': curvature})
        except: continue
    return pd.DataFrame(skew_terms)

# --- Financial Math ---

def d1_calculation(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0: return 0.0
    d1 = d1_calculation(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def calculate_delta(S, K, T, r, sigma, option_type='call'):
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0: return np.nan
    d1 = d1_calculation(S, K, T, r, sigma)
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

# --- Plotting ---

def plot_volatility_skew_plotly(
    T,
    expiration,
    calls,
    puts,
    current_price,
    lower_band,
    upper_band,
    atm_iv,
    days_to_expiry,
    template="plotly_white",
    height=520,
    x_mode="logm",  # "strike" or "logm"
):
    # Select X column
    if x_mode == "logm":
        x_col = "log_moneyness"
        x_title = "Log-moneyness ln(K/S)"
    else:
        x_col = "strike"
        x_title = "Strike Price ($)"

    # ========= 1) Data Prep + Liquidity Filter =========
    # Keep openInterest to filter out low liquidity data
    c = calls[[x_col, "impliedVolatility", "openInterest"]].dropna().copy()
    p = puts[[x_col, "impliedVolatility", "openInterest"]].dropna().copy()

    # Filter outlier IVs (0%, > 300%)
    c = c[(c["impliedVolatility"] > 0.01) & (c["impliedVolatility"] < 3)]
    p = p[(p["impliedVolatility"] > 0.01) & (p["impliedVolatility"] < 3)]

    # Filter strikes with very low OI (using quantile, not fixed)
    if not c.empty:
        oi_threshold_calls = c["openInterest"].quantile(0.4)
        c = c[c["openInterest"] >= oi_threshold_calls]
    if not p.empty:
        oi_threshold_puts = p["openInterest"].quantile(0.4)
        p = p[p["openInterest"] >= oi_threshold_puts]

    # ========= 2) Sorting and Smoothing (Rolling) =========
    def smooth_iv(df, x_col, window=3):
        """Sorts by x and applies rolling mean to smooth IV."""
        if df.empty:
            return df
        df = df.sort_values(x_col).copy()
        df["iv_smooth"] = (
            df["impliedVolatility"]
            .rolling(window=window, center=True)
            .mean()
        )
        # Fill edges with original value
        df["iv_plot"] = df["iv_smooth"].fillna(df["impliedVolatility"])
        return df

    c = smooth_iv(c, x_col=x_col, window=5)
    p = smooth_iv(p, x_col=x_col, window=5)

    fig = go.Figure()

    # ========= 3) CALLS =========
    if not c.empty:
        fig.add_trace(go.Scatter(
            x=c[x_col],
            y=c["iv_plot"],          # Using smoothed series
            mode="lines+markers",
            name="Calls",
            hovertemplate=(
                f"Type: Call<br>"
                f"{x_title}: %{{x}}<br>"
                "IV: %{y:.1%}<br>"
                "OI: %{customdata[0]:,.0f}<extra></extra>"
            ),
            customdata=c[["openInterest"]].to_numpy(),
        ))

    # ========= 4) PUTS =========
    if not p.empty:
        fig.add_trace(go.Scatter(
            x=p[x_col],
            y=p["iv_plot"],
            mode="lines+markers",
            name="Puts",
            hovertemplate=(
                f"Type: Put<br>"
                f"{x_title}: %{{x}}<br>"
                "IV: %{y:.1%}<br>"
                "OI: %{customdata[0]:,.0f}<extra></extra>"
            ),
            customdata=p[["openInterest"]].to_numpy(),
        ))

    # ========= 5) ±1σ Range & SPOT / ATM =========
    if x_mode == "strike":
        fig.add_vrect(
            x0=lower_band,
            x1=upper_band,
            fillcolor="lightgray",
            opacity=0.2,
            line_width=0,
            annotation_text="±1σ range",
            annotation_position="top left"
        )

        fig.add_vline(
            x=current_price,
            line_width=2,
            line_dash="dash",
            line_color="blue",
            annotation_text=f"Spot: {current_price:.2f}",
            annotation_position="top right"
        )
    else:
        # In log-moneyness: spot = 0
        if current_price - lower_band > 0 and upper_band > 0:
            lower_lm = np.log(lower_band / current_price)
            upper_lm = np.log(upper_band / current_price)
        else:
            lower_lm, upper_lm = -0.3, 0.3  # fallback

        fig.add_vrect(
            x0=lower_lm,
            x1=upper_lm,
            fillcolor="lightgray",
            opacity=0.2,
            line_width=0,
            annotation_text="±1σ around ATM",
            annotation_position="top left"
        )

        fig.add_vline(
            x=0.0,
            line_width=2,
            line_dash="dash",
            line_color="blue",
            annotation_text="ATM (ln(K/S)=0)",
            annotation_position="top right"
        )

    fig.update_layout(
        title=f"{T} Volatility Skew | Exp: {expiration} ({days_to_expiry} DTE)",
        template=template,
        height=height,
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title="Implied Volatility", tickformat=".0%")

    return fig

def plot_oi_volume_profile(calls, puts, current_price):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax1.bar(calls['strike'], calls['openInterest'], width=1.5, alpha=0.6, label='Call OI', color='#00CC96')
    ax1.bar(puts['strike'], puts['openInterest'], width=1.5, alpha=0.6, label='Put OI', color='#EF553B')
    ax1.axvline(x=current_price, color='blue', linestyle='--', label='Spot')
    ax1.set_title("Open Interest")
    ax1.legend()
    ax2.bar(calls['strike'], calls['volume'], width=1.5, alpha=0.6, label='Call Vol', color='#00CC96')
    ax2.bar(puts['strike'], puts['volume'], width=1.5, alpha=0.6, label='Put Vol', color='#EF553B')
    ax2.axvline(x=current_price, color='blue', linestyle='--', label='Spot')
    ax2.set_title("Volume")
    ax2.set_xlabel("Strike ($)")
    ax2.legend()
    return fig

def plot_atm_term_structure_plotly(df, template="plotly_white", height=400):
    fig = px.line(
        df,
        x="DTE",
        y="ATM_IV",
        markers=True,
    )
    fig.update_layout(
        title="ATM IV Term Structure",
        template=template,
        height=height,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    fig.update_yaxes(tickformat=".1%")
    fig.update_xaxes(title="DTE")
    fig.update_yaxes(title="ATM IV")
    return fig


def plot_skew_term_structure_plotly(df, template="plotly_white", height=400):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["DTE"],
        y=df["slope"],
        mode="lines+markers",
        name="Skew Slope"
    ))
    fig.add_trace(go.Scatter(
        x=df["DTE"],
        y=df["curvature"],
        mode="lines+markers",
        name="Curvature"
    ))
    fig.update_layout(
        title="Skew & Curvature Dynamics",
        template=template,
        height=height,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title="DTE")
    fig.update_yaxes(title="Value")
    return fig

def plot_liquidity_oi_plotly(calls, puts, current_price, template="plotly_white"):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=calls['strike'],
        y=calls['openInterest'],
        name='Call OI'
    ))

    fig.add_trace(go.Bar(
        x=puts['strike'],
        y=puts['openInterest'],
        name='Put OI'
    ))

    fig.add_vline(
        x=current_price,
        line_width=2,
        line_dash="dash",
        line_color="blue",
        annotation_text="Spot",
        annotation_position="top"
    )

    fig.update_layout(
        title="Open Interest by Strike",
        barmode='group',
        template=template,
        height=420,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title="Strike")
    fig.update_yaxes(title="Open Interest")

    return fig


def plot_liquidity_volume_plotly(calls, puts, current_price, template="plotly_white"):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=calls['strike'],
        y=calls['volume'],
        name='Call Vol'
    ))

    fig.add_trace(go.Bar(
        x=puts['strike'],
        y=puts['volume'],
        name='Put Vol'
    ))

    fig.add_vline(
        x=current_price,
        line_width=2,
        line_dash="dash",
        line_color="blue",
        annotation_text="Spot",
        annotation_position="top"
    )

    fig.update_layout(
        title="Volume by Strike",
        barmode='group',
        template=template,
        height=420,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title="Strike")
    fig.update_yaxes(title="Volume")

    return fig


def plot_liquidity_bubblemap(calls, puts, template="plotly_white", height=380):
    # Merge by strike to avoid NaNs from different indices
    c = calls[['strike', 'openInterest', 'volume']].rename(
        columns={'openInterest': 'call_oi', 'volume': 'call_vol'}
    )
    p = puts[['strike', 'openInterest', 'volume']].rename(
        columns={'openInterest': 'put_oi', 'volume': 'put_vol'}
    )

    df = pd.merge(c, p, on='strike', how='inner')

    # Calculate metrics
    df['total_oi'] = df['call_oi'].fillna(0) + df['put_oi'].fillna(0)
    df['total_vol'] = df['call_vol'].fillna(0) + df['put_vol'].fillna(0)

    # Avoid division by zero / infinities
    df['put_call_oi_ratio'] = df['put_oi'] / df['call_oi'].replace(0, np.nan)
    df['put_call_oi_ratio'] = df['put_call_oi_ratio'].replace([np.inf, -np.inf], np.nan)

    # Remove rows with zero liquidity
    df = df[(df['total_oi'] > 0) | (df['total_vol'] > 0)]

    # Safety check: size cannot contain NaN
    df['total_vol'] = df['total_vol'].fillna(0)

    if df.empty:
        # If empty after filtering, return a friendly empty figure
        fig = go.Figure()
        fig.update_layout(
            title="Liquidity Bubble Map (no liquid strikes after filters)",
            template=template,
            height=height,
        )
        return fig

    fig = px.scatter(
        df,
        x="strike",
        y="total_oi",
        size="total_vol",
        color="put_call_oi_ratio",
        color_continuous_scale="RdBu",
        labels={
            "total_oi": "Total OI",
            "total_vol": "Total Volume",
            "put_call_oi_ratio": "Put/Call OI"
        },
        template=template,
        height=height,
        title="Liquidity Bubble Map (OI vs Volume by Strike)"
    )

    fig.update_layout(
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_xaxes(title="Strike")
    fig.update_yaxes(title="Total OI")

    return fig

# Mini-charts / Sparklines
def plot_atm_sparkline(df, template="plotly_white"):
    fig = px.line(df, x="DTE", y="ATM_IV")
    fig.update_layout(
        template=template,
        height=120,
        margin=dict(l=10, r=10, t=20, b=20),
        showlegend=False,
    )
    fig.update_yaxes(tickformat=".1%", title=None)
    fig.update_xaxes(title=None)
    return fig


def plot_skew_sparkline(df, template="plotly_white"):
    fig = px.line(df, x="DTE", y="slope")
    fig.update_layout(
        template=template,
        height=120,
        margin=dict(l=10, r=10, t=20, b=20),
        showlegend=False,
    )
    fig.update_yaxes(title=None)
    fig.update_xaxes(title=None)
    return fig

# --- Main App ---
st.title("Advanced Quantitative Volatility Dashboard")

with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        T = st.text_input("Ticker Symbol:", "SPY").upper()
        ticker = get_ticker_obj(T)
    with col2:
        expirations = get_expirations(ticker)
        if not expirations: st.stop()
        default_idx = min(3, len(expirations)-1)
        expiration = st.selectbox("Expiration Date:", expirations, index=default_idx)
    with col3:
        st.write("")
        if st.button("Refresh Data"): st.cache_data.clear()

# Data Fetching
current_price, _ = get_spot_price(ticker)
hist = get_historical_data(ticker)
calls_raw, puts_raw = get_option_data(ticker, expiration)

if calls_raw.empty: st.stop()

# Sidebar
st.sidebar.header("Configuration")
filter_type = st.sidebar.radio("Filter Method", ("None", "Liquidity", "Moneyness", "Smart"), index=3)

calls, puts = calls_raw, puts_raw
if filter_type == "Liquidity":
    min_oi = st.sidebar.slider("Min OI", 0, 5000, 500, step=50)
    calls = calls[calls.openInterest >= min_oi]
    puts = puts[puts.openInterest >= min_oi]
elif filter_type == "Moneyness":
    calls = calls[(calls['strike'] >= current_price*0.8) & (calls['strike'] <= current_price*1.2)]
    puts = puts[(puts['strike'] >= current_price*0.8) & (puts['strike'] <= current_price*1.2)]
elif filter_type == "Smart":
    # Evaluate general liquidity
    median_oi_calls = calls['openInterest'].median()

    if median_oi_calls > 500:
        # 1) Filter by relative OI (quantile) instead of fixed number
        oi_threshold_calls = calls['openInterest'].quantile(0.30)  # top 70% liquidity
        oi_threshold_puts  = puts['openInterest'].quantile(0.30)

        calls = calls[calls['openInterest'] >= oi_threshold_calls]
        puts  = puts[puts['openInterest'] >= oi_threshold_puts]

        # 2) Bound by wider moneyness (remove deep tails)
        calls = calls[(calls['strike'] >= current_price * 0.7) & (calls['strike'] <= current_price * 1.3)]
        puts  = puts[(puts['strike']  >= current_price * 0.7) & (puts['strike']  <= current_price * 1.3)]
    else:
        # Illiquid market → more conservative strike filter
        calls = calls[(calls['strike'] >= current_price * 0.8) & (calls['strike'] <= current_price * 1.2)]
        puts  = puts[(puts['strike']  >= current_price * 0.8) & (puts['strike']  <= current_price * 1.2)]

risk_free_rate = st.sidebar.number_input("Risk-Free Rate", 0.0, 10.0, 4.5, step=0.1) / 100
realized_vol_lookback = st.sidebar.slider("Realized Vol Lookback", 10, 252, 30)
term_structure_n = st.sidebar.slider("Term Structure Exp", 3, 20, 8)
dark_mode = st.sidebar.checkbox("Dark Mode (Plots)", value=False)
plotly_template = "plotly_dark" if dark_mode else "plotly_white"

# Core Calcs
days_to_expiry = (datetime.strptime(expiration, "%Y-%m-%d") - datetime.now()).days + 1
if days_to_expiry <= 0: days_to_expiry = 1
T_years = days_to_expiry / 365.0

if not calls.empty:
    calls['BS_Price'] = calls.apply(lambda x: black_scholes_price(current_price, x['strike'], T_years, risk_free_rate, x['impliedVolatility'], 'call'), axis=1)
    calls['delta'] = calls.apply(lambda x: calculate_delta(current_price, x['strike'], T_years, risk_free_rate, x['impliedVolatility'], 'call'), axis=1)
    calls['Arb_Diff'] = calls['lastPrice'] - calls['BS_Price']

if not puts.empty:
    puts['BS_Price'] = puts.apply(lambda x: black_scholes_price(current_price, x['strike'], T_years, risk_free_rate, x['impliedVolatility'], 'put'), axis=1)
    puts['delta'] = puts.apply(lambda x: calculate_delta(current_price, x['strike'], T_years, risk_free_rate, x['impliedVolatility'], 'put'), axis=1)
    puts['Arb_Diff'] = puts['lastPrice'] - puts['BS_Price']

atm_iv = calls.iloc[(calls['strike'] - current_price).abs().argsort().iloc[0]]['impliedVolatility'] if not calls.empty else 0
exp_move = current_price * atm_iv * np.sqrt(T_years)

# --- Skew analytics (slope, curvature, RR, wings, noise) ---

slope = np.nan
curvature = np.nan
smile_noise = np.nan
rr25 = np.nan          # Risk Reversal 25Δ
bf25 = np.nan          # Butterfly 25Δ
tail_pressure = np.nan # Deep tails: put vs call wings

# 1) Slope, curvature, and noise (quadratic fit in log-moneyness)
valid_skew = calls.dropna(subset=['log_moneyness', 'impliedVolatility'])
if len(valid_skew) > 4:
    try:
        res = np.polyfit(valid_skew['log_moneyness'], valid_skew['impliedVolatility'], 2)
        curvature = res[0]
        slope = res[1]

        fitted = np.polyval(res, valid_skew['log_moneyness'])
        smile_noise = np.sqrt(((valid_skew['impliedVolatility'] - fitted) ** 2).mean())
    except:
        pass

# 2) Risk Reversal 25Δ and Butterfly 25Δ (if we have deltas)
if (
    not calls.empty and not puts.empty
    and 'delta' in calls.columns and 'delta' in puts.columns
):
    try:
        call25 = calls.iloc[(calls['delta'] - 0.25).abs().argsort().iloc[0]]
        put25  = puts.iloc[(puts['delta'] + 0.25).abs().argsort().iloc[0]]  # delta ~ -0.25

        rr25 = call25['impliedVolatility'] - put25['impliedVolatility']
        atm_iv_local = atm_iv  # calculated previously
        bf25 = (call25['impliedVolatility'] + put25['impliedVolatility']) / 2 - atm_iv_local

        # 3) Tail pressure: deep OTM puts vs deep OTM calls
        deep_puts = puts[puts['strike'] < current_price * 0.7]
        deep_calls = calls[calls['strike'] > current_price * 1.3]
        if not deep_puts.empty and not deep_calls.empty:
            tail_pressure = deep_puts['impliedVolatility'].mean() - deep_calls['impliedVolatility'].mean()
    except:
        pass

# Realized & Prem
realized_vol = np.nan
vol_premium = np.nan
if not hist.empty:
    rets = np.log(hist['Close'] / hist['Close'].shift(1))
    realized_vol = rets.rolling(window=realized_vol_lookback).std().iloc[-1] * np.sqrt(252)
    vol_premium = atm_iv - realized_vol

# ====== Metric Cards ======
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Spot Price</div>
        <div class="metric-value">${current_price:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">ATM Implied Vol</div>
        <div class="metric-value">{atm_iv:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    # Determine color and arrow based on premium/discount
    if pd.isna(vol_premium):
        vp_color = "#9ca3af" # Gray for NaN
        vp_arrow = ""
    elif vol_premium >= 0:
        vp_color = "#10b981" # Green
        vp_arrow = "↑"
    else:
        vp_color = "#ef4444" # Red
        vp_arrow = "↓"

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Realized Vol</div>
        <div class="metric-value">{realized_vol:.1%}</div>
        <div class="metric-sub" style="color: {vp_color}">
            {vp_arrow} {vol_premium:.1%} Prem
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Exp. Move (±1σ)</div>
        <div class="metric-value">±${exp_move:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# --- Tabs ---
# Added "Quant Metrics" as the 5th tab
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Volatility Surface", 
    "Liquidity", 
    "Term Structure", 
    "Arbitrage",
    "Quant Metrics" 
])

with tab1:
    st.subheader("Volatility Surface / Skew")

    st.markdown("""
    This chart shows **implied volatility by strike** for calls and puts on the selected expiration.

    - The **blue dashed line** marks the current spot price (or ATM in log-moneyness).
    - The **shaded area** highlights the expected **±1σ move** based on ATM IV.
    - The **shape of the curves** (smile / smirk) reflects how the market prices downside vs upside tail risk.
    """)

    # X-axis Toggle
    x_choice = st.radio(
        "X-axis scale",
        ("Strike", "Log-moneyness"),
        horizontal=True,
        key="skew_xaxis_mode"
    )
    x_mode = "strike" if x_choice == "Strike" else "logm"

    if not calls.empty:
        fig_skew = plot_volatility_skew_plotly(
            T=T,
            expiration=expiration,
            calls=calls,
            puts=puts,
            current_price=current_price,
            lower_band=current_price - exp_move,
            upper_band=current_price + exp_move,
            atm_iv=atm_iv,
            days_to_expiry=days_to_expiry,
            template=plotly_template,
            x_mode=x_mode,
        )

        st.plotly_chart(fig_skew, use_container_width=True)

        if x_mode == "strike":
            st.caption(
                "Strike-based view: useful for intuitive mapping of strikes vs IV. "
                "Focus on the ATM region and how fast IV increases in the downside vs upside."
            )
        else:
            st.caption(
                "Log-moneyness view: normalized smile in terms of ln(K/S). "
                "Negative values = downside (puts OTM), positive values = upside (calls OTM). "
                "Ideal for comparing smiles across tickers and expirations."
            )

with tab2:
    st.subheader("Liquidity & Market Depth")

    st.markdown("""
    This section focuses on **option market liquidity**:

    - **Open Interest:** how much open positioning there is at each strike (position crowding).
    - **Volume:** how actively contracts are trading intraday.
    - **Put/Call Balance:** whether traders are leaning more towards protection (puts) or upside (calls).

    Higher liquidity usually means **tighter spreads** and **better execution**.
    """)

    if not calls.empty and not puts.empty:
        # ---- Quick Metrics ----
        liq1, liq2, liq3 = st.columns(3)

        median_oi = float(calls["openInterest"].median())
        median_vol = float(calls["volume"].median())
        put_call_oi_ratio = (
            puts["openInterest"].sum() / calls["openInterest"].sum()
            if calls["openInterest"].sum() > 0 else float("nan")
        )

        with liq1:
            st.metric("Median Call OI", f"{median_oi:,.0f}")

        with liq2:
            st.metric("Median Call Volume", f"{median_vol:,.0f}")

        with liq3:
            st.metric("Put/Call OI Ratio", f"{put_call_oi_ratio:.2f}")

        st.markdown("---")

        # ---- Main Charts (Plotly) ----
        g1, g2 = st.columns(2)

        with g1:
            st.plotly_chart(
                plot_liquidity_oi_plotly(calls, puts, current_price, template=plotly_template),
                use_container_width=True
            )
            st.caption(
                "Open Interest highlights where the market is most positioned. "
                "Clusters around certain strikes often act as **magnet levels** for price."
            )

        with g2:
            st.plotly_chart(
                plot_liquidity_volume_plotly(calls, puts, current_price, template=plotly_template),
                use_container_width=True
            )
            st.caption(
                "Volume shows where trading activity is concentrated intraday. "
                "High volume + high OI → **high-liquidity strikes** for execution."
            )

        st.markdown("### Liquidity Bubble Map")

        st.plotly_chart(
            plot_liquidity_bubblemap(calls, puts, template=plotly_template),
            use_container_width=True
        )
        st.caption(
            "Each bubble combines **open interest (Y-axis)** and **trading activity (size = volume)**. "
            "Color shows the **Put/Call OI ratio** (blue ≈ call-heavy, red ≈ put-heavy). "
            "Large, high bubbles with neutral color ≈ best liquidity for execution."
        )
    else:
        st.info("No option liquidity data available for this expiration.")

with tab3:
    st.subheader("Term Structure & Skew Dynamics")

    st.markdown("""
    This section tracks how **implied volatility** and **skew** evolve across maturities:

    - **ATM IV Term Structure:** how expensive options are across different expirations (contango/backwardation).
    - **Skew Slope & Curvature:** how much the market is paying for downside vs upside tails and how “U-shaped” the smile is.
    """)

    exp_scan = expirations[:term_structure_n]
    atm_df = calculate_atm_term_structure(ticker, exp_scan, current_price)
    skew_df = calculate_skew_term_structure(ticker, exp_scan, current_price)

    # ---- Sparklines (Top Mini-charts) ----
    if not atm_df.empty or not skew_df.empty:
        sp1, sp2 = st.columns(2)
        with sp1:
            if not atm_df.empty:
                st.markdown("**ATM IV Trend (Sparkline)**")
                st.plotly_chart(
                    plot_atm_sparkline(atm_df, template=plotly_template),
                    use_container_width=True
                )
        with sp2:
            if not skew_df.empty:
                st.markdown("**Skew Slope Trend (Sparkline)**")
                st.plotly_chart(
                    plot_skew_sparkline(skew_df, template=plotly_template),
                    use_container_width=True
                )

    st.markdown("---")

    # ---- Main Charts ----
    col_a, col_b = st.columns(2)

    with col_a:
        if not atm_df.empty:
            st.plotly_chart(
                plot_atm_term_structure_plotly(atm_df, template=plotly_template),
                use_container_width=True
            )
            st.caption(
                "The **term structure** shows whether the market is pricing more risk in the short end "
                "or the long end (contango vs. backwardation)."
            )

    with col_b:
        if not skew_df.empty:
            st.plotly_chart(
                plot_skew_term_structure_plotly(skew_df, template=plotly_template),
                use_container_width=True
            )
            st.caption(
                "The **skew slope** reflects demand for crash protection vs upside calls, while "
                "**curvature** captures how much the wings (deep OTM options) are being bid up."
            )

with tab4:

    # ---- SUBTITLE ----
    st.subheader("Quantitative Arbitrage Scanner")

    # ---- DESCRIPTION ----
    st.markdown("### What this tool does")
    st.markdown("""
    This scanner compares **Real-Time Market Prices** against 
    **Theoretical Black–Scholes Prices** to detect potential arbitrage opportunities.
    """)

    # ---- ARBITRAGE LOGIC ----
    st.markdown('<div class="arb-section-title">Arbitrage Logic</div>', unsafe_allow_html=True)

    # --- CASE 1: OVERVALUED (SELL) ---
    with st.container():
        st.markdown("""
        <div class="info-card sell-card">
            <div class="small-label">Overvaluation · Potential SELL</div>
            <div class="card-title"> Positive Arbitrage Difference</div>
            <div class="card-subtitle">
                The market price is <b>higher</b> than the theoretical price.
            </div>
            <ul>
                <li><b>P<sub>market</sub> &gt; P<sub>theoretical</sub></b></li>
                <li>The option may be <b>overvalued</b> → potential <b>SELL</b> signal.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\text{Arb\_Diff} = P_{\text{market}} - P_{\text{theoretical}} > 0")

    # --- CASE 2: UNDERVALUED (BUY) ---
    with st.container():
        st.markdown("""
        <div class="info-card buy-card">
            <div class="small-label">Undervaluation · Potential BUY</div>
            <div class="card-title"> Negative Arbitrage Difference</div>
            <div class="card-subtitle">
                The market price is <b>lower</b> than the theoretical price.
            </div>
            <ul>
                <li><b>P<sub>market</sub> &lt; P<sub>theoretical</sub></b></li>
                <li>The option may be <b>undervalued</b> → potential <b>BUY</b> signal.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\text{Arb\_Diff} = P_{\text{market}} - P_{\text{theoretical}} < 0")

    # ---- BLACK–SCHOLES ----
    st.markdown('<div class="arb-section-title"> Black–Scholes Reference</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown("""
        <div class="info-card bs-card">
            <div class="small-label">Pricing Model</div>
            <div class="card-title">Theoretical Option Price</div>
            <div class="card-subtitle">
                The theoretical price is computed using the Black–Scholes model:
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"P_{\text{theoretical}} = BS(S, K, r, T, \sigma)")

    # ---- ARBITRAGE TABLES ----
    c1, c2 = st.columns(2)
    cols = ['strike', 'lastPrice', 'BS_Price', 'Arb_Diff', 'delta', 'impliedVolatility']

    with c1:
        st.markdown("### CALLS")
        st.dataframe(
            calls[cols]
            .style.format({'BS_Price':'{:.2f}','Arb_Diff':'{:.2f}','delta':'{:.2f}','impliedVolatility':'{:.1%}'})
            .background_gradient(subset=['Arb_Diff'], cmap='RdYlGn_r'),
            height=400
        )

    with c2:
        st.markdown("### PUTS")
        st.dataframe(
            puts[cols]
            .style.format({'BS_Price':'{:.2f}','Arb_Diff':'{:.2f}','delta':'{:.2f}','impliedVolatility':'{:.1%}'})
            .background_gradient(subset=['Arb_Diff'], cmap='RdYlGn_r'),
            height=400
        )

def fmt_pct(x, decimals=2):
    if x is None or isinstance(x, float) and np.isnan(x):
        return "N/A"
    fmt = f"{{:.{decimals}%}}"
    return fmt.format(x)

rr25_fmt          = fmt_pct(rr25, decimals=2)
bf25_fmt          = fmt_pct(bf25, decimals=2)
tail_pressure_fmt = fmt_pct(tail_pressure, decimals=2)
smile_noise_fmt   = fmt_pct(smile_noise, decimals=2)

with tab5:
    st.subheader("Advanced Volatility Metrics")
    st.markdown("Deep dive into the structural components of the volatility surface.")

    # ========== Row 1: Vol Risk Premium & DTE ==========
    r1_c1, r1_c2 = st.columns(2)

    with r1_c1:
        st.markdown(f"""
        <div class="qm-row">
          <div class="qm-card">
            <div class="qm-card-header">
              <div class="qm-card-label">Vol Risk Premium · The Edge</div>
              <div class="qm-card-value">{vol_premium:.1%}</div>
            </div>
            <span class="qm-pill-blue">Selling insurance</span>
            <p style="margin-top:0.35rem; margin-bottom:0.25rem;">
              Why do we care?
            </p>
            <ul>
              <li><b>Positive:</b> IV &gt; RV. Options are expensive → selling strategies (iron condors, covered calls).</li>
              <li><b>Negative:</b> IV &lt; RV. Options are cheap → buying strategies (long straddles, calls/puts directional).</li>
            </ul>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with r1_c2:
        st.markdown(f"""
        <div class="qm-row">
          <div class="qm-card">
            <div class="qm-card-header">
              <div class="qm-card-label">Time to Expiry · The Clock</div>
              <div class="qm-card-value">{days_to_expiry} days</div>
            </div>
            <span class="qm-pill-blue">Gamma / Theta balance</span>
            <p style="margin-top:0.35rem; margin-bottom:0.25rem;">
              Why is time context vital?
            </p>
            <ul>
              <li><b>Gamma risk:</b> high in short DTE (moves are violent).</li>
              <li><b>Theta risk:</b> fastest decay in short DTE.</li>
              <li><b>Context:</b> 20% IV means something different at 2 days vs 2 years.</li>
            </ul>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ========== Row 2: Skew Slope & Curvature ==========
    r2_c1, r2_c2 = st.columns(2)

    with r2_c1:
        st.markdown(f"""
        <div class="qm-row">
          <div class="qm-card">
            <div class="qm-card-header">
              <div class="qm-card-label">Skew Slope · The Fear</div>
              <div class="qm-card-value">{slope:.4f}</div>
            </div>
            <span class="qm-pill-green">Left / right tail pricing</span>
            <p style="margin-top:0.35rem; margin-bottom:0.25rem;">
              Why is the curve tilted?
            </p>
            <ul>
              <li><b>Negative:</b> crash protection. Puts richer than calls (typical for indices like SPY).</li>
              <li><b>Positive:</b> FOMO. Calls richer than puts (typical for meme stocks / crypto rallies).</li>
            </ul>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with r2_c2:
        st.markdown(f"""
        <div class="qm-row">
          <div class="qm-card">
            <div class="qm-card-header">
              <div class="qm-card-label">Curvature · The Tails</div>
              <div class="qm-card-value">{curvature:.4f}</div>
            </div>
            <span class="qm-pill-amber">Tail risk</span>
            <p style="margin-top:0.35rem; margin-bottom:0.25rem;">
              Why is the curve U-shaped?
            </p>
            <ul>
              <li><b>High:</b> market fears extreme moves (fat tails). Deep OTM options are bid up.</li>
              <li><b>Low:</b> market prices closer to Normal returns (thin tails).</li>
            </ul>
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")

    # ========== Row 3: Risk Reversal & Butterfly ==========
    r3_c1, r3_c2 = st.columns(2)

    with r3_c1:
        st.markdown(f"""
        <div class="qm-row">
          <div class="qm-card">
            <div class="qm-card-header">
              <div class="qm-card-label">RR25 · Risk Reversal</div>
              <div class="qm-card-value">{rr25_fmt}</div>
            </div>
            <span class="qm-pill-blue">Call25Δ – Put25Δ</span>
            <p style="margin-top:0.35rem; margin-bottom:0.25rem;">
              Measures the asymmetry between upside calls and downside puts:
            </p>
            <ul>
              <li><b>Positive:</b> upside calls richer → market paying for upside convexity.</li>
              <li><b>Negative:</b> downside puts richer → crash protection in demand.</li>
            </ul>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with r3_c2:
        st.markdown(f"""
        <div class="qm-row">
          <div class="qm-card">
            <div class="qm-card-header">
              <div class="qm-card-label">BF25 · Butterfly</div>
              <div class="qm-card-value">{bf25_fmt}</div>
            </div>
            <span class="qm-pill-green">Wing premium vs ATM</span>
            <p style="margin-top:0.35rem; margin-bottom:0.25rem;">
              Captures how expensive the wings are relative to ATM volatility:
            </p>
            <ul>
              <li><b>High BF:</b> market pays up for convexity (fat tails).</li>
              <li><b>Low BF:</b> wings cheap, smile closer to flat/normal.</li>
            </ul>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ========== Row 4: Tail Pressure & Smile Noise ==========
    r4_c1, r4_c2 = st.columns(2)

    with r4_c1:
        st.markdown(f"""
        <div class="qm-row">
          <div class="qm-card">
            <div class="qm-card-header">
              <div class="qm-card-label">Tail Pressure</div>
              <div class="qm-card-value">{bf25_fmt}</div>
            </div>
            <span class="qm-pill-amber">Deep OTM skew</span>
            <p style="margin-top:0.35rem; margin-bottom:0.25rem;">
              Compares deep OTM puts vs deep OTM calls:
            </p>
            <ul>
              <li><b>Positive:</b> deep puts much richer → downside tail hedging.</li>
              <li><b>Near zero / negative:</b> more balanced or even upside panic.</li>
            </ul>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with r4_c2:
        st.markdown(f"""
        <div class="qm-row">
          <div class="qm-card">
            <div class="qm-card-header">
              <div class="qm-card-label">Smile Noise</div>
              <div class="qm-card-value">{smile_noise_fmt}</div>
            </div>
            <span class="qm-pill-blue">Data quality</span>
            <p style="margin-top:0.35rem; margin-bottom:0.25rem;">
              Root-mean-square error of the smile vs a quadratic fit:
            </p>
            <ul>
              <li><b>Low noise:</b> clean, well-behaved smile → reliable quotes.</li>
              <li><b>High noise:</b> illiquid strikes, crossed markets, stale ticks.</li>
            </ul>
          </div>
        </div>
        """, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption("Built by **Carlos Alonso** | Data: Yahoo Finance")

st.sidebar.markdown("### Connect")
st.sidebar.markdown("[GitHub](https://github.com/carlosalonsose)")
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/carlos-alonso-serrano/)")
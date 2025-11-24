# Quant Volatility Dashboard

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

**An advanced quantitative analytics suite for equity options, volatility surface modeling, and arbitrage screening.**

---

## 📖 Abstract

The **Quant Volatility Dashboard** is a research-grade tool designed to analyze the microstructure of the options market. By leveraging real-time data from Yahoo Finance, this application reconstructs the **Implied Volatility (IV) Surface**, analyzes **Term Structure dynamics**, and screens for pricing anomalies using the **Black-Scholes-Merton (BSM)** framework.

This project serves as a comprehensive interface for:
* **Volatility Surface Construction:** Analysis of Smile/Smirk dynamics across moneyness.
* **Arbitrage Detection:** Identification of violations in put-call parity or model-price deviations.
* **Risk Metrics:** Calculation of second-order Greeks and structural market parameters (VRP, RR25, BF25).

---

## 🧮 Theoretical Framework & Methodology

### 1. Black-Scholes Pricing Model
The theoretical fair value of European options is calculated using the standard Black-Scholes-Merton formula. For a call option $C$ and put option $P$:

$$
C(S, t) = S N(d_1) - K e^{-r(T-t)} N(d_2)
$$

$$
P(S, t) = K e^{-r(T-t)} N(-d_2) - S N(-d_1)
$$

Where:
* $S$: Spot price of the underlying asset.
* $K$: Strike price.
* $r$: Risk-free interest rate (continuously compounded).
* $\sigma$: Implied Volatility (IV).
* $T-t$: Time to maturity (in years).
* $N(\cdot)$: Cumulative distribution function of the standard normal distribution.

### 2. Log-Moneyness Transformation
To ensure cross-asset comparability and normalize the volatility smile, the dashboard utilizes **Log-Moneyness** ($k$) instead of raw strikes:

$$
k = \ln\left(\frac{K}{S}\right)
$$

* $k < 0$: Downside / OTM Puts (ITM Calls)
* $k > 0$: Upside / OTM Calls (ITM Puts)

### 3. Skew Modeling (Polynomial Fit)
The volatility skew is parametrized by fitting a quadratic function to the IV data in log-moneyness space to extract structural metrics:

$$
\sigma_{IV}(k) \approx \beta_0 + \beta_1 k + \beta_2 k^2
$$

* **Slope ($\beta_1$):** Proxies the market's demand for downside protection (Skew).
* **Curvature ($\beta_2$):** Proxies the kurtosis or "fat-tailed" nature of the return distribution.

---

## 📊 Key Modules

### 1. Volatility Surface & Skew
Visualizes the relationship between Implied Volatility and Strike/Moneyness.
* **Features:** Dual-axis plotting (Strike vs. Log-Moneyness), spot price reference, and $\pm 1\sigma$ expected move cones.
* **Goal:** Assess the cost of tail risk and identifying market sentiment (Smile vs. Smirk).

### 2. Liquidity & Market Depth
Analyzes the positioning of market participants.
* **Bubble Map:** A 3D scatter plot representing Strike ($x$), Open Interest ($y$), and Volume (bubble size), colored by Put/Call ratio.
* **Metrics:** Median Open Interest (OI), Volume profiles, and Put/Call imbalances.
* **Goal:** Identify liquidity clusters and "pinning" levels for expiry.

### 3. Term Structure Analysis
Examines volatility across different time horizons.
* **ATM Term Structure:** Plots ATM IV against Days to Expiry (DTE) to detect Contango (normal) or Backwardation (stress) regimes.
* **Skew Dynamics:** Tracks how the Slope and Curvature of the smile evolve as expiration approaches.

### 4. Arbitrage Scanner
A screening tool that computes the spread between the **Market Price** (Last/Mid) and the **Theoretical Model Price**.

$$
\Delta_{\text{Arb}} = P_{\text{Market}} - P_{\text{BSM}}
$$

* **Positive Spread:** Market > Model (Potential Overvaluation / Short Signal).
* **Negative Spread:** Market < Model (Potential Undervaluation / Long Signal).

### 5. Advanced Quant Metrics
A dedicated dashboard for structural volatility metrics:
* **Vol Risk Premium (VRP):** $IV_{\text{ATM}} - RV_{\text{Historical}}$. Measures the premium paid for optionality over realized movement.
* **Risk Reversal (RR25):** $IV_{25\Delta C} - IV_{25\Delta P}$. Measures the directional bias of the market.
* **Butterfly (BF25):** $\frac{IV_{25\Delta C} + IV_{25\Delta P}}{2} - IV_{\text{ATM}}$. Measures the value of the "wings" (kurtosis).
* **Smile Noise:** RMSE of the quadratic fit, serving as a proxy for data cleanliness and market efficiency.

---

## ⚙️ Tech Stack

* **Core:** Python 3.10+
* **UI Framework:** Streamlit
* **Financial Data:** `yfinance` (Yahoo Finance API)
* **Numerical Computing:** `numpy`, `pandas`, `scipy.stats`
* **Visualization:** `plotly.graph_objects`, `plotly.express`, `matplotlib`

---

## 🚀 Installation & Usage

### Prerequisites
Ensure you have Python installed (version 3.8 or higher is recommended).

### 1. Clone the repository
```bash
git clone [https://github.com/carlosalonsose/quant-volatility-dashboard.git](https://github.com/carlosalonsose/quant-volatility-dashboard.git)
cd quant-volatility-dashboard

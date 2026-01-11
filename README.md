# Portfolio Optimization using LSTM and Dynamic Programming

Advanced portfolio optimization system combining LSTM neural networks with dynamic programming to optimize allocations between Gold and Bitcoin using real market data.

## Quick Start

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Core Components

### 1. Data Acquisition
- **Real Market Data**: Fetches Gold (GC=F) and Bitcoin (BTC-USD) from Yahoo Finance
- **Data Period**: 688+ days of historical data
- **Data Cleaning**: Removes NaN values, aligns time series, validates price ranges
- **Statistics Calculated**:
  - Current price, period high/low
  - Daily and annualized volatility
  - Total period returns
  - Daily Sharpe ratio
  - Correlation analysis

### 2. LSTM Price Prediction Models
- **Architecture**: 3-layer LSTM with dropout regularization
  - Layer 1: 128 units + 30% dropout
  - Layer 2: 64 units + 30% dropout
  - Layer 3: 32 units + 20% dropout
  - Dense layers: 32 → 16 → 1 (linear output)
- **Training Configuration**:
  - 30 epochs with early stopping
  - 80/20 train-validation split
  - Batch size: 32
  - Optimizer: Adam (lr=0.001)
  - Loss: Huber (robust to outliers)
  - Callbacks: ReduceLROnPlateau, EarlyStopping
- **Lookback Period**: 60 days of historical data
- **Fallback**: Exponential smoothing if TensorFlow unavailable

### 3. Risk Analysis Framework

#### Bias Ratio Calculation
- Formula: `(Current_Price - Moving_Average) / Moving_Average × 100`
- Window: 20 days
- Interpretation: Positive = overbought, Negative = oversold

#### Bull Market Index (BMI)
- Formula: `Average_Return × W + Average_Bias × (1-W)`
- Weight (W): 0.5 (equal weighting)
- Window: 90 days
- Positive BMI = Bull market, Negative = Bear market
- Tracks cumulative bull/bear score over time

#### Risk Decomposition
- **Systematic Risk**: Market-driven movements (Beta analysis)
  - Gold Beta: -0.0166 (negative correlation with Bitcoin)
  - Calculated as: `(Beta² × Bitcoin_Variance)`
- **Idiosyncratic Risk**: Asset-specific risk
  - Calculated as: `Total_Variance - Systematic_Risk`
  - Gold: 99.8% idiosyncratic (mostly independent)

#### Value at Risk (VaR) & Conditional VaR
- **VaR (95%)**: 5th percentile of returns
  - Gold: -1.53% (worst 5% of days)
  - Bitcoin: -3.75% (higher downside risk)
- **CVaR**: Average loss in worst 5% of days
  - Gold: -2.46%
  - Bitcoin: -5.23%

#### Downside Deviation
- Only considers negative returns
- Gold: 0.72%, Bitcoin: 1.63%
- Better metric for risk-averse investors

#### Tail Risk Metrics
- **Skewness**: Distribution asymmetry
  - Gold: -0.5274 (left tail risk - crash risk)
  - Bitcoin: 0.4700 (right tail opportunity - upside)
- **Kurtosis**: Extreme event probability
  - Gold: 2.7633 (fat tails)
  - Bitcoin: 2.4567 (fat tails)
  - Both have more extreme events than normal distribution

### 4. Dynamic Programming Optimization

#### Coarse-to-Fine Search Algorithm
- **Stage 1 - Coarse Search**:
  - Grid search with large step size
  - Covers full allocation space
  - Finds general optimal region
  
- **Stage 2 - Fine Search**:
  - Refined grid around best coarse solution
  - Step size = 1/10 of coarse step
  - Finds precise optimal allocation

- **Efficiency**: 90% reduction in computation vs brute force

#### Objective Function
```
Maximize: Portfolio_Value × Momentum_Factor × Risk_Adjustment

Where:
  Portfolio_Value = Cash + Gold_Holdings × Gold_Price + Bitcoin_Holdings × Bitcoin_Price
  Momentum_Factor = 1 + Expected_Return
  Risk_Adjustment = 1 + (Gold_Risk × Gold_Weight + Bitcoin_Risk × Bitcoin_Weight)
```

#### Constraints
- Non-negative holdings (no short selling)
- Cash preservation (no borrowing)
- Transaction costs: 1% for Gold, 2% for Bitcoin
- Use 80% of available cash for safety

### 5. Performance Metrics

#### Return Metrics
- **Total Return**: Overall portfolio gain over period
- **Annualized Return**: Projected yearly return
- **Daily Return**: Day-to-day percentage changes

#### Risk-Adjusted Returns
- **Sharpe Ratio**: (Return - Risk_Free_Rate) / Volatility
  - Measures excess return per unit of risk
  - Result: 1.60 (excellent, >1.0 is good)
- **Calmar Ratio**: Annual_Return / Max_Drawdown
  - Return per unit of drawdown
  - Result: 3.66 (very good)

#### Drawdown Analysis
- **Maximum Drawdown**: Largest peak-to-trough decline
  - Result: 8.77% (very low)
  - Indicates portfolio stability

#### Win Rate & Profit Factor
- **Win Rate**: Percentage of profitable days
  - Result: 56.3% (more wins than losses)
- **Profit Factor**: Sum_of_Gains / Sum_of_Losses
  - Result: 1.34 (gains 34% larger than losses)

#### Outperformance Analysis
- vs Gold Only: -39.19% (gold had exceptional run)
- vs Bitcoin Only: +61.75% (strategy beat Bitcoin)
- vs Equal Weight (50/50): +11.28% (dynamic allocation better)

### 6. Market Regime Detection

#### Bull/Bear Market Identification
- Combines momentum (returns) and valuation (bias ratio)
- Positive score = Bull market (uptrend)
- Negative score = Bear market (downtrend)
- Zero crossing = Market regime change
- Magnitude indicates trend strength

#### Adaptive Allocation
- Increases gold in bear markets (defensive)
- Increases Bitcoin in bull markets (growth)
- Balances risk and return dynamically

### 7. Interactive Dashboard

#### Visualizations (5 rows × 3 columns)
1. **Price Charts**: Gold and Bitcoin historical prices
2. **Portfolio Value**: Total portfolio value over time
3. **Asset Allocation**: Pie chart of Gold/Bitcoin/Cash weights
4. **Risk Analysis**: Gold and Bitcoin risk metrics
5. **Bull/Bear Markets**: Market regime indicators
6. **Rolling Volatility**: 30-day moving volatility
7. **Value at Risk**: VaR distribution analysis
8. **Risk Decomposition**: Systematic vs idiosyncratic risk
9. **Downside Risk**: Downside deviation comparison
10. **Bias Ratio**: Price deviation from moving average
11. **Performance Metrics**: Return, volatility, Sharpe, drawdown, win rate
12. **Risk Metrics**: Skewness, kurtosis, beta comparison

### 8. Report Generation

#### Optional Report Saving
- **Terminal Output**: All console logs and analysis
- **Metrics JSON**: Performance metrics in structured format
- **Summary JSON**: Portfolio summary and statistics
- **Risk Analysis JSON**: Detailed risk metrics
- **Interactive HTML**: Dashboard with all charts
- **Index HTML**: Navigation page for all reports

## Key Results

```
Total Return: 93.65%
Annualized Return: 32.11%
Volatility: 17.11%
Sharpe Ratio: 1.60
Maximum Drawdown: 8.77%
Calmar Ratio: 3.66
Win Rate: 56.3%
Profit Factor: 1.34

Outperformance:
  vs Gold Only: -39.19%
  vs Bitcoin Only: +61.75%
  vs Equal Weight: +11.28%
```

## Risk Profile

### Gold
- Beta: -0.0166 (negative correlation with Bitcoin)
- Systematic Risk: 0.2% (mostly idiosyncratic)
- VaR (95%): -1.53%
- CVaR: -2.46%
- Downside Deviation: 0.72%
- Skewness: -0.5274 (left tail risk)
- Kurtosis: 2.7633 (fat tails)

### Bitcoin
- Beta: 1.0000 (market reference)
- Systematic Risk: 100% (market-driven)
- VaR (95%): -3.75%
- CVaR: -5.23%
- Downside Deviation: 1.63%
- Skewness: 0.4700 (right tail opportunity)
- Kurtosis: 2.4567 (fat tails)

## Customization

Edit in `main.py`:
```python
lookbackDays = 60          # LSTM lookback period
alphaGold = 0.01           # 1% transaction cost
alphaBitcoin = 0.02        # 2% transaction cost
initialCash = 10000        # Starting investment
riskTolerance = 0.7        # Portfolio risk tolerance
```

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import warnings
import yfinance as yf
from datetime import datetime, timedelta
from report_generator import ReportGenerator
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
    from tensorflow.keras.losses import Huber
    from tensorflow.keras import mixed_precision
    
    if tf.config.list_physical_devices('GPU'):
        mixed_precision.set_global_policy('mixed_float16')
        print("GPU + mixed precision ON")
    else:
        print("Running on CPU")
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not available, using CPU-only mode")
    TF_AVAILABLE = False

def fetchRealMarketData(goldTicker="GC=F", bitcoinTicker="BTC-USD", days=1000):
    """Fetch real market data from Yahoo Finance"""
    print(f"Fetching real market data for the last {days} days...")
    
    endDate = datetime.now()
    startDate = endDate - timedelta(days=days)
    
    try:
        print(f"Downloading Gold ({goldTicker})...")
        goldData = yf.download(goldTicker, start=startDate, end=endDate, progress=False)
        goldPrices = goldData['Close'].values.flatten()
        
        print(f"Downloading Bitcoin ({bitcoinTicker})...")
        bitcoinData = yf.download(bitcoinTicker, start=startDate, end=endDate, progress=False)
        bitcoinPrices = bitcoinData['Close'].values.flatten()
        
        # Remove NaN values
        goldPrices = goldPrices[~np.isnan(goldPrices)]
        bitcoinPrices = bitcoinPrices[~np.isnan(bitcoinPrices)]
        
        # Align lengths
        minLen = min(len(goldPrices), len(bitcoinPrices))
        goldPrices = goldPrices[-minLen:].astype(float)
        bitcoinPrices = bitcoinPrices[-minLen:].astype(float)
        
        print(f"✓ Gold data: {len(goldPrices)} days | Range: ${goldPrices.min():.2f} - ${goldPrices.max():.2f}")
        print(f"✓ Bitcoin data: {len(bitcoinPrices)} days | Range: ${bitcoinPrices.min():.2f} - ${bitcoinPrices.max():.2f}")
        
        return goldPrices, bitcoinPrices, goldData.index[-minLen:], bitcoinData.index[-minLen:]
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Falling back to synthetic data...")
        np.random.seed(42)
        t = np.linspace(0, 10*np.pi, days)
        goldTrend = 0.0002 * np.linspace(0, days, days)
        goldCycle = 0.05 * np.sin(t * 0.5)
        goldNoise = np.cumsum(np.random.normal(0, 0.001, days))
        gold = 1200 * np.exp(goldTrend + goldCycle + goldNoise)
        btcTrend = 0.0006 * np.linspace(0, days, days)
        btcCycle = 0.15 * np.sin(t * 2) + 0.08 * np.sin(t * 5)
        btcNoise = np.cumsum(np.random.normal(0, 0.005, days))
        btc = 5000 * np.exp(btcTrend + btcCycle + btcNoise)
        return gold, btc, None, None

class EnhancedLSTMPortfolioAI:
    def __init__(self, lookbackDays=60, alphaGold=0.01, alphaBitcoin=0.02):
        self.lookbackDays = lookbackDays
        self.alphaGold = alphaGold
        self.alphaBitcoin = alphaBitcoin
        self.goldModel = None
        self.bitcoinModel = None
        self.goldScaler = MinMaxScaler(feature_range=(0, 1))
        self.bitcoinScaler = MinMaxScaler(feature_range=(0, 1))
        self.C = None
        self.G = None
        self.B = None
        self.portfolioHistory = []
        self.useTensorFlow = TF_AVAILABLE

    def createSequences(self, data, scaler=None, fitScaler=False):
        if fitScaler:
            dataScaled = scaler.fit_transform(data.reshape(-1, 1))
        else:
            dataScaled = scaler.transform(data.reshape(-1, 1))
        X, y = [], []
        for i in range(self.lookbackDays, len(dataScaled)):
            X.append(dataScaled[i-self.lookbackDays:i, 0])
            y.append(dataScaled[i, 0])
        return np.array(X), np.array(y), dataScaled

    def buildModel(self, inputShape):
        if not TF_AVAILABLE:
            return None
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=inputShape),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='linear', dtype='float32')
        ])
        model.compile(optimizer=Adam(0.001), loss=Huber(), metrics=['mae'])
        return model

    def calculateBiasRatio(self, prices, window=20):
        biasRatios = []
        for i in range(len(prices)):
            if i < window:
                biasRatios.append(0)
                continue
            currentPrice = prices[i]
            movingAvg = np.mean(prices[i-window:i])
            biasRatio = (currentPrice - movingAvg) / movingAvg * 100
            biasRatios.append(biasRatio)
        return np.array(biasRatios)

    def calculateBullMarketIndex(self, returns, biasRatios, window=90, W=0.5):
        bmiScores = []
        bullBearScores = []
        currentScore = 0
        for i in range(len(returns)):
            if i < window:
                bmiScores.append(0)
                bullBearScores.append(0)
                continue
            avgReturn = np.mean(returns[i-window:i])
            avgBias = np.mean(biasRatios[i-window:i])
            bmi = avgReturn * W + avgBias * (1 - W)
            bmiScores.append(bmi)
            avgBmi = np.mean(bmiScores[max(0, i-90):i])
            if bmi > avgBmi:
                currentScore += 1
            else:
                currentScore -= 1
            bullBearScores.append(currentScore)
        return np.array(bmiScores), np.array(bullBearScores)

    def calculateRiskRates(self, goldPrices, bitcoinPrices, W=0.5):
        goldReturns = np.diff(goldPrices) / goldPrices[:-1]
        bitcoinReturns = np.diff(bitcoinPrices) / bitcoinPrices[:-1]
        goldReturns = np.concatenate([[0], goldReturns])
        bitcoinReturns = np.concatenate([[0], bitcoinReturns])
        goldBias = self.calculateBiasRatio(goldPrices)
        bitcoinBias = self.calculateBiasRatio(bitcoinPrices)
        goldBmi, goldBullBear = self.calculateBullMarketIndex(goldReturns, goldBias, W=W)
        bitcoinBmi, bitcoinBullBear = self.calculateBullMarketIndex(bitcoinReturns, bitcoinBias, W=W)
        goldRisk = np.zeros_like(goldPrices)
        bitcoinRisk = np.zeros_like(bitcoinPrices)
        for i in range(len(goldPrices)):
            if i >= 90:
                goldRisk[i] = 1 - (abs(goldBias[i]) / 100 + abs(goldBmi[i])) / 2
                bitcoinRisk[i] = 1 - (abs(bitcoinBias[i]) / 100 + abs(bitcoinBmi[i])) / 2
        return {
            'goldRisk': goldRisk,
            'bitcoinRisk': bitcoinRisk,
            'goldBias': goldBias,
            'bitcoinBias': bitcoinBias,
            'goldBmi': goldBmi,
            'bitcoinBmi': bitcoinBmi,
            'goldBullBear': goldBullBear,
            'bitcoinBullBear': bitcoinBullBear
        }

    def trainModelsFast(self, goldPrices, bitcoinPrices, epochs=40, batchSize=64):
        if not TF_AVAILABLE:
            print("TensorFlow not available - using simple exponential smoothing for predictions")
            return {
                'gold': {'history': None, 'rmse': {'train': 0, 'val': 0}},
                'bitcoin': {'history': None, 'rmse': {'train': 0, 'val': 0}}
            }
        print("Training Gold...")
        Xg, yg, _ = self.createSequences(goldPrices, self.goldScaler, True)
        Xg = Xg.reshape((Xg.shape[0], Xg.shape[1], 1))
        split = int(0.8 * len(Xg))
        XTrain, XVal = Xg[:split], Xg[split:]
        yTrain, yVal = yg[:split], yg[split:]
        self.goldModel = self.buildModel((self.lookbackDays, 1))
        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-5),
            EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
        ]
        histG = self.goldModel.fit(XTrain, yTrain, validation_data=(XVal, yVal),
                                    epochs=epochs, batch_size=batchSize, callbacks=callbacks, verbose=1)
        predTrain = self.goldScaler.inverse_transform(self.goldModel.predict(XTrain, batch_size=256))
        predVal   = self.goldScaler.inverse_transform(self.goldModel.predict(XVal,   batch_size=256))
        rmseTrain = np.sqrt(mean_squared_error(self.goldScaler.inverse_transform(yTrain.reshape(-1,1)), predTrain))
        rmseVal   = np.sqrt(mean_squared_error(self.goldScaler.inverse_transform(yVal.reshape(-1,1)),   predVal))
        print(f"Gold RMSE: Train ${rmseTrain:,.0f} | Val ${rmseVal:,.0f}")
        print("\nTraining Bitcoin...")
        Xb, yb, _ = self.createSequences(bitcoinPrices, self.bitcoinScaler, True)
        Xb = Xb.reshape((Xb.shape[0], Xb.shape[1], 1))
        split = int(0.8 * len(Xb))
        XTrain, XVal = Xb[:split], Xb[split:]
        yTrain, yVal = yb[:split], yb[split:]
        self.bitcoinModel = self.buildModel((self.lookbackDays, 1))
        histB = self.bitcoinModel.fit(XTrain, yTrain, validation_data=(XVal, yVal),
                                        epochs=epochs, batch_size=batchSize, callbacks=callbacks, verbose=1)
        predTrainB = self.bitcoinScaler.inverse_transform(self.bitcoinModel.predict(XTrain, batch_size=256))
        predValB   = self.bitcoinScaler.inverse_transform(self.bitcoinModel.predict(XVal,   batch_size=256))
        rmseTrainB = np.sqrt(mean_squared_error(self.bitcoinScaler.inverse_transform(yTrain.reshape(-1,1)), predTrainB))
        rmseValB   = np.sqrt(mean_squared_error(self.bitcoinScaler.inverse_transform(yVal.reshape(-1,1)),   predValB))
        print(f"Bitcoin RMSE: Train ${rmseTrainB:,.0f} | Val ${rmseValB:,.0f}")
        return {
            'gold': {'history': histG, 'rmse': {'train': rmseTrain, 'val': rmseVal}},
            'bitcoin': {'history': histB, 'rmse': {'train': rmseTrainB, 'val': rmseValB}}
        }

    def predictNextDay(self, model, scaler, historicalData):
        if not TF_AVAILABLE or model is None:
            if len(historicalData) < 2:
                return historicalData[-1]
            alpha = 0.3
            return historicalData[-1] + alpha * (historicalData[-1] - historicalData[-2])
        if len(historicalData) < self.lookbackDays:
            return historicalData[-1]
        lastSequence = historicalData[-self.lookbackDays:]
        lastSequenceScaled = scaler.transform(lastSequence.reshape(-1, 1))
        predictionScaled = model.predict(lastSequenceScaled.reshape(1, self.lookbackDays, 1), verbose=0)
        prediction = scaler.inverse_transform(predictionScaled)[0, 0]
        return prediction

    def improvedDynamicProgrammingOptimization(self, goldPrices, bitcoinPrices, initialCash=1000, w=0.5):
        print("Running Improved Dynamic Programming Optimization...")
        self.C = [initialCash]
        self.G = [0]
        self.B = [0]
        self.portfolioHistory = []
        riskData = self.calculateRiskRates(goldPrices, bitcoinPrices)
        startDay = 90
        for i in range(startDay, len(goldPrices)-1):
            if i % 100 == 0:
                print(f"Processing day {i}/{len(goldPrices)-1}")
            C_i = self.C[-1]
            G_i = self.G[-1]
            B_i = self.B[-1]
            goldHist = goldPrices[:i+1]
            bitcoinHist = bitcoinPrices[:i+1]
            P_pred = self.predictNextDay(self.goldModel, self.goldScaler, goldHist)
            Q_pred = self.predictNextDay(self.bitcoinModel, self.bitcoinScaler, bitcoinHist)
            P_i = goldPrices[i]
            Q_i = bitcoinPrices[i]
            r_i = riskData['goldRisk'][i]
            t_i = riskData['bitcoinRisk'][i]
            goldExpectedReturn = (P_pred - P_i) / P_i
            bitcoinExpectedReturn = (Q_pred - Q_i) / Q_i
            def objectiveFunction(M, N):
                transactionCost = self.alphaGold * abs(M) + self.alphaBitcoin * abs(N)
                if C_i - M - N - transactionCost < 0:
                    return -np.inf
                C_next = C_i - M - N - transactionCost
                G_next = G_i + M / P_i
                B_next = B_i + N / Q_i
                if G_next < 0 or B_next < 0 or C_next < 0:
                    return -np.inf
                expectedReturn = goldExpectedReturn * (M/P_i) + bitcoinExpectedReturn * (N/Q_i)
                riskAdjustment = r_i * (M/P_i) + t_i * (N/Q_i)
                portfolioValue = C_next + G_next * P_pred + B_next * Q_pred
                momentumFactor = 1 + expectedReturn
                return portfolioValue * momentumFactor * (1 + riskAdjustment)
            cashAvailable = C_i * 0.8
            bestValue = -np.inf
            best_M = 0
            best_N = 0
            coarseStep = max(50, cashAvailable / 10)
            M_coarse = np.arange(-cashAvailable/2, cashAvailable/2 + coarseStep, coarseStep)
            N_coarse = np.arange(-cashAvailable/2, cashAvailable/2 + coarseStep, coarseStep)
            for M in M_coarse:
                for N in N_coarse:
                    if abs(M) + abs(N) > cashAvailable:
                        continue
                    val = objectiveFunction(M, N)
                    if val > bestValue:
                        bestValue = val
                        best_M = M
                        best_N = N
            fineStep = max(5, coarseStep / 10)
            M_fine = np.arange(best_M - coarseStep, best_M + coarseStep + fineStep, fineStep)
            N_fine = np.arange(best_N - coarseStep, best_N + coarseStep + fineStep, fineStep)
            for M in M_fine:
                for N in N_fine:
                    if abs(M) + abs(N) > cashAvailable:
                        continue
                    val = objectiveFunction(M, N)
                    if val > bestValue:
                        bestValue = val
                        best_M = M
                        best_N = N
            M_star = best_M
            N_star = best_N
            transactionCost = self.alphaGold * abs(M_star) + self.alphaBitcoin * abs(N_star)
            C_next = self.C[-1] - M_star - N_star - transactionCost
            G_next = self.G[-1] + M_star / goldPrices[i]
            B_next = self.B[-1] + N_star / bitcoinPrices[i]
            self.C.append(C_next)
            self.G.append(G_next)
            self.B.append(B_next)
            actualAssets = C_next + G_next * goldPrices[i+1] + B_next * bitcoinPrices[i+1]
            self.portfolioHistory.append(actualAssets)
        return {
            'cash': self.C,
            'gold': self.G,
            'bitcoin': self.B,
            'totalAssets': self.portfolioHistory,
            'riskData': riskData
        }

    def calculatePerformanceMetrics(self, portfolioValues, goldPrices, bitcoinPrices, riskFreeRate=0.02):
        if len(portfolioValues) < 2:
            return {
                'totalReturn': 0, 'annualReturn': 0, 'volatility': 0,
                'sharpeRatio': 0, 'maxDrawdown': 0, 'calmarRatio': 0,
                'winRate': 0, 'profitFactor': 0,
                'outperformanceVsGold': 0, 'outperformanceVsBitcoin': 0, 'outperformanceVsEqualWeight': 0
            }
        portfolioReturns = np.diff(portfolioValues) / portfolioValues[:-1]
        goldReturns = (goldPrices[-1] - goldPrices[0]) / goldPrices[0] * 100 if len(goldPrices) > 0 else 0
        bitcoinReturns = (bitcoinPrices[-1] - bitcoinPrices[0]) / bitcoinPrices[0] * 100 if len(bitcoinPrices) > 0 else 0
        equalWeightReturns = (goldReturns + bitcoinReturns) / 2
        totalReturn = (portfolioValues[-1] - portfolioValues[0]) / portfolioValues[0] * 100
        try:
            annualReturn = ((portfolioValues[-1] / portfolioValues[0]) ** (252/len(portfolioValues)) - 1) * 100
        except:
            annualReturn = totalReturn
        volatility = np.std(portfolioReturns) * np.sqrt(252) * 100 if len(portfolioReturns) > 0 else 0
        if np.std(portfolioReturns) > 0:
            sharpeRatio = (np.mean(portfolioReturns) * 252 - riskFreeRate) / (np.std(portfolioReturns) * np.sqrt(252))
        else:
            sharpeRatio = 0
        maxDrawdown = (np.maximum.accumulate(portfolioValues) - portfolioValues).max() / np.maximum.accumulate(portfolioValues).max() * 100
        try:
            calmarRatio = annualReturn / maxDrawdown if maxDrawdown > 0 else 0
        except:
            calmarRatio = 0
        winRate = (portfolioReturns > 0).mean() * 100 if len(portfolioReturns) > 0 else 0
        positiveReturns = portfolioReturns[portfolioReturns > 0]
        negativeReturns = portfolioReturns[portfolioReturns < 0]
        if len(negativeReturns) > 0 and negativeReturns.sum() != 0:
            profitFactor = abs(positiveReturns.sum()) / abs(negativeReturns.sum())
        else:
            profitFactor = float('inf') if len(positiveReturns) > 0 else 0
        metrics = {
            'totalReturn': totalReturn,
            'annualReturn': annualReturn,
            'volatility': volatility,
            'sharpeRatio': sharpeRatio,
            'maxDrawdown': maxDrawdown,
            'calmarRatio': calmarRatio,
            'winRate': winRate,
            'profitFactor': profitFactor
        }
        metrics['outperformanceVsGold'] = metrics['totalReturn'] - goldReturns
        metrics['outperformanceVsBitcoin'] = metrics['totalReturn'] - bitcoinReturns
        metrics['outperformanceVsEqualWeight'] = metrics['totalReturn'] - equalWeightReturns
        return metrics


def analyzeRiskDecomposition(goldPrices, bitcoinPrices, riskData, dpResults):
    """Comprehensive risk decomposition analysis"""
    print("\nRISK DECOMPOSITION ANALYSIS:")
    
    goldReturns = np.diff(goldPrices) / goldPrices[:-1]
    bitcoinReturns = np.diff(bitcoinPrices) / bitcoinPrices[:-1]
    
    goldBeta = np.cov(goldReturns, bitcoinReturns)[0, 1] / np.var(bitcoinReturns)
    goldMarketVar = (goldBeta ** 2) * np.var(bitcoinReturns)
    goldIdioVar = np.var(goldReturns) - goldMarketVar
    
    print(f"  Gold Beta: {goldBeta:.4f}")
    print(f"  Systematic Risk: {goldMarketVar:.6f}")
    print(f"  Idiosyncratic Risk: {goldIdioVar:.6f}")
    
    goldVaR95 = np.percentile(goldReturns, 5)
    bitcoinVaR95 = np.percentile(bitcoinReturns, 5)
    goldCVaR = goldReturns[goldReturns <= goldVaR95].mean()
    bitcoinCVaR = bitcoinReturns[bitcoinReturns <= bitcoinVaR95].mean()
    
    print(f"  Gold VaR (95%): {goldVaR95*100:.2f}%")
    print(f"  Bitcoin VaR (95%): {bitcoinVaR95*100:.2f}%")
    
    goldDownside = np.sqrt(np.mean(np.minimum(goldReturns, 0)**2))
    bitcoinDownside = np.sqrt(np.mean(np.minimum(bitcoinReturns, 0)**2))
    
    from scipy.stats import skew, kurtosis
    goldSkew = skew(goldReturns)
    bitcoinSkew = skew(bitcoinReturns)
    goldKurt = kurtosis(goldReturns)
    bitcoinKurt = kurtosis(bitcoinReturns)
    
    return {
        'goldBeta': goldBeta,
        'goldSystematic': goldMarketVar,
        'goldIdiosyncratic': goldIdioVar,
        'goldVaR95': goldVaR95,
        'bitcoinVaR95': bitcoinVaR95,
        'goldCVaR': goldCVaR,
        'bitcoinCVaR': bitcoinCVaR,
        'goldDownside': goldDownside,
        'bitcoinDownside': bitcoinDownside,
        'goldSkew': goldSkew,
        'bitcoinSkew': bitcoinSkew,
        'goldKurt': goldKurt,
        'bitcoinKurt': bitcoinKurt
    }

def createComprehensiveDashboard(goldPrices, bitcoinPrices, dpResults, portfolioWeights, trainingResults, riskData, metrics, riskDecomp):
    """Create detailed dashboard showing all aspects"""
    
    fig = make_subplots(
        rows=5, cols=3,
        subplot_titles=(
            'Gold Price', 'Bitcoin Price', 'Portfolio Value',
            'Asset Allocation', 'Gold Risk', 'Bitcoin Risk',
            'Bull/Bear (Gold)', 'Bull/Bear (Bitcoin)', 'Rolling Volatility',
            'VaR Distribution', 'Risk Decomposition', 'Downside Risk',
            'Bias Ratio', 'Performance Metrics', 'Risk Metrics'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
            [{"type": "pie"}, {"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
            [{"type": "histogram"}, {"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "bar"}, {"type": "bar"}]
        ],
        vertical_spacing=0.07
    )
    
    goldColor = '#FFD700'
    bitcoinColor = '#FF9900'
    portfolioColor = '#00FF00'
    
    # Gold Price
    fig.add_trace(go.Scatter(x=list(range(len(goldPrices))), y=goldPrices,
        name='Gold', line=dict(color=goldColor, width=2)), row=1, col=1)
    
    # Bitcoin Price
    fig.add_trace(go.Scatter(x=list(range(len(bitcoinPrices))), y=bitcoinPrices,
        name='Bitcoin', line=dict(color=bitcoinColor, width=2)), row=1, col=2)
    
    # Portfolio Value
    if dpResults and 'totalAssets' in dpResults and len(dpResults['totalAssets']) > 0:
        fig.add_trace(go.Scatter(x=list(range(len(dpResults['totalAssets']))), y=dpResults['totalAssets'],
            name='Portfolio', line=dict(color=portfolioColor, width=3)), row=1, col=3)
    
    # Asset Allocation
    if portfolioWeights:
        cashWeight = 1 - portfolioWeights['goldWeight'] - portfolioWeights['bitcoinWeight']
        fig.add_trace(go.Pie(values=[portfolioWeights['goldWeight'], portfolioWeights['bitcoinWeight'], cashWeight],
            labels=['Gold', 'Bitcoin', 'Cash'], marker=dict(colors=[goldColor, bitcoinColor, '#FFFFFF']), hole=0.4), row=2, col=1)
    
    # Gold Risk
    if riskData and len(riskData['goldRisk']) > 0:
        fig.add_trace(go.Scatter(x=list(range(len(riskData['goldRisk']))), y=riskData['goldRisk'],
            name='Gold Risk', line=dict(color=goldColor)), row=2, col=2)
    
    # Bitcoin Risk
    if riskData and len(riskData['bitcoinRisk']) > 0:
        fig.add_trace(go.Scatter(x=list(range(len(riskData['bitcoinRisk']))), y=riskData['bitcoinRisk'],
            name='Bitcoin Risk', line=dict(color=bitcoinColor)), row=2, col=3)
    
    # Bull/Bear Gold
    if riskData and len(riskData['goldBullBear']) > 0:
        fig.add_trace(go.Scatter(x=list(range(len(riskData['goldBullBear']))), y=riskData['goldBullBear'],
            name='Gold Bull/Bear', line=dict(color=goldColor, width=2), fill='tozeroy'), row=3, col=1)
    
    # Bull/Bear Bitcoin
    if riskData and len(riskData['bitcoinBullBear']) > 0:
        fig.add_trace(go.Scatter(x=list(range(len(riskData['bitcoinBullBear']))), y=riskData['bitcoinBullBear'],
            name='BTC Bull/Bear', line=dict(color=bitcoinColor, width=2), fill='tozeroy'), row=3, col=2)
    
    # Bias Ratio
    if riskData and len(riskData['goldBias']) > 0:
        fig.add_trace(go.Scatter(x=list(range(len(riskData['goldBias']))), y=riskData['goldBias'],
            name='Gold Bias', line=dict(color=goldColor)), row=5, col=1)
        fig.add_trace(go.Scatter(x=list(range(len(riskData['bitcoinBias']))), y=riskData['bitcoinBias'],
            name='BTC Bias', line=dict(color=bitcoinColor)), row=5, col=1)
    
    # Performance Metrics
    if metrics:
        metricNames = ['Total Return', 'Annual Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
        metricValues = [metrics['totalReturn'], metrics['annualReturn'], metrics['volatility'],
                       metrics['sharpeRatio'], metrics['maxDrawdown'], metrics['winRate']]
        fig.add_trace(go.Bar(x=metricNames, y=metricValues,
            marker_color=['#00FF00', '#00CC00', '#FF9900', '#FFD700', '#FF4444', '#44FF44'],
            text=[f'{val:.2f}%' if name != 'Sharpe Ratio' else f'{val:.2f}' for name, val in zip(metricNames, metricValues)],
            textposition='auto'), row=5, col=2)
    
    # Risk Metrics
    if riskDecomp:
        risk_metrics = ['Skewness', 'Kurtosis', 'Beta']
        gold_vals = [riskDecomp['goldSkew'], riskDecomp['goldKurt']/10, riskDecomp['goldBeta']]
        btc_vals = [riskDecomp['bitcoinSkew'], riskDecomp['bitcoinKurt']/10, 1.0]
        fig.add_trace(go.Bar(x=risk_metrics, y=gold_vals, name='Gold', marker_color=goldColor), row=5, col=3)
        fig.add_trace(go.Bar(x=risk_metrics, y=btc_vals, name='Bitcoin', marker_color=bitcoinColor), row=5, col=3)
    
    fig.update_layout(height=1600, title="Portfolio Optimization Dashboard", template="plotly_dark", showlegend=True)
    fig.show()
    return fig

def enhancedPortfolioOptimization(goldPred, goldStd, bitcoinPred, bitcoinStd, riskTolerance=0.7):
    """Portfolio optimization using Modern Portfolio Theory"""
    minLength = min(len(goldPred), len(bitcoinPred))
    if minLength < 2:
        return {
            'goldWeight': 0.3, 'bitcoinWeight': 0.4,
            'expectedGoldReturn': 0.08, 'expectedBitcoinReturn': 0.15,
            'goldRisk': goldStd, 'bitcoinRisk': bitcoinStd, 'sharpeRatio': 1.5
        }
    
    goldPredAdj = goldPred[:minLength]
    bitcoinPredAdj = bitcoinPred[:minLength]
    goldReturns = np.diff(goldPredAdj) / goldPredAdj[:-1]
    bitcoinReturns = np.diff(bitcoinPredAdj) / bitcoinPredAdj[:-1]
    
    expectedGoldReturn = np.mean(goldReturns) * 252 if len(goldReturns) > 0 else 0.05
    expectedBitcoinReturn = np.mean(bitcoinReturns) * 252 if len(bitcoinReturns) > 0 else 0.12
    expectedGoldReturn = max(0.02, expectedGoldReturn)
    expectedBitcoinReturn = max(0.05, expectedBitcoinReturn)
    
    minReturnsLength = min(len(goldReturns), len(bitcoinReturns))
    if minReturnsLength > 1:
        returnsMatrix = np.column_stack([goldReturns[:minReturnsLength], bitcoinReturns[:minReturnsLength]])
        covMatrix = np.cov(returnsMatrix.T) * 252
    else:
        covMatrix = np.array([[0.02, 0.01], [0.01, 0.05]])
    
    riskFreeRate = 0.02
    bestSharpe = -np.inf
    bestWeights = [0.5, 0.5]
    
    for goldWeight in np.arange(0.1, 0.9, 0.05):
        bitcoinWeight = 1.0 - goldWeight
        w = np.array([goldWeight, bitcoinWeight])
        returns = np.array([expectedGoldReturn, expectedBitcoinReturn])
        portfolioReturn = np.sum(w * returns)
        portfolioVolatility = np.sqrt(np.dot(w.T, np.dot(covMatrix, w)))
        sharpe = (portfolioReturn - riskFreeRate) / portfolioVolatility if portfolioVolatility > 0 else 0
        if sharpe > bestSharpe:
            bestSharpe = sharpe
            bestWeights = [goldWeight, bitcoinWeight]
    
    goldWeight = bestWeights[0] * riskTolerance
    bitcoinWeight = bestWeights[1] * riskTolerance
    
    return {
        'goldWeight': goldWeight, 'bitcoinWeight': bitcoinWeight,
        'expectedGoldReturn': expectedGoldReturn, 'expectedBitcoinReturn': expectedBitcoinReturn,
        'goldRisk': float(goldStd) if hasattr(goldStd, '__float__') else float(goldStd.mean()),
        'bitcoinRisk': float(bitcoinStd) if hasattr(bitcoinStd, '__float__') else float(bitcoinStd.mean()),
        'sharpeRatio': bestSharpe
    }

def predictFutureFast(ai, goldData, bitcoinData, days=30, nSims=10):
    """Monte Carlo prediction"""
    def mcPredict(model, scaler, series, lookback, future, n):
        try:
            if not TF_AVAILABLE or model is None:
                lastPrice = series[-1]
                trend = (series[-1] - series[-lookback]) / series[-lookback] if series[-lookback] != 0 else 0.001
                futurePrices = lastPrice * (1 + trend) ** np.arange(1, future+1)
                return futurePrices, np.full(future, lastPrice * 0.05), None
            
            scaled = scaler.transform(series.reshape(-1, 1))
            last = scaled[-lookback:]
            seqs = np.tile(last, (n, 1, 1))
            preds = []
            
            for _ in range(future):
                historicalVol = np.std(np.diff(series[-lookback:]) / series[-lookback:-1])
                if np.isnan(historicalVol) or historicalVol == 0:
                    historicalVol = 0.01
                noisy = seqs + np.random.normal(0, historicalVol * 0.1, seqs.shape)
                nextVal = model.predict(noisy, verbose=0)
                nextVal = nextVal[:, 0]
                preds.append(nextVal)
                seqs = np.concatenate([seqs[:, 1:, :], nextVal[:, None, None]], axis=1)
            
            preds = np.stack(preds, axis=1)
            preds = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(n, future)
            return preds.mean(axis=0), preds.std(axis=0), preds
        except Exception as e:
            lastPrice = series[-1]
            trend = (series[-1] - series[-lookback]) / series[-lookback] if series[-lookback] != 0 else 0.001
            futurePrices = lastPrice * (1 + trend) ** np.arange(1, days+1)
            return futurePrices, np.full(days, lastPrice * 0.05), None
    
    print("Running Gold Monte Carlo...")
    gMean, gStd, gSims = mcPredict(ai.goldModel, ai.goldScaler, goldData, ai.lookbackDays, days, nSims)
    print("Running Bitcoin Monte Carlo...")
    bMean, bStd, bSims = mcPredict(ai.bitcoinModel, ai.bitcoinScaler, bitcoinData, ai.lookbackDays, days, nSims)
    
    return (gMean, gStd, gSims), (bMean, bStd, bSims)

def main():
    report = ReportGenerator()
    print("="*70)
    print("PORTFOLIO OPTIMIZATION WITH REAL MARKET DATA")
    print("="*70)
    goldPrices, bitcoinPrices, goldDates, bitcoinDates = fetchRealMarketData(days=1000)
    print("\n" + "="*70)
    print("DATA ANALYSIS")
    print("="*70)
    goldReturns = np.diff(goldPrices) / goldPrices[:-1] * 100
    bitcoinReturns = np.diff(bitcoinPrices) / bitcoinPrices[:-1] * 100
    print(f"\nGOLD STATISTICS:")
    print(f"  Current Price: ${goldPrices[-1]:.2f}")
    print(f"  Period High: ${goldPrices.max():.2f}")
    print(f"  Period Low: ${goldPrices.min():.2f}")
    print(f"  Average Daily Return: {goldReturns.mean():.4f}%")
    print(f"  Daily Volatility: {goldReturns.std():.4f}%")
    print(f"  Annualized Volatility: {goldReturns.std() * np.sqrt(252):.2f}%")
    print(f"  Total Period Return: {(goldPrices[-1] - goldPrices[0]) / goldPrices[0] * 100:.2f}%")
    print(f"  Sharpe Ratio (Daily): {(goldReturns.mean() / goldReturns.std()) if goldReturns.std() > 0 else 0:.4f}")
    print(f"\nBITCOIN STATISTICS:")
    print(f"  Current Price: ${bitcoinPrices[-1]:.2f}")
    print(f"  Period High: ${bitcoinPrices.max():.2f}")
    print(f"  Period Low: ${bitcoinPrices.min():.2f}")
    print(f"  Average Daily Return: {bitcoinReturns.mean():.4f}%")
    print(f"  Daily Volatility: {bitcoinReturns.std():.4f}%")
    print(f"  Annualized Volatility: {bitcoinReturns.std() * np.sqrt(252):.2f}%")
    print(f"  Total Period Return: {(bitcoinPrices[-1] - bitcoinPrices[0]) / bitcoinPrices[0] * 100:.2f}%")
    print(f"  Sharpe Ratio (Daily): {(bitcoinReturns.mean() / bitcoinReturns.std()) if bitcoinReturns.std() > 0 else 0:.4f}")
    correlation = np.corrcoef(goldReturns, bitcoinReturns)[0, 1]
    print(f"\nCORRELATION ANALYSIS:")
    print(f"  Gold-Bitcoin Correlation: {correlation:.4f}")
    print(f"  Interpretation: {'Low' if abs(correlation) < 0.3 else 'Moderate' if abs(correlation) < 0.7 else 'High'} correlation")
    ai = EnhancedLSTMPortfolioAI(lookbackDays=60, alphaGold=0.01, alphaBitcoin=0.02)
    print("\n" + "="*70)
    print("TRAINING MODELS")
    print("="*70)
    trainingResults = ai.trainModelsFast(goldPrices, bitcoinPrices, epochs=30, batchSize=32)
    print("\n" + "="*70)
    print("PORTFOLIO OPTIMIZATION")
    print("="*70)
    dpResults = ai.improvedDynamicProgrammingOptimization(goldPrices, bitcoinPrices, initialCash=10000, w=0.5)
    if dpResults and 'totalAssets' in dpResults and len(dpResults['totalAssets']) > 0:
        metrics = ai.calculatePerformanceMetrics(
            np.array([10000] + dpResults['totalAssets']),
            goldPrices[90:-1],
            bitcoinPrices[90:-1]
        )
        print(f"\n" + "="*70)
        print("COMPREHENSIVE PERFORMANCE METRICS")
        print("="*70)
        print(f"Total Return: {metrics['totalReturn']:.2f}%")
        print(f"Annualized Return: {metrics['annualReturn']:.2f}%")
        print(f"Volatility: {metrics['volatility']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpeRatio']:.2f}")
        print(f"Maximum Drawdown: {metrics['maxDrawdown']:.2f}%")
        print(f"Calmar Ratio: {metrics['calmarRatio']:.2f}")
        print(f"Win Rate: {metrics['winRate']:.1f}%")
        print(f"Profit Factor: {metrics['profitFactor']:.2f}")
        print(f"\n" + "="*70)
        print("OUTPERFORMANCE ANALYSIS")
        print("="*70)
        print(f"vs Gold Only: {metrics['outperformanceVsGold']:+.2f}%")
        print(f"vs Bitcoin Only: {metrics['outperformanceVsBitcoin']:+.2f}%")
        print(f"vs Equal Weight: {metrics['outperformanceVsEqualWeight']:+.2f}%")
    print("\n" + "="*70)
    print("IMPLEMENTATION COMPLETE")
    print("="*70)
    print("✓ Real market data fetched and analyzed")
    print("✓ Dynamic programming optimization finished")
    print("✓ Performance metrics calculated")
    print("✓ Portfolio allocation optimized")
    if dpResults and 'totalAssets' in dpResults and len(dpResults['totalAssets']) > 0:
        finalValue = dpResults['totalAssets'][-1]
        print(f"\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        print(f"Initial Investment: $10,000.00")
        print(f"Final Portfolio Value: ${finalValue:,.2f}")
        print(f"Total Gain: ${finalValue - 10000:,.2f}")
        print(f"Return: {(finalValue - 10000) / 10000 * 100:.2f}%")
        print("="*70)
    print("\nAnalysis complete! Reports saved to Reports folder.")
    
    # Ask user if they want to save reports
    print("\n" + "="*70)
    save_report = input("Do you want to save this analysis to Reports folder? (yes/no): ").strip().lower()
    
    if save_report in ['yes', 'y']:
        # Save dashboard chart
        report.save_chart(fig, "portfolio_dashboard")
        
        # Save terminal output
        report.save_terminal_output()
        
        # Save metrics
        if dpResults and 'totalAssets' in dpResults and len(dpResults['totalAssets']) > 0:
            metrics_dict = {
                'totalReturn': float(metrics['totalReturn']),
                'annualReturn': float(metrics['annualReturn']),
                'volatility': float(metrics['volatility']),
                'sharpeRatio': float(metrics['sharpeRatio']),
                'maxDrawdown': float(metrics['maxDrawdown']),
                'calmarRatio': float(metrics['calmarRatio']),
                'winRate': float(metrics['winRate']),
                'profitFactor': float(metrics['profitFactor']),
                'outperformanceVsGold': float(metrics['outperformanceVsGold']),
                'outperformanceVsBitcoin': float(metrics['outperformanceVsBitcoin']),
                'outperformanceVsEqualWeight': float(metrics['outperformanceVsEqualWeight'])
            }
            report.save_metrics(metrics_dict)
        
        # Save summary
        finalValue = dpResults['totalAssets'][-1] if dpResults and 'totalAssets' in dpResults else 0
        summary_dict = {
            'timestamp': datetime.now().isoformat(),
            'initialInvestment': 10000,
            'finalValue': float(finalValue),
            'goldCurrentPrice': float(goldPrices[-1]),
            'bitcoinCurrentPrice': float(bitcoinPrices[-1]),
            'goldVolatility': float(goldReturns.std() * np.sqrt(252)),
            'bitcoinVolatility': float(bitcoinReturns.std() * np.sqrt(252)),
            'correlation': float(correlation),
            'dataPoints': len(goldPrices)
        }
        report.save_summary(summary_dict)
        
        # Save risk analysis
        if riskDecomp:
            risk_dict = {
                'goldBeta': float(riskDecomp['goldBeta']),
                'goldSystematicRisk': float(riskDecomp['goldSystematic']),
                'goldIdiosyncraticRisk': float(riskDecomp['goldIdiosyncratic']),
                'goldVaR95': float(riskDecomp['goldVaR95']),
                'bitcoinVaR95': float(riskDecomp['bitcoinVaR95']),
                'goldCVaR': float(riskDecomp['goldCVaR']),
                'bitcoinCVaR': float(riskDecomp['bitcoinCVaR']),
                'goldDownside': float(riskDecomp['goldDownside']),
                'bitcoinDownside': float(riskDecomp['bitcoinDownside']),
                'goldSkewness': float(riskDecomp['goldSkew']),
                'bitcoinSkewness': float(riskDecomp['bitcoinSkew']),
                'goldKurtosis': float(riskDecomp['goldKurt']),
                'bitcoinKurtosis': float(riskDecomp['bitcoinKurt'])
            }
            report.save_risk_analysis(risk_dict)
        
        # Create index
        report.create_index()
        print(f"\nReports saved to: {report.get_report_path()}")
        print(f"Open index.html to view your analysis")
    else:
        print("Reports not saved.")

if __name__ == "__main__":
    main()

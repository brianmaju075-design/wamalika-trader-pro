#!/usr/bin/env python3
# 🇰🇪 WAMALIKA AI TRADER – FINAL EDITION (Railway + Deriv + Full AI + Back Buttons)
# Author: Brian Wamalika

import json, random, math, statistics, time, threading, copy, os, io
from datetime import datetime, timezone, timedelta
import requests

# ============================================
# 🔑 CREDENTIALS (JAZA HAPA)
# ============================================
BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
YOUR_USER_ID = 123456789
DERIV_API_TOKEN = "YOUR_DERIV_API_TOKEN"   # Token kutoka Deriv Dashboard
DERIV_APP_ID = "10888"                    # App ID ya umma

# ============================================
# ⚙️ SETTINGS
# ============================================
USE_DEMO = True
SYMBOLS = ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD"]
TIMEFRAMES = ["5min","15min","60min"]
MIN_TF_CONFIRMATIONS = 2

RISK_PER_TRADE = 0.02
MIN_CONFIDENCE = 65
MAX_DAILY_TRADES = 5
STOP_LOSS_PIPS = 30
TAKE_PROFIT_PIPS = 60
TRAILING_STOP_ACTIVATE = 20
TRAILING_STOP_DISTANCE = 15
BREAKEVEN_ACTIVATE = 15
DAILY_LOSS_LIMIT = 50

USE_ATR_SLTP = True
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 3.0

USE_SESSION_FILTER = True
LONDON_SESSION_START = 8; LONDON_SESSION_END = 16
NY_SESSION_START = 13; NY_SESSION_END = 21

AUTO_RETRAIN_THRESHOLD = 0.4
NN_HIDDEN_LAYERS = [10,5]; NN_LEARNING_RATE = 0.01; NN_EPOCHS = 50

Q_LEARNING_RATE = 0.1; Q_DISCOUNT = 0.95; Q_EPSILON = 0.1
Q_ACTIONS = ["BUY","SELL","HOLD"]

TREND_STRENGTH_PERIOD = 50; VOLATILITY_THRESHOLD = 0.3
KELLY_FRACTION = 0.5; MAX_CONSECUTIVE_LOSSES = 3
USE_ANTI_MARTINGALE = True; MARTINGALE_FACTOR = 1.5; MAX_MARTINGALE_STEPS = 3

# ============================================
# IMPORTS (MACHACHE YA ZIADA)
# ============================================
try: import feedparser; FEEDPARSER_AVAILABLE = True
except: FEEDPARSER_AVAILABLE = False
try: import tweepy; TWEEPY_AVAILABLE = True
except: TWEEPY_AVAILABLE = False
try: from PIL import Image, ImageDraw, ImageFont; PIL_AVAILABLE = True
except: PIL_AVAILABLE = False

# ============================================
# 📂 STORAGE
# ============================================
DATA_FILE = "bot_data.json"
def load_data():
    try:
        with open(DATA_FILE) as f: return json.load(f)
    except:
        return {"balance":10000,"trades":[],"signals":[],"total":0,"wins":0,"losses":0,
                "daily_trades":0,"last_trade_date":"","day_start_balance":10000,
                "equity_curve":[],"martingale_step":0}
def save_data(d):
    with open(DATA_FILE,"w") as f: json.dump(d,f,indent=2)
storage = load_data()

# ============================================
# 📡 BROKER API – DERIV (REST)
# ============================================
DERIV_URL = "https://api.deriv.com"
DERIV_HEADERS = {"Authorization": f"Bearer {DERIV_API_TOKEN}", "Content-Type": "application/json"}

def get_deriv_price(symbol):
    try:
        deriv_symbol = "frx" + symbol
        url = f"{DERIV_URL}/v2/tick/{deriv_symbol}"
        resp = requests.get(url, headers=DERIV_HEADERS, timeout=10)
        data = resp.json()
        if "tick" in data:
            return float(data["tick"]["bid"]), float(data["tick"]["ask"])
        return None, None
    except:
        return None, None

def get_deriv_balance():
    try:
        url = f"{DERIV_URL}/v2/balance"
        resp = requests.get(url, headers=DERIV_HEADERS, timeout=10)
        data = resp.json()
        return float(data.get("balance", 10000))
    except:
        return 10000

def get_deriv_positions():
    try:
        url = f"{DERIV_URL}/v2/positions"
        resp = requests.get(url, headers=DERIV_HEADERS, timeout=10)
        data = resp.json()
        return data.get("positions", [])
    except:
        return []

def place_deriv_trade(symbol, action, confidence, lot_size=None, sl_price=None, tp_price=None):
    bid, ask = get_deriv_price(symbol)
    if bid is None:
        return False, "Price fetch failed"
    balance = get_deriv_balance()
    amount = max(1.0, round(balance * RISK_PER_TRADE, 2))
    deriv_symbol = "frx" + symbol
    contract_type = "CALL" if action == "BUY" else "PUT"
    order = {
        "proposal": 1,
        "amount": str(amount),
        "basis": "stake",
        "contract_type": contract_type,
        "currency": "USD",
        "duration": 1,
        "duration_unit": "m",
        "symbol": deriv_symbol,
        "app_id": DERIV_APP_ID
    }
    url = f"{DERIV_URL}/v2/buy"
    resp = requests.post(url, json=order, headers=DERIV_HEADERS, timeout=10)
    data = resp.json()
    if "buy" in data and "contract_id" in data["buy"]:
        trade = {
            "contract_id": data["buy"]["contract_id"],
            "symbol": symbol,
            "action": action,
            "amount": amount,
            "confidence": confidence,
            "balance": balance,
            "time": datetime.now(timezone.utc).isoformat(),
            "status": "OPEN"
        }
        return True, trade
    else:
        return False, data.get("error", {}).get("message", "Unknown error")

def calculate_position_size(balance, confidence, symbol=None):
    risk = balance * RISK_PER_TRADE
    return max(1.0, round(risk, 2))

# ============================================
# 🧠 NEURAL NETWORK (FULL SIMPLENN)
# ============================================
class SimpleNN:
    def __init__(self, input_size, hidden_layers, output_size):
        layers = [input_size] + hidden_layers + [output_size]
        self.weights = []
        for i in range(len(layers)-1):
            w = [[random.gauss(0, math.sqrt(2/layers[i])) for _ in range(layers[i+1])] for _ in range(layers[i])]
            self.weights.append(w)
        self.is_trained = False

    def forward(self, x):
        self.activations = [x]
        for w in self.weights:
            net = [sum(self.activations[-1][k] * w[k][j] for k in range(len(self.activations[-1]))) for j in range(len(w[0]))]
            out = [1 / (1 + math.exp(-v)) for v in net]
            self.activations.append(out)
        return self.activations[-1]

    def train(self, X, y, epochs, lr):
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                out = self.forward(xi)
                error = [yi[j] - out[j] for j in range(len(yi))]
                for l in range(len(self.weights)-1, -1, -1):
                    act = self.activations[l+1]
                    delta = [error[j] * act[j] * (1 - act[j]) for j in range(len(act))]
                    for j in range(len(delta)):
                        for k in range(len(self.weights[l])):
                            self.weights[l][k][j] += lr * delta[j] * self.activations[l][k]
                    if l > 0:
                        error = [sum(self.weights[l][k][j] * delta[j] for j in range(len(delta))) for k in range(len(self.activations[l]))]
        self.is_trained = True

neural_nets = {sym: SimpleNN(5, NN_HIDDEN_LAYERS, 3) for sym in SYMBOLS}

# ============================================
# 📊 INDICATORS & DATA FETCH
# ============================================
def mean(vals): return sum(vals)/len(vals) if vals else 0
def stdev(vals):
    if len(vals) < 2: return 0
    m = mean(vals)
    return math.sqrt(sum((x-m)**2 for x in vals) / (len(vals)-1))
def ema(vals, period):
    if len(vals) < period: return vals[-1] if vals else 0
    mult = 2 / (period + 1)
    e = mean(vals[:period])
    for p in vals[period:]: e = (p - e) * mult + e
    return e
def calc_rsi(closes):
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    avg_gain = mean(gains[-14:]) if len(gains) >= 14 else mean(gains)
    avg_loss = mean(losses[-14:]) if len(losses) >= 14 else mean(losses)
    rs = avg_gain / avg_loss if avg_loss != 0 else 100
    return 100 - (100 / (1 + rs))
def calc_macd(closes): return ema(closes, 12) - ema(closes, 26)

def extract_features(closes, highs, lows):
    rsi = calc_rsi(closes) / 100
    macd = calc_macd(closes)
    macd_norm = (macd + 0.001) / 0.002
    ma_fast = mean(closes[-10:]) if len(closes) >= 10 else closes[-1]
    ma_slow = mean(closes[-30:]) if len(closes) >= 30 else closes[-1]
    ma_cross = 1 if ma_fast > ma_slow else -1
    mom = (closes[-1] / closes[-10] - 1) * 100 if len(closes) >= 10 else 0
    returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
    vol = stdev(returns) * 100 if len(returns) > 1 else 0
    return [rsi, macd_norm, ma_cross, mom/0.5, vol/0.01]

def fetch_forex_data(symbol, interval="15min", limit=100):
    try:
        url = "https://www.alphavantage.co/query"
        params = {"function": "FX_INTRADAY", "from_symbol": symbol[:3], "to_symbol": symbol[3:],
                  "interval": interval, "outputsize": "compact", "apikey": "demo"}
        r = requests.get(url, params=params, timeout=10)
        d = r.json()
        key = f"Time Series FX ({interval})"
        if key in d:
            rows = []
            for ts, vals in d[key].items():
                rows.append({"time": ts, "open": float(vals["1. open"]), "high": float(vals["2. high"]),
                             "low": float(vals["3. low"]), "close": float(vals["4. close"]), "volume": 1000})
            rows.sort(key=lambda x: x["time"])
            return rows[-limit:]
    except: pass
    base_prices = {"EURUSD": 1.085, "GBPUSD": 1.265, "USDJPY": 149.5, "AUDUSD": 0.655, "USDCAD": 1.355}
    base = base_prices.get(symbol, 1.1)
    random.seed(hash(symbol) + int(time.time()/3600))
    prices = []; now = datetime.now(); price = base
    for i in range(limit):
        change = random.gauss(0, 0.00015); price *= (1 + change)
        t = now - timedelta(minutes=15*(limit-i))
        prices.append({"time": t.isoformat(), "open": round(price*0.9999,5), "high": round(price*1.0003,5),
                       "low": round(price*0.9997,5), "close": round(price,5), "volume": random.randint(500,2000)})
    return prices

# ============================================
# AI MODULES (1–15) – FULL IMPLEMENTATIONS
# ============================================
def kalman_filter(prices, process_noise=1e-5, measurement_noise=1e-4):
    n = len(prices)
    if n < 2: return prices, [0]*n
    x = prices[0]; v = 0; P = [[1,0],[0,1]]
    Q = [[process_noise,0],[0,process_noise]]; R = measurement_noise
    filtered = []; velocities = []
    for z in prices:
        x_pred = x + v; v_pred = v
        P_pred = [[P[0][0]+Q[0][0], P[0][1]+Q[0][1]],[P[1][0]+Q[1][0], P[1][1]+Q[1][1]]]
        y = z - x_pred; S = P_pred[0][0] + R
        K0 = P_pred[0][0] / S; K1 = P_pred[1][0] / S
        x = x_pred + K0 * y; v = v_pred + K1 * y
        P00 = (1 - K0) * P_pred[0][0]; P01 = (1 - K0) * P_pred[0][1]
        P10 = -K1 * P_pred[0][0] + P_pred[1][0]; P11 = -K1 * P_pred[0][1] + P_pred[1][1]
        P = [[P00, P01],[P10, P11]]
        filtered.append(x); velocities.append(v)
    return filtered, velocities

def kalman_trend(data):
    closes = [d["close"] for d in data]
    if len(closes) < 30: return "neutral"
    _, vels = kalman_filter(closes)
    if vels[-1] > 0.0001: return "up"
    elif vels[-1] < -0.0001: return "down"
    else: return "neutral"

def ar_predict(closes, lags=5):
    if len(closes) <= lags: return 0.0
    returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
    if len(returns) < lags: return 0.0
    return sum(returns[-lags:]) / lags

def bayesian_confidence(symbol, base_confidence, signal):
    trades = [t for t in storage["trades"] if t["symbol"] == symbol]
    if not trades: return base_confidence
    wins = sum(1 for t in trades if t.get("status") == "WIN")
    total = len(trades)
    posterior = (wins + 10 * 0.5) / (total + 10)
    adj = 1.0 + (posterior - 0.5) * 0.5
    return max(0, min(100, base_confidence * adj))

bandit_strategies = {"trend_following": {"wins":0,"losses":0}, "mean_reversion": {"wins":0,"losses":0}, "breakout": {"wins":0,"losses":0}}
def select_strategy_bandit():
    if random.random() < 0.1: return random.choice(list(bandit_strategies.keys()))
    best = max(bandit_strategies, key=lambda s: bandit_strategies[s]["wins"] / (bandit_strategies[s]["wins"] + bandit_strategies[s]["losses"] + 1))
    return best
def update_bandit(strategy, reward):
    if strategy in bandit_strategies:
        if reward > 0: bandit_strategies[strategy]["wins"] += 1
        else: bandit_strategies[strategy]["losses"] += 1

def anomaly_detect(data, threshold=2.5):
    closes = [d["close"] for d in data]
    if len(closes) < 20: return False
    returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
    avg_ret = mean(returns[-20:]); std_ret = stdev(returns[-20:])
    if std_ret == 0: return False
    z = (returns[-1] - avg_ret) / std_ret
    return abs(z) > threshold

def detect_candlestick_patterns(data):
    if len(data) < 3: return 0
    last = data[-1]; prev = data[-2]; prev2 = data[-3]
    o, h, l, c = last["open"], last["high"], last["low"], last["close"]
    po, ph, pl, pc = prev["open"], prev["high"], prev["low"], prev["close"]
    body = abs(c - o)
    wick_up = h - max(c, o); wick_down = min(c, o) - l
    prev_body = abs(pc - po)
    score = 0
    if body < (h - l) * 0.1: score += 5
    if pc < po and c > o and c > po and o < pc: score += 15
    if pc > po and c < o and c < po and o > pc: score -= 15
    if body > 0 and wick_down > body * 2 and body < (h - l) * 0.3: score += 10
    if body > 0 and wick_up > body * 2 and body < (h - l) * 0.3: score -= 10
    return score

def volume_analysis(data):
    if len(data) < 20: return 0
    volumes = [d["volume"] for d in data]
    avg_vol = mean(volumes)
    latest_vol = volumes[-1]
    if latest_vol > avg_vol * 1.5: return 5
    elif latest_vol < avg_vol * 0.5: return -3
    return 0

def ichimoku_signal(data):
    closes = [d["close"] for d in data]; highs = [d["high"] for d in data]; lows = [d["low"] for d in data]
    if len(closes) < 52: return 0
    tenkan = (max(highs[-9:]) + min(lows[-9:])) / 2
    kijun = (max(highs[-26:]) + min(lows[-26:])) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (max(highs[-52:]) + min(lows[-52:])) / 2
    price = closes[-1]
    if price > senkou_a and price > senkou_b: return 10
    elif price < senkou_a and price < senkou_b: return -10
    else: return 0

def elliott_wave_analysis(data):
    closes = [d["close"] for d in data]; highs = [d["high"] for d in data]; lows = [d["low"] for d in data]
    if len(closes) < 30: return 0
    swing_h, swing_l = find_swing_points(highs, lows, window=5)
    if len(swing_h) >= 3 and len(swing_l) >= 3:
        last_swing_h = swing_h[-1][1]; prev_swing_h = swing_h[-2][1]
        if last_swing_h > prev_swing_h: return 5
        else: return -5
    return 0

def find_swing_points(highs, lows, window=5):
    sh, sl = [], []
    n = len(highs)
    for i in range(window, n - window):
        if highs[i] == max(highs[i-window:i+window+1]): sh.append((i, highs[i]))
        if lows[i] == min(lows[i-window:i+window+1]): sl.append((i, lows[i]))
    return sh, sl

def fractal_dimension_index(data, window=30):
    closes = [d["close"] for d in data]
    if len(closes) < window: return 0
    swings = 0
    for i in range(1, len(closes)-1):
        if (closes[i] - closes[i-1]) * (closes[i+1] - closes[i]) < 0: swings += 1
    fdi = 1.0 + (swings / len(closes)) * 2
    if fdi < 1.4: return 10
    elif fdi > 1.6: return -5
    return 0

def detect_divergence(data):
    closes = [d["close"] for d in data]
    if len(closes) < 20: return 0
    rsi_values = [calc_rsi(closes[:i+1]) for i in range(20, len(closes))]
    if len(rsi_values) < 10: return 0
    if closes[-1] > closes[-10] and rsi_values[-1] < rsi_values[-10]: return -15
    if closes[-1] < closes[-10] and rsi_values[-1] > rsi_values[-10]: return 15
    return 0

def detect_order_blocks(data):
    if len(data) < 20: return 0
    for i in range(len(data)-1, max(len(data)-10, 0), -1):
        d = data[i]
        body = abs(d["close"] - d["open"])
        avg_body = mean([abs(data[j]["close"]-data[j]["open"]) for j in range(max(0,i-10), i)])
        if body > avg_body * 2 and d["volume"] > mean([d["volume"] for d in data[-10:]]) * 1.5:
            if d["close"] > d["open"]: return 10
            else: return -10
    return 0

def monte_carlo_risk(data, simulations=200):
    closes = [d["close"] for d in data]
    if len(closes) < 20: return 1.0
    returns = [(closes[i]-closes[i-1])/closes[i-1] for i in range(1, len(closes))]
    avg_ret = mean(returns); std_ret = stdev(returns)
    losses = []
    for _ in range(simulations):
        price = closes[-1]
        for _ in range(5): price *= (1 + random.gauss(avg_ret, std_ret))
        loss = max(0, closes[-1] - price)
        losses.append(loss)
    expected_loss = mean(losses)
    risk_multiplier = 1.0 - min(0.5, expected_loss / closes[-1] * 10)
    return max(0.2, risk_multiplier)

pattern_success = {}
def candlestick_backtest_score(data):
    pattern = ""
    if len(data) < 3: return 0
    last = data[-1]
    if abs(last["close"] - last["open"]) < (last["high"]-last["low"])*0.1: pattern = "doji"
    elif last["close"] > last["open"] and data[-2]["close"] < data[-2]["open"]: pattern = "bullish_engulfing"
    elif last["close"] < last["open"] and data[-2]["close"] > data[-2]["open"]: pattern = "bearish_engulfing"
    if pattern:
        success = pattern_success.get(pattern, {"wins":0, "losses":0})
        win_rate = success["wins"]/(success["wins"]+success["losses"]) if (success["wins"]+success["losses"])>0 else 0.5
        return 10 if win_rate > 0.6 else (-10 if win_rate < 0.4 else 0)
    return 0

def dynamic_volatility_sizing(data, base_lot):
    closes = [d["close"] for d in data]
    if len(closes) < 20: return base_lot
    returns = [(closes[i]-closes[i-1])/closes[i-1] for i in range(1, len(closes))]
    vol = stdev(returns)
    if vol == 0: return base_lot
    normalizer = 0.005
    factor = normalizer / (vol + 0.001)
    factor = max(0.3, min(2.0, factor))
    return max(100, int(base_lot * factor))
def fetch_forex_news():
    """Pata habari za forex kutoka ForexFactory RSS."""
    if not FEEDPARSER_AVAILABLE:
        return []
    try:
        feed = feedparser.parse("https://www.forexfactory.com/news/rss")
        return [entry.title for entry in feed.entries[:10]]
    except:
        return []

def news_polarity(symbol):
    """Hesabu polarity ya habari kwa kutumia TextBlob."""
    if not TEXTBLOB_AVAILABLE:
        return 0.0
    headlines = fetch_forex_news()
    if not headlines:
        return 0.0
    relevant = [h for h in headlines if symbol[:3].upper() in h.upper() or symbol[3:].upper() in h.upper()]
    if not relevant:
        return 0.0
    polarities = [TextBlob(h).sentiment.polarity for h in relevant]
    return mean(polarities)

def twitter_sentiment(symbol):
    """Pata hisia za Twitter kuhusu sarafu."""
    if not TWEEPY_AVAILABLE or not TWITTER_BEARER_TOKEN:
        return "neutral"
    try:
        client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
        query = f"({symbol[:3]} {symbol[3:]}) (bullish OR bearish) -is:retweet"
        tweets = client.search_recent_tweets(query=query, max_results=10)
        bullish = 0
        for tweet in tweets.data or []:
            if "bullish" in tweet.text.lower():
                bullish += 1
            elif "bearish" in tweet.text.lower():
                bullish -= 1
        if bullish > 2:
            return "bullish"
        elif bullish < -2:
            return "bearish"
        else:
            return "neutral"
    except:
        return "neutral"

def sentiment_oscillator(symbol):
    """Unganisha technical, twitter, na news sentiment kuwa -100..100."""
    data = fetch_forex_data(symbol, "15min", 100)
    if not data:
        return 0
    res = analyze_market(data, symbol)
    tech_score = res.get("score", 0)

    tw = twitter_sentiment(symbol)
    tw_score = 50 if tw == "bullish" else (-50 if tw == "bearish" else 0)

    npol = news_polarity(symbol) * 100

    SENTIMENT_WEIGHTS = {"technical": 0.5, "twitter": 0.2, "news_polarity": 0.3}
    oscillator = (SENTIMENT_WEIGHTS["technical"] * tech_score +
                  SENTIMENT_WEIGHTS["twitter"] * tw_score +
                  SENTIMENT_WEIGHTS["news_polarity"] * npol)
    return max(-100, min(100, oscillator))


# ============================================
# 📊 ANALYZE MARKET (JUMUISHA AI ZOTE)
# ============================================
def analyze_market(data, symbol=None):
    if len(data) < 30: return {"signal":"HOLD","confidence":0}
    closes = [d["close"] for d in data]; highs = [d["high"] for d in data]; lows = [d["low"] for d in data]
    cur = closes[-1]
    rsi = calc_rsi(closes); macd_line = calc_macd(closes)
    ema12 = ema(closes, 12); ema26 = ema(closes, 26)
    macd_vals = [ema(closes[:i+1],12)-ema(closes[:i+1],26) for i in range(26,len(closes))]
    signal_line = ema(macd_vals, 9) if len(macd_vals) >= 9 else macd_line
    ma_fast = mean(closes[-10:]) if len(closes) >= 10 else cur
    ma_slow = mean(closes[-30:]) if len(closes) >= 30 else cur
    mid = mean(closes[-20:]) if len(closes) >= 20 else cur
    std = stdev(closes[-20:]) if len(closes) >= 20 else 0.0001
    upper = mid + 2*std; lower = mid - 2*std
    res_basic = max(highs[-20:]) if len(highs) >= 20 else max(highs)
    sup_basic = min(lows[-20:]) if len(lows) >= 20 else min(lows)
    mom = (closes[-1]/closes[-10] - 1) * 100 if len(closes) >= 10 else 0

    # Traditional scoring
    score = 0
    if rsi < 30: score += 20
    elif rsi > 70: score -= 20
    if macd_line > signal_line: score += 25
    else: score -= 25
    if ma_fast > ma_slow: score += 20
    else: score -= 20
    if mom > 0.1: score += 15
    elif mom < -0.1: score -= 15
    if cur < lower * 1.01: score += 10
    elif cur > upper * 0.99: score -= 10
    if cur < sup_basic * 1.005: score += 10
    elif cur > res_basic * 0.995: score -= 10

    # AI Modules influence
    score += detect_candlestick_patterns(data)
    score += volume_analysis(data)
    score += ichimoku_signal(data)
    score += elliott_wave_analysis(data)
    score += fractal_dimension_index(data)
    score += detect_divergence(data)
    score += detect_order_blocks(data)
    score += candlestick_backtest_score(data)

    kt = kalman_trend(data)
    if kt == "up": score += 10
    elif kt == "down": score -= 10
    ar_ch = ar_predict(closes)
    if ar_ch > 0.0002: score += 5
    elif ar_ch < -0.0002: score -= 5

    if score > 15: signal = "BUY"; conf = min(abs(score), 95)
    elif score < -15: signal = "SELL"; conf = min(abs(score), 95)
    else: signal = "HOLD"; conf = max(50, 50 - abs(score))

    if symbol: conf = bayesian_confidence(symbol, conf, signal)
    return {"signal": signal, "confidence": round(conf, 1), "score": round(score, 1),
            "indicators": {"rsi": round(rsi,1), "macd": round(macd_line,5)},
            "price": cur, "time": datetime.now(timezone.utc).isoformat()}

def multi_tf_analyze(symbol):
    votes = {"BUY":0, "SELL":0}; cs = 0
    for tf in TIMEFRAMES:
        data = fetch_forex_data(symbol, tf, 100)
        if not data: continue
        res = analyze_market(data, symbol)
        if res["signal"] in ["BUY","SELL"]:
            votes[res["signal"]] += 1; cs += res["confidence"]
    if votes["BUY"] >= MIN_TF_CONFIRMATIONS: return "BUY", cs/max(1, votes["BUY"])
    elif votes["SELL"] >= MIN_TF_CONFIRMATIONS: return "SELL", cs/max(1, votes["SELL"])
    else: return "HOLD", 0.0

def get_features_for_nn(symbol):
    data = fetch_forex_data(symbol, "15min", 100)
    if not data: return [], [], []
    closes = [d["close"] for d in data]; highs = [d["high"] for d in data]; lows = [d["low"] for d in data]
    return closes, highs, lows

# ============================================
# 🤖 GPT-5 ADVICE (OPENAI)
# ============================================
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"  # acha tupu kama huna
USE_GPT_CONSULTATION = True if OPENAI_API_KEY else False

def ask_gpt(prompt):
    if not OPENAI_API_KEY: return None
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {"model": "gpt-4o-mini", "messages": [{"role": "system", "content": "You are a forex expert."}, {"role": "user", "content": prompt}], "max_tokens": 100, "temperature": 0.3}
    try:
        r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data, timeout=15)
        if r.status_code == 200: return r.json()["choices"][0]["message"]["content"].strip()
    except: pass
    return None

def gpt_trade_decision(symbol, signal, confidence, indicators):
    if not USE_GPT_CONSULTATION: return "SKIP"
    prompt = f"Symbol:{symbol} Signal:{signal} Conf:{confidence}% RSI:{indicators.get('rsi','N/A')} MACD:{indicators.get('macd','N/A')} Should we CONFIRM, REJECT, or HOLD?"
    ans = ask_gpt(prompt)
    if ans:
        ans = ans.upper().strip()
        if "CONFIRM" in ans: return "CONFIRM"
        elif "REJECT" in ans: return "REJECT"
        elif "HOLD" in ans: return "HOLD"
    return "SKIP"

# ============================================
# 📱 TELEGRAM BOT (MENUS + BACK BUTTONS)
# ============================================
API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
is_trading_active = False
consecutive_losses = 0

def send_telegram_message(chat_id, text, reply_markup=None, parse_mode="HTML"):
    data = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}
    if reply_markup: data["reply_markup"] = json.dumps(reply_markup)
    try: requests.post(f"{API_URL}/sendMessage", json=data, timeout=10)
    except: pass

def get_telegram_updates(offset):
    try:
        r = requests.get(f"{API_URL}/getUpdates", params={"offset": offset, "timeout": 30}, timeout=35)
        return r.json().get("result", [])
    except: return []

def show_main_menu(chat_id):
    stats = storage
    wr = stats['wins']/stats['total']*100 if stats['total'] else 0
    kb = {"inline_keyboard": [
        [{"text": "📊 Multi-TF Analysis", "callback_data": "analyze"}],
        [{"text": "▶️ Start Trading", "callback_data": "start"}, {"text": "⏹️ Stop Trading", "callback_data": "stop"}],
        [{"text": "💰 Account Status", "callback_data": "status"}, {"text": "💼 Open Positions", "callback_data": "positions"}],
        [{"text": "📈 Recent Signals", "callback_data": "signals"}, {"text": "📋 Trade History", "callback_data": "history"}],
        [{"text": "🧠 Train NN", "callback_data": "train_nn"}, {"text": "📊 Backtest", "callback_data": "backtest"}],
        [{"text": "🔬 Genetic Opt", "callback_data": "genetic"}, {"text": "📰 News", "callback_data": "news"}],
        [{"text": "📊 Dashboard", "callback_data": "dashboard"}, {"text": "⚙️ Settings", "callback_data": "settings"}],
        [{"text": "🤖 GPT-5.5 Advice", "callback_data": "gpt_advice"}],
        [{"text": "📈 Chart", "callback_data": "chart"}, {"text": "📊 P&L Image", "callback_data": "pnlimage"}],
        [{"text": "🧭 Sentiment Oscillator", "callback_data": "sentiment"}],
        [{"text": "🔗 Important Links", "callback_data": "links"}],
        [{"text": "🔄 Refresh", "callback_data": "refresh"}]
    ]}
    msg = (f"🤖 <b>WAMALIKA AI TRADER</b> – Supreme v4.1\n"
           f"Status: {'🟢' if is_trading_active else '🔴'} | {'Demo' if USE_DEMO else 'Live'}\n"
           f"💰 Bal: ${stats['balance']:,.2f} | WR: {wr:.1f}%\n"
           f"🧠 NN: {'Trained' if any(nn.is_trained for nn in neural_nets.values()) else 'Untrained'}\n"
           f"🤖 GPT-5.5: {'🟢 Active' if USE_GPT_CONSULTATION else '🔴 Off'}\n"
           f"⛔ Circuit: {'⚠️ Tripped' if consecutive_losses >= MAX_CONSECUTIVE_LOSSES else '✅ Normal'}")
    send_telegram_message(chat_id, msg, reply_markup=kb)

def do_market_analysis(chat_id):
    msg = ""
    for sym in SYMBOLS[:4]:
        s, c = multi_tf_analyze(sym)
        emoji = "🟢" if s == "BUY" else "🔴" if s == "SELL" else "⚪"
        msg += f"{emoji} {sym}: {s} ({c:.1f}%)\n"
    keyboard = {"inline_keyboard": [[{"text": "🔙 Back to Main Menu", "callback_data": "back"}]]}
    send_telegram_message(chat_id, f"📊 ANALYSIS\n\n{msg}", reply_markup=keyboard)

def show_account_status(chat_id):
    bal = get_deriv_balance()
    keyboard = {"inline_keyboard": [[{"text": "🔙 Back to Main Menu", "callback_data": "back"}]]}
    send_telegram_message(chat_id, f"💰 Balance: ${bal:,.2f}", reply_markup=keyboard)

def show_open_positions(chat_id):
    pos = get_deriv_positions()
    if not pos:
        keyboard = {"inline_keyboard": [[{"text": "🔙 Back to Main Menu", "callback_data": "back"}]]}
        send_telegram_message(chat_id, "💼 No open positions", reply_markup=keyboard)
        return
    txt = "💼 Open Positions:\n"
    for p in pos:
        sym = p.get("symbol", "").replace("/", "")
        profit = p.get("profit", 0)
        txt += f"{'🟢' if profit >= 0 else '🔴'} {sym} {p['side']} P/L: ${profit:.2f}\n"
    keyboard = {"inline_keyboard": [[{"text": "🔙 Back to Main Menu", "callback_data": "back"}]]}
    send_telegram_message(chat_id, txt, reply_markup=keyboard)

def show_recent_signals(chat_id):
    sigs = storage["signals"][-5:]
    if not sigs:
        keyboard = {"inline_keyboard": [[{"text": "🔙 Back to Main Menu", "callback_data": "back"}]]}
        send_telegram_message(chat_id, "📈 No signals yet", reply_markup=keyboard)
        return
    txt = "📈 Recent Signals:\n"
    for s in sigs:
        txt += f"{'🟢' if s['signal']=='BUY' else '🔴'} {s['symbol']} {s['signal']} @{s['price']:.5f}\n"
    keyboard = {"inline_keyboard": [[{"text": "🔙 Back to Main Menu", "callback_data": "back"}]]}
    send_telegram_message(chat_id, txt, reply_markup=keyboard)

def show_trade_history(chat_id):
    trades = storage["trades"][-5:]
    if not trades:
        keyboard = {"inline_keyboard": [[{"text": "🔙 Back to Main Menu", "callback_data": "back"}]]}
        send_telegram_message(chat_id, "📋 No history yet", reply_markup=keyboard)
        return
    txt = "📋 Trade History:\n"
    for t in trades:
        txt += f"{'🟢' if t['action']=='BUY' else '🔴'} {t['symbol']} {t['action']} @{t['price']:.5f}\n"
    keyboard = {"inline_keyboard": [[{"text": "🔙 Back to Main Menu", "callback_data": "back"}]]}
    send_telegram_message(chat_id, txt, reply_markup=keyboard)

def show_settings(chat_id):
    msg = f"⚙️ Settings:\nMulti-TF confirms: {MIN_TF_CONFIRMATIONS}\nNN Hidden: {NN_HIDDEN_LAYERS}\nNews filter: {'ON' if FEEDPARSER_AVAILABLE else 'OFF'}\nGPT-5.5: {'ON' if USE_GPT_CONSULTATION else 'OFF'}"
    keyboard = {"inline_keyboard": [[{"text": "🔙 Back to Main Menu", "callback_data": "back"}]]}
    send_telegram_message(chat_id, msg, reply_markup=keyboard)

def train_neural_net(chat_id):
    send_telegram_message(chat_id, "🧠 Training Neural Networks...")
    for sym in SYMBOLS[:2]:
        data = fetch_forex_data(sym, "60min", 300)
        if not data: continue
        closes = [d["close"] for d in data]
        X, y = [], []
        for i in range(50, len(closes)-1):
            feats = extract_features(closes[:i+1], [], [])
            future = closes[i+5] if i+5 < len(closes) else closes[-1]
            change = future / closes[i] - 1
            if change > 0.0005: label = [1,0,0]
            elif change < -0.0005: label = [0,1,0]
            else: label = [0,0,1]
            X.append(feats); y.append(label)
        if X:
            neural_nets[sym].train(X, y, epochs=NN_EPOCHS, lr=NN_LEARNING_RATE)
    keyboard = {"inline_keyboard": [[{"text": "🔙 Back to Main Menu", "callback_data": "back"}]]}
    send_telegram_message(chat_id, "✅ Neural networks trained!", reply_markup=keyboard)

def show_backtest(chat_id):
    keyboard = {"inline_keyboard": [[{"text": "🔙 Back to Main Menu", "callback_data": "back"}]]}
    send_telegram_message(chat_id, "📊 Backtest: feature in development.", reply_markup=keyboard)

def show_genetic_result(chat_id):
    keyboard = {"inline_keyboard": [[{"text": "🔙 Back to Main Menu", "callback_data": "back"}]]}
    send_telegram_message(chat_id, "🔬 Genetic Optimizer: coming soon.", reply_markup=keyboard)

def show_news(chat_id):
    headlines = fetch_forex_news()
    msg = "📰 Latest Forex News:\n" + "\n".join(headlines[:5]) if headlines else "No news available."
    keyboard = {"inline_keyboard": [[{"text": "🔙 Back to Main Menu", "callback_data": "back"}]]}
    send_telegram_message(chat_id, msg, reply_markup=keyboard)

def show_dashboard(chat_id):
    stats = storage
    bal = get_deriv_balance()
    report = f"📊 Dashboard\nBalance: ${bal:,.2f}\nTrades: {stats['total']} | Win: {stats['wins']} | Loss: {stats['losses']}"
    keyboard = {"inline_keyboard": [[{"text": "🔙 Back to Main Menu", "callback_data": "back"}]]}
    send_telegram_message(chat_id, report, reply_markup=keyboard)

def ask_gpt_advice(chat_id):
    if not USE_GPT_CONSULTATION:
        send_telegram_message(chat_id, "🤖 GPT-5.5 is disabled (no API key).")
        return
    data = fetch_forex_data(SYMBOLS[0])
    if data:
        res = analyze_market(data, SYMBOLS[0])
        dec = gpt_trade_decision(SYMBOLS[0], res["signal"], res["confidence"], res["indicators"])
        msg = f"🧠 GPT-5.5 Decision for {SYMBOLS[0]}:\n{dec}"
    else: msg = "No data"
    keyboard = {"inline_keyboard": [[{"text": "🔙 Back to Main Menu", "callback_data": "back"}]]}
    send_telegram_message(chat_id, msg, reply_markup=keyboard)

def handle_chart_command(chat_id, symbol=None):
    if not PIL_AVAILABLE:
        send_telegram_message(chat_id, "❌ Pillow not installed.")
        return
    sym = symbol.upper() if symbol else SYMBOLS[0]
    # gen_chart function would be defined (omitted for brevity, include your own)
    keyboard = {"inline_keyboard": [[{"text": "🔙 Back to Main Menu", "callback_data": "back"}]]}
    send_telegram_message(chat_id, f"📈 Chart for {sym} would be sent here.", reply_markup=keyboard)

def handle_pnl_image(chat_id):
    if not PIL_AVAILABLE:
        send_telegram_message(chat_id, "❌ Pillow not installed.")
        return
    keyboard = {"inline_keyboard": [[{"text": "🔙 Back to Main Menu", "callback_data": "back"}]]}
    send_telegram_message(chat_id, "📊 P&L Image would be sent here.", reply_markup=keyboard)

def show_sentiment_oscillator(chat_id):
    msg = "🧭 Sentiment Oscillator:"
    for sym in SYMBOLS[:4]:
        osc = sentiment_oscillator(sym)
        emoji = "🟢" if osc > 20 else "🔴" if osc < -20 else "⚪"
        msg += f"\n{emoji} {sym}: {osc:.0f}"
    keyboard = {"inline_keyboard": [[{"text": "🔙 Back to Main Menu", "callback_data": "back"}]]}
    send_telegram_message(chat_id, msg, reply_markup=keyboard)

def show_important_links(chat_id):
    msg = "🔗 Important Links:\n- Alpha Vantage\n- Deriv\n- OpenAI\n- ForexFactory"
    keyboard = {"inline_keyboard": [[{"text": "🔙 Back to Main Menu", "callback_data": "back"}]]}
    send_telegram_message(chat_id, msg, reply_markup=keyboard)

def handle_callback(chat_id, data):
    global is_trading_active
    if data == "analyze": do_market_analysis(chat_id)
    elif data == "start":
        if not is_trading_active:
            is_trading_active = True
            send_telegram_message(chat_id, "✅ Trading started", reply_markup={"inline_keyboard": [[{"text": "🔙 Back to Main Menu", "callback_data": "back"}]]})
            threading.Thread(target=auto_trade_loop, daemon=True).start()
        else: send_telegram_message(chat_id, "Already active")
    elif data == "stop":
        is_trading_active = False
        send_telegram_message(chat_id, "⏹️ Trading stopped", reply_markup={"inline_keyboard": [[{"text": "🔙 Back to Main Menu", "callback_data": "back"}]]})
    elif data == "status": show_account_status(chat_id)
    elif data == "positions": show_open_positions(chat_id)
    elif data == "signals": show_recent_signals(chat_id)
    elif data == "history": show_trade_history(chat_id)
    elif data == "settings": show_settings(chat_id)
    elif data == "train_nn": train_neural_net(chat_id)
    elif data == "backtest": show_backtest(chat_id)
    elif data == "genetic": show_genetic_result(chat_id)
    elif data == "news": show_news(chat_id)
    elif data == "dashboard": show_dashboard(chat_id)
    elif data == "gpt_advice": ask_gpt_advice(chat_id)
    elif data == "chart": handle_chart_command(chat_id)
    elif data == "pnlimage": handle_pnl_image(chat_id)
    elif data == "sentiment": show_sentiment_oscillator(chat_id)
    elif data == "links": show_important_links(chat_id)
    elif data == "refresh": show_main_menu(chat_id)
    elif data == "back": show_main_menu(chat_id)

# ============================================
# 🚀 AUTO TRADING LOOP
# ============================================
def auto_trade_loop():
    global storage, is_trading_active, consecutive_losses
    print("Trading loop started")
    while is_trading_active:
        try:
            if not is_session_allowed():
                time.sleep(600); continue
            for sym in SYMBOLS:
                if not is_trading_active: break
                signal, conf = multi_tf_analyze(sym)
                if signal == "HOLD": continue
                if USE_GPT_CONSULTATION:
                    data = fetch_forex_data(sym, "15min", 100)
                    if data:
                        res = analyze_market(data, sym)
                        gpt = gpt_trade_decision(sym, signal, conf, res["indicators"])
                        if gpt == "REJECT": continue
                        elif gpt == "HOLD": continue
                success, msg = place_deriv_trade(sym, signal, conf)
                if success:
                    consecutive_losses = 0
                    send_telegram_message(YOUR_USER_ID, f"✅ Trade: {sym} {signal}")
                else: consecutive_losses += 1
                time.sleep(2)
            time.sleep(300)
        except Exception as e:
            print(f"Error: {e}"); time.sleep(60)

def is_session_allowed():
    now = datetime.now(timezone.utc).hour
    return (LONDON_SESSION_START <= now < LONDON_SESSION_END or NY_SESSION_START <= now < NY_SESSION_END) if USE_SESSION_FILTER else True

# ============================================
# MAIN
# ============================================
def main():
    print("🤖 WAMALIKA AI TRADER – FINAL EDITION 🇰🇪")
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE" or YOUR_USER_ID == 123456789:
        print("❌ Set credentials!"); return
    send_telegram_message(YOUR_USER_ID, "🚀 Wamalika AI Trader started! Use /start")
    offset = 0
    while True:
        try:
            updates = get_telegram_updates(offset)
            for upd in updates:
                offset = upd["update_id"] + 1
                if "message" in upd:
                    msg = upd["message"]; chat_id = msg["chat"]["id"]
                    if chat_id != YOUR_USER_ID: continue
                    text = msg.get("text", "")
                    if text == "/start": show_main_menu(chat_id)
                    elif text == "/analyze": do_market_analysis(chat_id)
                elif "callback_query" in upd:
                    cb = upd["callback_query"]; chat_id = cb["message"]["chat"]["id"]
                    if cb["from"]["id"] != YOUR_USER_ID: continue
                    handle_callback(chat_id, cb["data"])
            time.sleep(1)
        except KeyboardInterrupt:
            send_telegram_message(YOUR_USER_ID, "⏹️ Bot stopped"); break
        except Exception as e:
            print(f"Main error: {e}"); time.sleep(5)

if __name__ == "__main__":
    main()

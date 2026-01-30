"""
Gold Trading Bot - GitHub Actions Version
Runs automatically every hour on GitHub's servers
Saves trades to JSON file for persistence
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import os

class GoldTradingBot:
    def __init__(self, initial_capital=10000, risk_per_trade=2.0, data_file='trading_data.json'):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.data_file = data_file
        
        # Load existing data or initialize
        self.load_data()
        
        print("="*80)
        print("🤖 GOLD TRADING BOT - GITHUB ACTIONS")
        print("="*80)
        print(f"Current Capital: ${self.capital:,.2f}")
        print(f"Risk Per Trade: {self.risk_per_trade}%")
        print(f"Total Trades: {len(self.trades)}")
        print("="*80)
    
    def load_data(self):
        """Load trading data from JSON file"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                data = json.load(f)
                self.capital = data.get('capital', self.initial_capital)
                self.trades = data.get('trades', [])
                self.equity_history = data.get('equity_history', [])
                self.position = data.get('position', None)
                
                if self.position:
                    self.position_size = self.position.get('position_size', 0)
                    self.entry_price = self.position.get('entry_price', 0)
                    self.stop_loss = self.position.get('stop_loss', 0)
                    self.take_profit = self.position.get('take_profit', 0)
                    self.entry_time = self.position.get('entry_time', None)
                    self.trade_type = self.position.get('trade_type', None)
                else:
                    self.position_size = 0
                    self.entry_price = 0
                    self.stop_loss = 0
                    self.take_profit = 0
                    self.entry_time = None
                    self.trade_type = None
                    
                print(f"✅ Loaded existing data: {len(self.trades)} trades")
        else:
            self.capital = self.initial_capital
            self.trades = []
            self.equity_history = []
            self.position = None
            self.position_size = 0
            self.entry_price = 0
            self.stop_loss = 0
            self.take_profit = 0
            self.entry_time = None
            self.trade_type = None
            print("✅ Initialized new trading data")
    
    def save_data(self):
        """Save trading data to JSON file"""
        data = {
            'capital': self.capital,
            'trades': self.trades,
            'equity_history': self.equity_history,
            'position': self.position,
            'last_update': datetime.now().isoformat()
        }
        
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Data saved to {self.data_file}")
    
    def fetch_live_data(self, lookback_candles=200):
        """Fetch latest 1-hour candles for Gold"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            print("📡 Fetching Gold price data...")
            df = yf.download('GC=F', start=start_date, end=end_date, interval='1h', progress=False)
            
            if df is not None and len(df) > 0:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                df = df.tail(lookback_candles)
                print(f"✅ Fetched {len(df)} candles")
                return df
            return None
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None
    
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        
        plus_dm = df['High'].diff()
        minus_dm = -df['Low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = ranges.max(axis=1)
        atr = tr.rolling(14).mean()
        
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['ADX'] = dx.rolling(14).mean()
        df['ATR'] = ranges.max(axis=1).rolling(14).mean()
        
        return df
    
    def check_entry_signal(self, df):
        """Check for entry signals"""
        if len(df) < 50:
            return None
        
        latest = df.iloc[-1]
        close = latest['Close']
        ema_9 = latest['EMA_9']
        ema_21 = latest['EMA_21']
        ema_50 = latest['EMA_50']
        rsi = latest['RSI']
        adx = latest['ADX']
        
        if pd.isna(adx) or pd.isna(rsi):
            return None
        
        if (close > ema_9 > ema_21 > ema_50 and rsi > 50 and rsi < 80 and adx > 25):
            return 'BUY'
        
        if (close < ema_9 < ema_21 < ema_50 and rsi < 50 and rsi > 20 and adx > 25):
            return 'SELL'
        
        return None
    
    def calculate_position_size(self, entry_price, stop_loss):
        """Calculate position size based on 2% risk"""
        risk_amount = self.capital * (self.risk_per_trade / 100)
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return 0
        
        position_size = risk_amount / price_risk
        return position_size
    
    def open_position(self, signal, current_price, atr):
        """Open a new position"""
        self.entry_price = current_price
        self.entry_time = datetime.now().isoformat()
        self.trade_type = signal
        
        if signal == 'BUY':
            self.stop_loss = self.entry_price - (1.5 * atr)
            self.take_profit = self.entry_price + (3.0 * atr)
        else:
            self.stop_loss = self.entry_price + (1.5 * atr)
            self.take_profit = self.entry_price - (3.0 * atr)
        
        self.position_size = self.calculate_position_size(self.entry_price, self.stop_loss)
        
        self.position = {
            'trade_type': self.trade_type,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_size': self.position_size,
            'entry_time': self.entry_time
        }
        
        print("\n" + "="*80)
        print(f"🟢 OPENING {signal} POSITION")
        print("="*80)
        print(f"Time: {self.entry_time}")
        print(f"Entry Price: ${self.entry_price:,.2f}")
        print(f"Stop Loss: ${self.stop_loss:,.2f}")
        print(f"Take Profit: ${self.take_profit:,.2f}")
        print(f"Position Size: {self.position_size:.4f} units")
        print(f"Risk: ${abs(self.entry_price - self.stop_loss) * self.position_size:,.2f}")
        print("="*80)
    
    def check_exit_conditions(self, current_price):
        """Check if position should be closed"""
        if self.trade_type == 'BUY':
            if current_price >= self.take_profit:
                return 'TAKE_PROFIT'
            elif current_price <= self.stop_loss:
                return 'STOP_LOSS'
        
        elif self.trade_type == 'SELL':
            if current_price <= self.take_profit:
                return 'TAKE_PROFIT'
            elif current_price >= self.stop_loss:
                return 'STOP_LOSS'
        
        return None
    
    def close_position(self, exit_price, exit_reason):
        """Close the current position"""
        exit_time = datetime.now().isoformat()
        
        if self.trade_type == 'BUY':
            pnl = (exit_price - self.entry_price) * self.position_size
        else:
            pnl = (self.entry_price - exit_price) * self.position_size
        
        pnl_pct = (pnl / self.capital) * 100
        result = 'WIN' if pnl > 0 else 'LOSS'
        
        capital_before = self.capital
        self.capital += pnl
        
        trade = {
            'entry_time': self.entry_time,
            'exit_time': exit_time,
            'trade_type': self.trade_type,
            'entry_price': self.entry_price,
            'exit_price': exit_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_size': self.position_size,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'result': result,
            'capital_before': capital_before,
            'capital_after': self.capital,
            'exit_reason': exit_reason
        }
        
        self.trades.append(trade)
        
        print("\n" + "="*80)
        print(f"🔴 CLOSING {self.trade_type} POSITION - {exit_reason}")
        print("="*80)
        print(f"Exit Time: {exit_time}")
        print(f"Entry Price: ${self.entry_price:,.2f}")
        print(f"Exit Price: ${exit_price:,.2f}")
        print(f"Result: {result}")
        print(f"P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
        print(f"Capital: ${capital_before:,.2f} → ${self.capital:,.2f}")
        print("="*80)
        
        self.position = None
        self.trade_type = None
        self.position_size = 0
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.entry_time = None
    
    def log_equity(self, current_price):
        """Log current equity"""
        equity_entry = {
            'timestamp': datetime.now().isoformat(),
            'equity': self.capital,
            'position_status': self.trade_type if self.trade_type else 'FLAT',
            'current_price': current_price
        }
        self.equity_history.append(equity_entry)
    
    def run_once(self):
        """Run one check cycle"""
        print(f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Checking market...")
        
        df = self.fetch_live_data()
        
        if df is None or len(df) < 50:
            print("❌ Insufficient data")
            return
        
        df = self.calculate_indicators(df)
        current_price = df['Close'].iloc[-1]
        current_atr = df['ATR'].iloc[-1]
        
        print(f"💰 Current Gold Price: ${current_price:,.2f}")
        print(f"💼 Current Capital: ${self.capital:,.2f}")
        
        self.log_equity(current_price)
        
        if self.position is not None:
            if self.trade_type == 'BUY':
                unrealized_pnl = (current_price - self.entry_price) * self.position_size
            else:
                unrealized_pnl = (self.entry_price - current_price) * self.position_size
            
            unrealized_pct = (unrealized_pnl / self.capital) * 100
            print(f"📊 Open Position: {self.trade_type}")
            print(f"📊 Unrealized P&L: ${unrealized_pnl:,.2f} ({unrealized_pct:+.2f}%)")
            
            exit_reason = self.check_exit_conditions(current_price)
            
            if exit_reason:
                self.close_position(current_price, exit_reason)
        
        if self.position is None:
            signal = self.check_entry_signal(df)
            
            if signal:
                self.open_position(signal, current_price, current_atr)
            else:
                print("⚪ No entry signal - waiting...")
        
        self.save_data()
        
        # Print summary
        if len(self.trades) > 0:
            wins = sum(1 for t in self.trades if t['result'] == 'WIN')
            losses = len(self.trades) - wins
            win_rate = (wins / len(self.trades)) * 100 if len(self.trades) > 0 else 0
            total_pnl = sum(t['pnl'] for t in self.trades)
            
            print(f"\n📊 PERFORMANCE SUMMARY:")
            print(f"Total Trades: {len(self.trades)} | Wins: {wins} | Losses: {losses}")
            print(f"Win Rate: {win_rate:.1f}% | Total P&L: ${total_pnl:,.2f}")


if __name__ == "__main__":
    bot = GoldTradingBot(initial_capital=10000, risk_per_trade=2.0)
    bot.run_once()

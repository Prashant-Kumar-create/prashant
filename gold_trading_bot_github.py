"""
Gold Trading Bot - IMPROVED VERSION
GitHub Actions Version with:
1. Dynamic Target/SL based on volatility (not fixed 3X)
2. Re-entry prevention after profitable exits
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
        
        # NEW: Re-entry prevention settings
        self.last_profit_exits = []
        self.min_price_move_pct = 1.5  # Require 1.5% price movement from last profit exit
        self.min_time_hours = 3  # Require 3 hours since last profit exit
        
        # Load existing data or initialize
        self.load_data()
        
        print("="*80)
        print("🤖 GOLD TRADING BOT - IMPROVED VERSION")
        print("="*80)
        print(f"Current Capital: ${self.capital:,.2f}")
        print(f"Risk Per Trade: {self.risk_per_trade}%")
        print(f"Total Trades: {len(self.trades)}")
        print(f"✨ NEW: Dynamic Targets (Volatility-Based)")
        print(f"✨ NEW: Re-entry Filter (Min {self.min_price_move_pct}% move, {self.min_time_hours}h wait)")
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
                self.last_profit_exits = data.get('last_profit_exits', [])
                
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
                    
                print(f"✅ Loaded existing data: {len(self.trades)} trades, {len(self.last_profit_exits)} recent profit exits")
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
            'last_profit_exits': self.last_profit_exits,
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
    
    def calculate_adaptive_targets(self, signal, entry_price, atr, df):
        """
        NEW: Calculate dynamic targets based on volatility
        High volatility = smaller targets (2X)
        Low volatility = larger targets (3.5X)
        """
        # Calculate ATR as percentage of price
        atr_pct = (atr / entry_price) * 100
        
        # Adjust multipliers based on volatility
        if atr_pct > 2.0:  # High volatility (choppy market)
            stop_multiplier = 1.2
            target_multiplier = 2.0
            volatility_state = "HIGH"
        elif atr_pct > 1.0:  # Medium volatility
            stop_multiplier = 1.5
            target_multiplier = 2.5
            volatility_state = "MEDIUM"
        else:  # Low volatility (trending market)
            stop_multiplier = 1.8
            target_multiplier = 3.5
            volatility_state = "LOW"
        
        print(f"📊 Volatility: {volatility_state} (ATR: {atr_pct:.2f}% of price)")
        print(f"📊 Using Stop: {stop_multiplier}x ATR, Target: {target_multiplier}x ATR")
        
        if signal == 'BUY':
            stop_loss = entry_price - (stop_multiplier * atr)
            take_profit = entry_price + (target_multiplier * atr)
        else:  # SELL
            stop_loss = entry_price + (stop_multiplier * atr)
            take_profit = entry_price - (target_multiplier * atr)
        
        return stop_loss, take_profit
    
    def check_entry_signal(self, df):
        """
        Check for entry signals with NEW re-entry prevention filter
        Blocks same-direction trades if:
        - Price hasn't moved enough from last profit exit (1.5%)
        - Not enough time has passed since last profit exit (3 hours)
        """
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
        
        # Determine raw signal
        signal = None
        if (close > ema_9 > ema_21 > ema_50 and rsi > 50 and rsi < 80 and adx > 25):
            signal = 'BUY'
        elif (close < ema_9 < ema_21 < ema_50 and rsi < 50 and rsi > 20 and adx > 25):
            signal = 'SELL'
        
        # NEW: Re-entry prevention filter
        if signal and self.last_profit_exits:
            for recent_exit in self.last_profit_exits:
                # Only check exits in the same direction
                if signal == recent_exit['direction']:
                    # Calculate price movement from exit
                    price_diff_pct = abs(close - recent_exit['price']) / recent_exit['price'] * 100
                    
                    # Calculate time since exit
                    exit_time = datetime.fromisoformat(recent_exit['time'])
                    time_diff = datetime.now() - exit_time
                    hours_passed = time_diff.total_seconds() / 3600
                    
                    # Block if too close in price OR too soon in time
                    if price_diff_pct < self.min_price_move_pct:
                        print(f"\n⚠️ BLOCKING {signal} ENTRY - Insufficient price movement")
                        print(f"   Last {signal} profit exit: ${recent_exit['price']:.2f} ({hours_passed:.1f}h ago)")
                        print(f"   Current price: ${close:.2f}")
                        print(f"   Price moved: {price_diff_pct:.2f}% (need {self.min_price_move_pct}%)")
                        return None
                    
                    if hours_passed < self.min_time_hours:
                        print(f"\n⚠️ BLOCKING {signal} ENTRY - Too soon after last exit")
                        print(f"   Last {signal} profit exit: ${recent_exit['price']:.2f}")
                        print(f"   Time since exit: {hours_passed:.1f}h (need {self.min_time_hours}h)")
                        print(f"   Price movement: {price_diff_pct:.2f}% ✓")
                        return None
        
        return signal
    
    def calculate_position_size(self, entry_price, stop_loss):
        """Calculate position size based on 2% risk"""
        risk_amount = self.capital * (self.risk_per_trade / 100)
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return 0
        
        position_size = risk_amount / price_risk
        return position_size
    
    def open_position(self, signal, current_price, atr, df):
        """Open a new position with adaptive targets"""
        self.entry_price = current_price
        self.entry_time = datetime.now().isoformat()
        self.trade_type = signal
        
        # NEW: Use adaptive targets instead of fixed 3X
        self.stop_loss, self.take_profit = self.calculate_adaptive_targets(
            signal, self.entry_price, atr, df
        )
        
        self.position_size = self.calculate_position_size(self.entry_price, self.stop_loss)
        
        self.position = {
            'trade_type': self.trade_type,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_size': self.position_size,
            'entry_time': self.entry_time
        }
        
        # Calculate R:R ratio
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        print("\n" + "="*80)
        print(f"🟢 OPENING {signal} POSITION")
        print("="*80)
        print(f"Time: {self.entry_time}")
        print(f"Entry Price: ${self.entry_price:,.2f}")
        print(f"Stop Loss: ${self.stop_loss:,.2f}")
        print(f"Take Profit: ${self.take_profit:,.2f}")
        print(f"Position Size: {self.position_size:.4f} units")
        print(f"Risk Amount: ${risk * self.position_size:,.2f}")
        print(f"R:R Ratio: 1:{rr_ratio:.2f}")
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
        """Close the current position and update profit exit tracking"""
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
        
        # NEW: Track profitable exits for re-entry prevention
        if result == 'WIN':
            self.last_profit_exits.append({
                'price': exit_price,
                'direction': self.trade_type,
                'time': exit_time
            })
            
            # Keep only last 3 profitable exits (24 hours window)
            current_time = datetime.now()
            self.last_profit_exits = [
                exit_data for exit_data in self.last_profit_exits
                if (current_time - datetime.fromisoformat(exit_data['time'])).total_seconds() < 86400
            ]
            
            if len(self.last_profit_exits) > 3:
                self.last_profit_exits = self.last_profit_exits[-3:]
        
        print("\n" + "="*80)
        print(f"🔴 CLOSING {self.trade_type} POSITION - {exit_reason}")
        print("="*80)
        print(f"Exit Time: {exit_time}")
        print(f"Entry Price: ${self.entry_price:,.2f}")
        print(f"Exit Price: ${exit_price:,.2f}")
        print(f"Result: {result}")
        print(f"P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
        print(f"Capital: ${capital_before:,.2f} → ${self.capital:,.2f}")
        if result == 'WIN':
            print(f"✨ Profit exit tracked - Future {self.trade_type} entries filtered near ${exit_price:.2f}")
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
        
        # Check if we have an open position
        if self.position is not None:
            if self.trade_type == 'BUY':
                unrealized_pnl = (current_price - self.entry_price) * self.position_size
            else:
                unrealized_pnl = (self.entry_price - current_price) * self.position_size
            
            unrealized_pct = (unrealized_pnl / self.capital) * 100
            print(f"📊 Open Position: {self.trade_type}")
            print(f"📊 Entry: ${self.entry_price:.2f} | Current: ${current_price:.2f}")
            print(f"📊 Unrealized P&L: ${unrealized_pnl:,.2f} ({unrealized_pct:+.2f}%)")
            
            exit_reason = self.check_exit_conditions(current_price)
            
            if exit_reason:
                self.close_position(current_price, exit_reason)
        
        # Check for new entry if no position
        if self.position is None:
            signal = self.check_entry_signal(df)
            
            if signal:
                self.open_position(signal, current_price, current_atr, df)
            else:
                print("⚪ No entry signal - waiting...")
        
        self.save_data()
        
        # Print summary
        if len(self.trades) > 0:
            wins = sum(1 for t in self.trades if t['result'] == 'WIN')
            losses = len(self.trades) - wins
            win_rate = (wins / len(self.trades)) * 100 if len(self.trades) > 0 else 0
            total_pnl = sum(t['pnl'] for t in self.trades)
            total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
            
            print(f"\n📊 PERFORMANCE SUMMARY:")
            print(f"Total Trades: {len(self.trades)} | Wins: {wins} | Losses: {losses}")
            print(f"Win Rate: {win_rate:.1f}% | Total P&L: ${total_pnl:,.2f}")
            print(f"Total Return: {total_return:+.2f}% | Capital: ${self.capital:,.2f}")
            
            if self.last_profit_exits:
                print(f"\n🛡️ Active Filters: {len(self.last_profit_exits)} recent profit exits being monitored")


if __name__ == "__main__":
    bot = GoldTradingBot(initial_capital=10000, risk_per_trade=2.0)
    bot.run_once()

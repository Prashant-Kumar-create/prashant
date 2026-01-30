"""
View Trades from JSON file (GitHub Actions version)
"""

import json
import os
from datetime import datetime

def view_trades(json_file='trading_data.json'):
    """View all trades from JSON file"""
    if not os.path.exists(json_file):
        print("❌ trading_data.json not found")
        print("Make sure you've cloned the repository and the bot has run at least once.")
        return
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    capital = data.get('capital', 0)
    trades = data.get('trades', [])
    position = data.get('position', None)
    
    print("\n" + "="*80)
    print("🤖 GOLD TRADING BOT - PERFORMANCE")
    print("="*80)
    
    print(f"\n💼 Current Capital: ${capital:,.2f}")
    
    if position:
        print(f"\n🟢 OPEN POSITION:")
        print(f"   Type: {position['trade_type']}")
        print(f"   Entry: ${position['entry_price']:,.2f}")
        print(f"   Stop Loss: ${position['stop_loss']:,.2f}")
        print(f"   Take Profit: ${position['take_profit']:,.2f}")
        print(f"   Opened: {position['entry_time']}")
    else:
        print(f"\n⚪ Position: FLAT (No open position)")
    
    if len(trades) == 0:
        print("\n📊 No trades yet - waiting for signals...")
        return
    
    # Calculate statistics
    total_trades = len(trades)
    wins = sum(1 for t in trades if t['result'] == 'WIN')
    losses = total_trades - wins
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    total_pnl = sum(t['pnl'] for t in trades)
    avg_win = sum(t['pnl'] for t in trades if t['result'] == 'WIN') / wins if wins > 0 else 0
    avg_loss = sum(t['pnl'] for t in trades if t['result'] == 'LOSS') / losses if losses > 0 else 0
    
    print(f"\n" + "="*80)
    print("📈 PERFORMANCE SUMMARY")
    print("="*80)
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {wins}")
    print(f"Losing Trades: {losses}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Total P&L: ${total_pnl:,.2f}")
    print(f"Average Win: ${avg_win:,.2f}")
    print(f"Average Loss: ${avg_loss:,.2f}")
    
    if wins > 0 and losses > 0:
        profit_factor = abs(avg_win * wins / (avg_loss * losses))
        print(f"Profit Factor: {profit_factor:.2f}")
    
    print("\n" + "="*80)
    print("📋 TRADE HISTORY")
    print("="*80)
    
    # Show trades
    for i, trade in enumerate(reversed(trades), 1):
        entry_time = trade['entry_time'][:19] if len(trade['entry_time']) > 19 else trade['entry_time']
        exit_time = trade['exit_time'][:19] if len(trade['exit_time']) > 19 else trade['exit_time']
        
        print(f"\nTrade #{i}:")
        print(f"  {trade['trade_type']:4} | Entry: ${trade['entry_price']:,.2f} → Exit: ${trade['exit_price']:,.2f}")
        print(f"  {trade['result']:4} | P&L: ${trade['pnl']:,.2f} ({trade['pnl_pct']:+.2f}%)")
        print(f"  Exit Reason: {trade['exit_reason']}")
        print(f"  Time: {entry_time} to {exit_time}")


def view_equity_curve(json_file='trading_data.json'):
    """View equity curve"""
    if not os.path.exists(json_file):
        print("❌ trading_data.json not found")
        return
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    equity_history = data.get('equity_history', [])
    
    if len(equity_history) == 0:
        print("No equity data yet")
        return
    
    print("\n" + "="*80)
    print("📈 EQUITY CURVE")
    print("="*80)
    
    start_equity = equity_history[0]['equity']
    current_equity = equity_history[-1]['equity']
    
    print(f"\nStarting Capital: ${start_equity:,.2f}")
    print(f"Current Capital: ${current_equity:,.2f}")
    print(f"Total Change: ${current_equity - start_equity:,.2f}")
    print(f"Total Return: {((current_equity / start_equity) - 1) * 100:+.2f}%")
    
    # Calculate peak and drawdown
    equities = [e['equity'] for e in equity_history]
    peak = max(equities)
    current = equities[-1]
    drawdown = ((current - peak) / peak * 100) if peak > 0 else 0
    
    print(f"Peak Capital: ${peak:,.2f}")
    print(f"Current Drawdown: {drawdown:.2f}%")
    
    print("\n" + "="*80)
    print("📊 RECENT EQUITY SNAPSHOTS (Last 20)")
    print("="*80)
    
    for entry in equity_history[-20:]:
        timestamp = entry['timestamp'][:19] if len(entry['timestamp']) > 19 else entry['timestamp']
        print(f"{timestamp} | ${entry['equity']:,.2f} | {entry['position_status']:4} | Gold: ${entry['current_price']:,.2f}")


def export_to_csv(json_file='trading_data.json', output_file='trades.csv'):
    """Export trades to CSV"""
    if not os.path.exists(json_file):
        print("❌ trading_data.json not found")
        return
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    trades = data.get('trades', [])
    
    if len(trades) == 0:
        print("No trades to export")
        return
    
    # Write CSV
    with open(output_file, 'w') as f:
        # Header
        f.write("entry_time,exit_time,trade_type,entry_price,exit_price,stop_loss,take_profit,position_size,pnl,pnl_pct,result,capital_before,capital_after,exit_reason\n")
        
        # Data
        for trade in trades:
            f.write(f"{trade['entry_time']},{trade['exit_time']},{trade['trade_type']},{trade['entry_price']},{trade['exit_price']},{trade['stop_loss']},{trade['take_profit']},{trade['position_size']},{trade['pnl']},{trade['pnl_pct']},{trade['result']},{trade['capital_before']},{trade['capital_after']},{trade['exit_reason']}\n")
    
    print(f"✅ Exported {len(trades)} trades to {output_file}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("📊 TRADING BOT - JSON VIEWER")
    print("="*80)
    print("\nOptions:")
    print("1. View all trades")
    print("2. View equity curve")
    print("3. Export trades to CSV")
    print("4. Show everything")
    print("="*80)
    
    choice = input("\nSelect option (1-4): ")
    
    if choice == '1':
        view_trades()
    
    elif choice == '2':
        view_equity_curve()
    
    elif choice == '3':
        export_to_csv()
    
    elif choice == '4':
        view_trades()
        view_equity_curve()
    
    else:
        print("Invalid option")

import os
import pandas as pd
import numpy as np

def run_mahabank_backtest_with_custom_exits(csv_path="data/MAHABANK.csv", require_ema_dip=False):
    if not os.path.exists(csv_path):
        print(f"Error: Could not locate file at path: '{csv_path}'")
        print("Please check that your 'data' subfolder exists and contains 'MAHABANK.csv'.")
        return
        
    # Load and clean structural dataset layout
    df = pd.read_csv(csv_path)
    df.columns = [str(col).strip().capitalize() for col in df.columns]
    df = df.dropna(subset=['Close']).reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate indicators
    df['EMA_220'] = df['Close'].ewm(span=220, adjust=False).mean()
    df['SMA_150'] = df['Close'].rolling(window=150).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Low_52W'] = df['Low'].rolling(window=252).min()
    
    # Pre-calculate rolling historical ATH window up to T-1
    historical_ath = []
    historical_ath_idx = []
    
    running_max_high = -1.0
    running_max_idx = -1
    
    for idx in range(len(df)):
        historical_ath.append(running_max_high)
        historical_ath_idx.append(running_max_idx)
        
        if df['High'].iloc[idx] > running_max_high:
            running_max_high = float(df['High'].iloc[idx])
            running_max_idx = idx
            
    df['Hist_ATH'] = historical_ath
    df['Hist_ATH_Idx'] = historical_ath_idx
    
    # Backtest Processing Engine Loop
    trades = []
    in_position = False
    entry_price = 0.0
    entry_date = None
    
    for i in range(252, len(df)):
        close = float(df['Close'].iloc[i])
        low = float(df['Low'].iloc[i])
        sma50 = float(df['SMA_50'].iloc[i])
        sma150 = float(df['SMA_150'].iloc[i])
        ema220 = float(df['EMA_220'].iloc[i])
        low52w = float(df['Low_52W'].iloc[i])
        hist_ath = float(df['Hist_ATH'].iloc[i])
        ath_idx = int(df['Hist_ATH_Idx'].iloc[i])
        date = df['Date'].iloc[i]
        
        if not in_position:
            # Evaluate Entry Conditions
            cond1 = sma150 > ema220
            cond2 = close > sma50
            cond3 = sma50 > sma150
            cond4 = close > (1.25 * low52w)
            cond5 = close > hist_ath
            
            cond6 = True
            if require_ema_dip and ath_idx != -1:
                if ath_idx + 1 <= i - 1:
                    interim_df = df.iloc[ath_idx + 1 : i]
                    cond6 = (interim_df['Close'] < interim_df['EMA_220']).any()
                else:
                    cond6 = False
            
            if cond1 and cond2 and cond3 and cond4 and cond5 and cond6:
                in_position = True
                entry_price = close
                entry_date = date
        else:
            # Evaluate Exit Constraints 
            # 1. Closed below 220 EMA
            exit_cond_ema = close < ema220
            
            # 2. Intraday low hit or dropped below 15% from execution entry price
            stop_loss_level = entry_price * 0.85
            exit_cond_stop = low <= stop_loss_level
            
            is_last_row = (i == len(df) - 1)
            
            if exit_cond_ema or exit_cond_stop or is_last_row:
                in_position = False
                
                if exit_cond_stop and not exit_cond_ema and not is_last_row:
                    exit_price = stop_loss_level
                    exit_reason = "15% Stop Loss Hit"
                elif exit_cond_ema and not exit_cond_stop:
                    exit_price = close
                    exit_reason = "Closed Below 220 EMA"
                elif exit_cond_ema and exit_cond_stop:
                    # If both thresholds were breached on the same day, prioritize stop loss level
                    exit_price = stop_loss_level
                    exit_reason = "15% Stop Loss Hit (Same Day)"
                else:
                    exit_price = close
                    exit_reason = "End of Data (Position Active)"
                    
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                
                trades.append({
                    'Entry Date': entry_date.strftime('%Y-%m-%d'),
                    'Entry Price': round(entry_price, 2),
                    'Exit Date': date.strftime('%Y-%m-%d'),
                    'Exit Price': round(exit_price, 2),
                    'PnL %': round(pnl_pct, 2),
                    'Exit Reason': exit_reason
                })
                
    # Output Strategy Statistics 
    trades_df = pd.DataFrame(trades)
    print("\n" + "="*90)
    print(f"MAHABANK STRATEGY BACKTEST (Require 220 EMA Dip After ATH: {require_ema_dip})")
    print("="*90)
    if trades_df.empty:
        print("No trades matched the criteria matrix constraints in this historical window.")
    else:
        print(trades_df.to_string(index=False))
        wins = (trades_df['PnL %'] > 0).sum()
        total = len(trades_df)
        win_rate = (wins / total) * 100 if total > 0 else 0
        avg_pnl = trades_df['PnL %'].mean()
        compounded = ((trades_df['PnL %'] / 100 + 1).prod() - 1) * 100
        print("-"*90)
        print(f"Total Completed Trades : {total}")
        print(f"Strategy Win Rate      : {win_rate:.2f}%")
        print(f"Average Return / Trade : {avg_pnl:.2f}%")
        print(f"Compounded Net Return  : {compounded:.2f}%")
    print("="*90 + "\n")

if __name__ == "__main__":
    # Test execution across both operational parameter tracking variants
    run_mahabank_backtest_with_custom_exits("data/MAHABANK.csv", require_ema_dip=False)
    run_mahabank_backtest_with_custom_exits("data/MAHABANK.csv", require_ema_dip=True)
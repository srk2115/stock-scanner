import os
import io
import sys
import pandas as pd
from flask import Flask, render_template, request, jsonify
from concurrent.futures import ThreadPoolExecutor
from const import INDEX_SYMBOLS
import yfinance as yf
import numpy as np

app = Flask(__name__)

DATA_FOLDER_AUTO_ADJUST = "data-AdjClose_True"
DATA_FOLDER_NO_ADJUST = "data"
DATA_FOLDER = "data-AdjClose_True"
# DATA_FOLDER = "data"

def clean_and_format_symbols(raw_input_string):
    if not raw_input_string:
        return []
    cleaned_text = raw_input_string.replace(',', ' ').replace(';', ' ').replace('\n', ' ')
    formatted_symbols = []
    for token in cleaned_text.split():
        token = token.strip().upper()
        if not token:
            continue
        formatted_symbols.append(token)
    return formatted_symbols

def find_csv_file(symbol, auto_adjust):
    base_symbol = symbol.replace('.NS', '')
    possible_names = [
        f"{base_symbol}.csv", f"{base_symbol}.NS.csv",
        f"{symbol}.csv", f"{symbol}.NS.csv"
    ]

    if(auto_adjust):
        data_folder = DATA_FOLDER_AUTO_ADJUST
    else:
        data_folder = DATA_FOLDER_NO_ADJUST

    if not os.path.exists(data_folder):
        return None
    for name in possible_names:
        for actual_file in os.listdir(data_folder):
            if actual_file.upper() == name.upper():
                return os.path.join(data_folder, actual_file)
    return None

def process_single_stock(symbol, require_ema_dip, auto_adjust):
    """
    Worker function executed inside separate concurrent threads 
    to scan single stock metrics without locking the main thread.
    """
    is_match = False
    current_close = "-"
    val_sma50 = "-"
    val_sma150 = "-"
    val_ema220 = "-"
    val_low52wk = "-"
    val_ath = "-"
    status_str = "FAILED"
    
    csv_path = find_csv_file(symbol, auto_adjust)
    if not csv_path:
        status_str = "FILE NOT FOUND"
    else:
        try:
            df = pd.read_csv(csv_path)
            # df.columns = [str(col).strip().capitalize() for col in df.columns]
            df.columns = [str(col).strip() for col in df.columns]
            
            if 'Close' not in df.columns or 'High' not in df.columns or 'Low' not in df.columns:
                status_str = "INVALID CSV FORMAT"
            else:
                # df = df.dropna(subset=['Close']).reset_index(drop=True)
                # df['Date'] = pd.to_datetime(df['Date'])

                # print(f"Fetching latest daily data for {symbol} with auto_adjust={auto_adjust}...")
                # df1 = yf.download(
                #         tickers=f"{symbol}.NS",
                #         period="1d",
                #         interval="1d",
                #         auto_adjust=auto_adjust,  # Perfect High/Close ratios 
                #         threads=False,     # Linear thread-safe tracking loops
                #         timeout=15,
                #         progress=False
                #     )
                # df1.columns = df1.columns.get_level_values(0)
                # # df1.columns = [str(col).strip().capitalize() for col in df.columns]
                # df1.columns = [str(col).strip() for col in df1.columns]
                # df['Date'] = pd.to_datetime(df['Date'])
                # df1.reset_index(inplace=True)
                # print(df1.tail())

                # print(f"Stacking historical data with latest daily data for {symbol}...")
                # cdf = pd.concat([df, df1], ignore_index=True)
                # cdf.tail()
                # print(f"Removing duplicate dates, keeping the latest entry for {symbol}...")
                # df = cdf.drop_duplicates(subset=['Date'], keep='last').reset_index(drop=True)
                # print(df.tail())

                df = df.dropna(subset=['Close']).reset_index(drop=True)
                df['Date'] = pd.to_datetime(df['Date'])

                print(f"Fetching latest daily data for {symbol} with auto_adjust={auto_adjust}...")
                df1 = yf.download(
                        tickers=f"{symbol}.NS",
                        period="1d",
                        interval="1d",
                        auto_adjust=auto_adjust, 
                        threads=False,     
                        timeout=15,
                        progress=False
                    )

                # --- CRITICAL FIX START: Handle MultiIndex Columns Safely ---
                if isinstance(df1.columns, pd.MultiIndex):
                    # If a batch multi-index leaks through, extract ONLY the current symbol's data slice
                    ticker_str = f"{symbol}.NS"
                    if ticker_str in df1.columns.get_level_values(1):
                        df1 = df1.xs(ticker_str, axis=1, level=1)
                    else:
                        # Fallback to level 0 if layout varies
                        df1.columns = df1.columns.get_level_values(0)
                else:
                    # If it's already flat, just normalize index layout
                    df1.columns = [str(col).strip() for col in df1.columns]

                # Ensure we have uniquely named columns before moving forward
                df1 = df1.loc[:, ~df1.columns.duplicated()]
                # --- CRITICAL FIX END ---

                df1.reset_index(inplace=True)

                # Double check Date normalization mapping rules match
                if 'Date' in df1.columns:
                    df1['Date'] = pd.to_datetime(df1['Date'])

                print(f"Stacking historical data with latest daily data for {symbol}...")
                cdf = pd.concat([df, df1], ignore_index=True)

                print(f"Removing duplicate dates, keeping the latest entry for {symbol}...")
                df = cdf.drop_duplicates(subset=['Date'], keep='last').reset_index(drop=True)
                print(df.tail())

                if len(df) < 252:
                    status_str = "INSUFFICIENT DATA"
                else:
                    ema_220_series = df['Close'].ewm(span=220, adjust=False).mean()
                    sma_150_series = df['Close'].rolling(window=150).mean()
                    sma_50_series = df['Close'].rolling(window=50).mean()
                    
                    low_52wk = float(df['Low'].iloc[-252:].min())
                    
                    history_df = df.iloc[:-1]
                    max_high_val = history_df['High'].max()
                    ath_index = history_df['High'].idxmax()
                    
                    if require_ema_dip:
                        interim_df = history_df.iloc[ath_index + 1:]
                        if len(interim_df) > 0:
                            dipped_below_220_ema = (interim_df['Close'] < ema_220_series.loc[interim_df.index]).any()
                        else:
                            dipped_below_220_ema = False
                    else:
                        dipped_below_220_ema = True
                    
                    current_close = round(float(df['Close'].iloc[-1]), 2)
                    val_ema220 = round(float(ema_220_series.iloc[-1]), 2)
                    val_sma150 = round(float(sma_150_series.iloc[-1]), 2)
                    val_sma50 = round(float(sma_50_series.iloc[-1]), 2)
                    val_low52wk = round(low_52wk, 2)
                    val_ath = round(float(max_high_val), 2)
                    
                    cond1 = val_sma150 > val_ema220
                    cond2 = current_close > val_sma50
                    cond3 = val_sma50 > val_sma150
                    cond4 = current_close > (1.25 * val_low52wk)
                    cond5 = current_close > max_high_val
                    cond6 = bool(dipped_below_220_ema)
                    
                    if cond1 and cond2 and cond3 and cond4 and cond5 and cond6:
                        status_str = "MATCH"
                        is_match = True
                    else:
                        status_str = "NO MATCH"
                        
        except Exception as e:
            status_str = "ERR: READ ERROR"
            print(f"[ERROR] Failed parsing data for {symbol}: {str(e)}")
            
    return {
        'ticker': symbol,
        'close': current_close,
        'sma_50': val_sma50,
        'sma_150': val_sma150,
        'ema_220': val_ema220,
        'low_52wk': val_low52wk,
        'historical_ath': val_ath,
        'status': status_str,
        'is_match': is_match
    }

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/get_indexes', methods=['GET'])
def get_indexes():
    return jsonify({'success': True, 'indexes': list(INDEX_SYMBOLS.keys())})

@app.route('/scan', methods=['POST'])
def scan_stocks():
    data = request.get_json() or {}
    input_mode = data.get('mode', 'index')
    require_ema_dip = data.get('require_ema_dip', True)
    print(f"Backtest request received for with require_ema_dip={require_ema_dip}, 220 EMA Dip checkbox state={require_ema_dip}")

    auto_adjust = data.get('auto_adjust', True)
    print(f"Backtest request received for with auto_adjust={auto_adjust}, auto_adjust checkbox state={auto_adjust}")


    print(f"Scanning stocks with parameters: require_ema_dip={require_ema_dip}, auto_adjust={auto_adjust}")
    if input_mode == 'index':
        selected_index = data.get('index_name', '').upper()
        symbols = INDEX_SYMBOLS.get(selected_index, [])
        if not symbols:
            return jsonify({'success': False, 'error': f'Index {selected_index} not found or empty.'})
    else:
        raw_input = data.get('symbols', '')
        symbols = clean_and_format_symbols(raw_input)

    if not symbols:
        return jsonify({'success': False, 'error': 'No valid symbols provided.'})
        
    results = []
    # Use max_workers=10 to rapidly speed up directory operations concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_single_stock, symbol, require_ema_dip, auto_adjust) for symbol in symbols]
        for future in futures:
            results.append(future.result())

    return jsonify({'success': True, 'results': results})

def run_strategy_backtest(df, require_ema_dip=False):
    """
    Core strategy evaluation logic. Tracks actual closed positions 
    to calculate live summary performance metrics.
    """
    trades = []
    in_position = False
    entry_price = 0.0
    
    running_ath = 0.0
    ath_idx = -1
    
    # Evaluate data rows starting from lookback threshold
    for i in range(252, len(df)):
        row = df.iloc[i]
        
        # Track running historical ATH barrier state manually for strategy matching
        # inside the loop to avoid forward-looking bias
        past_window = df.iloc[252:i]
        if not past_window.empty:
            running_ath = past_window['High'].max()
            # Find the row index where that specific max High occurred
            ath_idx = past_window['High'].idxmax()
        else:
            running_ath = 0.0
            ath_idx = -1

        if not in_position:
            # Check entry criteria matches
            cond1 = row['SMA_150'] > row['EMA_220']
            cond2 = row['Close'] > row['SMA_50']
            cond3 = row['SMA_50'] > row['SMA_150']
            cond4 = row['Close'] > (1.25 * row['Low_52W'])
            cond5 = row['Close'] > running_ath
            
            cond6 = True
            if require_ema_dip and ath_idx != -1:
                if ath_idx + 1 <= i - 1:
                    interim_segment = df.iloc[ath_idx + 1 : i]
                    cond6 = (interim_segment['Close'] < interim_segment['EMA_220']).any()
                else:
                    cond6 = False
            
            if cond1 and cond2 and cond3 and cond4 and cond5 and cond6:
                in_position = True
                entry_price = row['Close']
        else:
            # Simple Exit Rule: Close crosses below SMA 50 or final day of dataset
            if row['Close'] < row['SMA_50'] or i == len(df) - 1:
                in_position = False
                pnl = (row['Close'] - entry_price) / entry_price
                trades.append(pnl)
                
    # Compute performance statistics arrays dynamically
    total_trades = len(trades)
    if total_trades > 0:
        winning_trades = sum(1 for pnl in trades if pnl > 0)
        win_rate = round((winning_trades / total_trades) * 100, 1)
        
        # Calculate compounded system returns
        compounded_return = round((np.prod([1 + pnl for pnl in trades]) - 1) * 100, 1)
    else:
        win_rate = 0.0
        compounded_return = 0.0
        
    return total_trades, win_rate, compounded_return

@app.route('/backtest', methods=['GET'])
def web_backtest():
    target_symbol = request.args.get('symbol', 'MAHABANK').strip().upper()
    
    # Read the checkbox/dropdown state from URL parameter (default to 'false' if missing)
    require_ema_dip_str = request.args.get('require_ema_dip', 'false').lower()
    require_ema_dip = (require_ema_dip_str == 'true')
    print(f"Backtest request received for {target_symbol} with require_ema_dip={require_ema_dip}, require_ema_dip_str={require_ema_dip_str}, 220 EMA Dip checkbox state={require_ema_dip}")
    
    # Read the auto_adjust state from URL parameter (default to 'true' if missing)
    auto_adjust_str = request.args.get('auto_adjust', 'true').lower()
    auto_adjust = (auto_adjust_str == 'true')
    print(f"Backtest request received for {target_symbol} with auto_adjust={auto_adjust}, auto_adjust_str={auto_adjust_str}, auto_adjust checkbox state={auto_adjust}")


    csv_path = find_csv_file(target_symbol, auto_adjust)
    print(csv_path)
    if not csv_path:
        return f"""
        <body style="background-color: #1a202c; color: #edf2f7; font-family: sans-serif; padding: 30px;">
            <h3 style="color: #e53e3e;">Error: CSV target history file for "{target_symbol}" was not found.</h3>
            <p>Make sure the file exists as <b>data/{target_symbol}.csv</b> or <b>data/{target_symbol}.NS.csv</b></p>
            <br><a href="/" style="color: #63b3ed; text-decoration: none;">&larr; Return to Core Scanner</a>
        </body>
        """, 404

    # Safe default fallbacks if data is missing
    total_trades, win_rate, net_return = 0, 0.0, 0.0

    try:
        df = pd.read_csv(csv_path)
        # df.columns = [str(col).strip().capitalize() for col in df.columns]
        df.columns = [str(col).strip() for col in df.columns]
        df = df.dropna(subset=['Close']).reset_index(drop=True)
        df['Date'] = pd.to_datetime(df['Date'])

        print(df.tail())

        # auto_adjust = True if data_folder == DATA_FOLDER_AUTO_ADJUST else False
        print(f"Fetching latest daily data for {target_symbol} with auto_adjust={auto_adjust}...")
        df1 = yf.download(
                tickers=f"{target_symbol}.NS",
                period="1d",
                interval="1d",
                auto_adjust=auto_adjust,  # Perfect High/Close ratios 
                threads=False,     # Linear thread-safe tracking loops
                timeout=15,
                progress=False
            )
        df1.columns = df1.columns.get_level_values(0)
        # df1.columns = [str(col).strip().capitalize() for col in df.columns]
        df1.columns = [str(col).strip() for col in df1.columns]
        df['Date'] = pd.to_datetime(df['Date'])
        df1.reset_index(inplace=True)
        print(df1.tail())

        print(f"Stacking historical data with latest daily data for {target_symbol}...")
        cdf = pd.concat([df, df1], ignore_index=True)
        cdf.tail()
        print(f"Removing duplicate dates, keeping the latest entry for {target_symbol}...")
        df = cdf.drop_duplicates(subset=['Date'], keep='last').reset_index(drop=True)
        print(df.tail())

        df['EMA_220'] = df['Close'].ewm(span=220, adjust=False).mean()
        df['SMA_150'] = df['Close'].rolling(window=150).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['Low_52W'] = df['Low'].rolling(window=252).min()

        total_trades, win_rate, net_return = run_strategy_backtest(df, require_ema_dip)
        print(f"Backtest completed for {target_symbol}: Total Trades={total_trades}, Win Rate={win_rate}%, Compounded Return={net_return}%")
    except Exception as e:
        print(f"An error occurred while generating the backtest report: {str(e)}")

    # old_stdout = sys.stdout
    # sys.stdout = buffer = io.StringIO()

    # try:
    #     df = pd.read_csv(csv_path)
    #     # df.columns = [str(col).strip().capitalize() for col in df.columns]
    #     df.columns = [str(col).strip() for col in df.columns]
    #     df = df.dropna(subset=['Close']).reset_index(drop=True)
    #     df['Date'] = pd.to_datetime(df['Date'])
        
    #     df['EMA_220'] = df['Close'].ewm(span=220, adjust=False).mean()
    #     df['SMA_150'] = df['Close'].rolling(window=150).mean()
    #     df['SMA_50'] = df['Close'].rolling(window=50).mean()
    #     df['Low_52W'] = df['Low'].rolling(window=252).min()
    #     historical_ath = []
    #     historical_ath_idx = []
    #     running_max_high, running_max_idx = -1.0, -1
        
    #     for idx in range(len(df)):
    #         historical_ath.append(running_max_high)
    #         historical_ath_idx.append(running_max_idx)
    #         if df['High'].iloc[idx] > running_max_high:
    #             running_max_high = float(df['High'].iloc[idx])
    #             running_max_idx = idx
                
    #     df['Hist_ATH'] = historical_ath
    #     df['Hist_ATH_Idx'] = historical_ath_idx
        
    #     trades = []
    #     in_position = False
    #     entry_price, entry_date = 0.0, None
        
    #     for i in range(252, len(df)):
    #         close = float(df['Close'].iloc[i])
    #         low = float(df['Low'].iloc[i])
    #         sma50 = float(df['SMA_50'].iloc[i])
    #         sma150 = float(df['SMA_150'].iloc[i])
    #         ema220 = float(df['EMA_220'].iloc[i])
    #         low52w = float(df['Low_52W'].iloc[i])
    #         hist_ath = float(df['Hist_ATH'].iloc[i])
    #         ath_idx = int(df['Hist_ATH_Idx'].iloc[i])
            
    #         if not in_position:
    #             # Core Matrix Conditions
    #             cond1 = sma150 > ema220
    #             cond2 = close > sma50
    #             cond3 = sma50 > sma150
    #             cond4 = close > (1.25 * low52w)
    #             cond5 = close > hist_ath
                
    #             # Dynamic Filter Validation Check
    #             cond6 = True
    #             if require_ema_dip and ath_idx != -1:
    #                 if ath_idx + 1 <= i - 1:
    #                     interim_df = df.iloc[ath_idx + 1 : i]
    #                     cond6 = (interim_df['Close'] < interim_df['EMA_220']).any()
    #                 else:
    #                     cond6 = False
                        
    #             if cond1 and cond2 and cond3 and cond4 and cond5 and cond6:
    #                 in_position = True
    #                 entry_price = close
    #                 entry_date = df['Date'].iloc[i]
    #         else:
    #             exit_cond_ema = close < ema220
    #             stop_loss_level = entry_price * 0.85
    #             exit_cond_stop = low <= stop_loss_level
    #             is_last_row = (i == len(df) - 1)
                
    #             if exit_cond_ema or exit_cond_stop or is_last_row:
    #                 in_position = False
    #                 if exit_cond_stop and not exit_cond_ema and not is_last_row:
    #                     exit_price = stop_loss_level
    #                     exit_reason = "15% Stop Loss Hit"
    #                 elif exit_cond_ema and not exit_cond_stop:
    #                     exit_price = close
    #                     exit_reason = "Closed Below 220 EMA"
    #                 elif exit_cond_ema and exit_cond_stop:
    #                     exit_price = stop_loss_level
    #                     exit_reason = "15% Stop Loss Hit (Same Day)"
    #                 else:
    #                     exit_price = close
    #                     exit_reason = "End of Data (Position Active)"
                        
    #                 pnl_pct = ((exit_price - entry_price) / entry_price) * 100
    #                 trades.append({
    #                     'Entry Date': entry_date.strftime('%Y-%m-%d'),
    #                     'Entry Price': round(entry_price, 2),
    #                     'Exit Date': df['Date'].iloc[i].strftime('%Y-%m-%d'),
    #                     'Exit Price': round(exit_price, 2),
    #                     'PnL %': round(pnl_pct, 2),
    #                     'Exit Reason': exit_reason
    #                 })
                    
    #     trades_df = pd.DataFrame(trades)
    #     print("=" * 95)
    #     print(f"STRATEGY BACKTEST HISTORICAL REPORT FOR TRADING SYMBOL: {target_symbol}")
    #     print(f"EMA 220 Dip Correction Constraint Applied: {require_ema_dip}")
    #     print("=" * 95)
    #     if trades_df.empty:
    #         print("No trend trade executions were logged under these rules parameters.")
    #     else:
    #         print(trades_df.to_string(index=False))
    #         wins = (trades_df['PnL %'] > 0).sum()
    #         total = len(trades_df)
    #         print("-" * 95)
    #         print(f"Total Completed Trades : {total}")
    #         print(f"Strategy Win Rate      : {(wins / total) * 100:.2f}%")
    #         print(f"Average Return / Trade : {trades_df['PnL %'].mean():.2f}%")
    #         print(f"Compounded Net Return  : {(((trades_df['PnL %'] / 100 + 1).prod() - 1) * 100):.2f}%")
    #     print("=" * 95)
        
    # except Exception as e:
    #     print(f"An error occurred while generating the backtest report: {str(e)}")
    # finally:
    #     sys.stdout = old_stdout
        
    # output_text = buffer.getvalue()

    
    # Simple selected string injectors for the dropdown on results view
    selected_true = "selected" if require_ema_dip else ""
    selected_false = "selected" if not require_ema_dip else ""
    return render_template('backtest.html', symbol=target_symbol, require_ema_dip=require_ema_dip, auto_adjust=auto_adjust, total_trades=total_trades, win_rate=win_rate, net_return=net_return)    
    
    # return f"""
    # <html>
    #     <head><title>{target_symbol} Historical Backtest</title></head>
    #     <body style="background-color: #1a202c; color: #edf2f7; font-family: -apple-system, BlinkMacSystemFont, sans-serif; padding: 30px;">
    #         <div style="max-width: 1100px; margin: 0 auto;">
    #             <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
    #                 <h2 style="margin: 0; color: #fff;">Strategy Validation Analysis Ledger</h2>
    #                 <a href="/" style="background-color: #4a5568; color: white; padding: 8px 16px; border-radius: 4px; text-decoration: none; font-weight: bold; font-size: 14px;">&larr; Back to Dashboard</a>
    #             </div>
                
    #             <form action="/backtest" method="GET" style="background-color: #2d3748; padding: 15px; border-radius: 6px; margin-bottom: 20px; display: flex; gap: 15px; align-items: center; flex-wrap: wrap;">
    #                 <div>
    #                     <label style="font-weight: bold; color: #cbd5e0; margin-right: 5px;">Ticker:</label>
    #                     <input type="text" name="symbol" value="{target_symbol}" style="padding: 8px 12px; border-radius: 4px; border: 1px solid #4a5568; background-color: #1a202c; color: white; font-size: 14px; text-transform: uppercase; width: 120px;">
    #                 </div>
    #                 <div>
    #                     <label style="font-weight: bold; color: #cbd5e0; margin-right: 5px;">Filter Rules Profile:</label>
    #                     <select name="require_ema_dip" style="padding: 8px 12px; border-radius: 4px; border: 1px solid #4a5568; background-color: #1a202c; color: white; font-size: 14px;">
    #                         <option value="false" {selected_false}>Without 220 EMA Dip adjustment</option>
    #                         <option value="true" {selected_true}>With 220 EMA Dip adjustment</option>
    #                     </select>
    #                 </div>
    #                 <button type="submit" style="background-color: #3182ce; color: white; padding: 8px 16px; border: none; border-radius: 4px; font-weight: bold; cursor: pointer;">Run Backtest</button>
    #             </form>

    #             <pre style="background-color: #2d3748; padding: 20px; border-radius: 6px; overflow-x: auto; font-family: monospace; font-size: 14px; line-height: 1.6; border: 1px solid #4a5568;">{output_text}</pre>
    #         </div>
    #     </body>
    # </html>
    # """

@app.route('/ath-analysis', methods=['GET'])
def ath_analysis():
    """
    Analyzes every historical day a symbol made a new All-Time High (ATH)
    and logs exactly which strategy entry filters failed or passed.
    """
    symbol = request.args.get('symbol', '').upper().strip()
    require_ema_dip = request.args.get('require_ema_dip', 'false').lower() == 'true'
    auto_adjust = request.args.get('auto_adjust', 'true').lower() == 'true'
    print(f"Received ATH analysis request for symbol: {symbol}, require_ema_dip={require_ema_dip}, auto_adjust={auto_adjust}")
    data_folder = DATA_FOLDER_AUTO_ADJUST if auto_adjust else DATA_FOLDER_NO_ADJUST
    
    file_path = os.path.join(data_folder, f"{symbol}.csv")
    print(f"Looking for data file at: {file_path}")
    if not os.path.exists(file_path):
        return jsonify({'error': f'Data file for symbol "{symbol}" not found.'}), 404

    try:
        # Load and conform column headers
        df = pd.read_csv(file_path)
        # df.columns = [str(col).strip().capitalize() for col in df.columns]
        df.columns = [str(col).strip() for col in df.columns]
        df = df.dropna(subset=['Close']).reset_index(drop=True)
        df['Date'] = pd.to_datetime(df['Date'])
        print(df.tail())

        # auto_adjust = True if data_folder == DATA_FOLDER_AUTO_ADJUST else False
        print(f"Fetching latest daily data for {symbol} with auto_adjust={auto_adjust}...")
        df1 = yf.download(
                tickers=f"{symbol}.NS",
                period="1d",
                interval="1d",
                auto_adjust=auto_adjust,  # Perfect High/Close ratios 
                threads=False,     # Linear thread-safe tracking loops
                timeout=15,
                progress=False
            )
        df1.columns = df1.columns.get_level_values(0)
        # df1.columns = [str(col).strip().capitalize() for col in df.columns]
        df1.columns = [str(col).strip() for col in df1.columns]
        df['Date'] = pd.to_datetime(df['Date'])
        df1.reset_index(inplace=True)
        print(df1.tail())

        print(f"Stacking historical data with latest daily data for {symbol}...")
        cdf = pd.concat([df, df1], ignore_index=True)
        cdf.tail()
        print(f"Removing duplicate dates, keeping the latest entry for {symbol}...")
        df = cdf.drop_duplicates(subset=['Date'], keep='last').reset_index(drop=True)
        print(df.tail())

        # Calculate matching strategy indicators
        df['EMA_220'] = df['Close'].ewm(span=220, adjust=False).mean()
        df['SMA_150'] = df['Close'].rolling(window=150).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['Low_52W'] = df['Low'].rolling(window=252).min()


        cdf = pd.concat([df, df1], ignore_index=True)
        cdf.tail()

        ath_logs = []
        running_ath = 0.0
        ath_idx = -1

        # Evaluate starting at index 252 so trailing metrics have structured history
        for i in range(252, len(df)):
            row = df.iloc[i]
            
            # Identify if a brand new breakout peak is printed today
            if row['High'] > running_ath:
                previous_ath = running_ath
                previous_ath_idx = ath_idx
                
                # Update current baseline reference points
                running_ath = row['High']
                ath_idx = i
                
                # Skip log generation for the initial baseline setup row
                if previous_ath == 0:
                    continue

                rejection_reasons = []
                
                # Verify individual rule boundaries
                if not (row['SMA_150'] > row['EMA_220']):
                    rejection_reasons.append("SMA 150 < EMA 220 (Trend Filter)")
                    
                if not (row['Close'] > row['SMA_50']):
                    rejection_reasons.append("Close < SMA 50 (Short-term Pullback)")
                    
                if not (row['SMA_50'] > row['SMA_150']):
                    rejection_reasons.append("SMA 50 < SMA 150 (Order Filter)")
                    
                if not (row['Close'] > (1.25 * row['Low_52W'])):
                    rejection_reasons.append("Close not 25% above 52W Low (Momentum Overstretched)")
                    
                if not (row['Close'] > previous_ath):
                    rejection_reasons.append("Close failed to finish above previous ATH candle high")

                # Structural filter: Confirm if price dropped under EMA 220 during the consolidation phase
                if require_ema_dip and previous_ath_idx != -1:
                    if previous_ath_idx + 1 <= i - 1:
                        interim_segment = df.iloc[previous_ath_idx + 1 : i]
                        has_dipped = (interim_segment['Close'] < interim_segment['EMA_220']).any()
                        if not has_dipped:
                            rejection_reasons.append("Missing dynamic pullback: No Close dipped below EMA 220 since last ATH")
                    else:
                        rejection_reasons.append("Insufficient track history between ATH points to evaluate EMA dip")

                status = "Selected for Entry" if len(rejection_reasons) == 0 else "Rejected"

                ath_logs.append({
                    'date': row['Date'].strftime('%Y-%m-%d'),
                    'ath_high': round(row['High'], 2),
                    'prev_ath': round(previous_ath, 2),
                    'close_price': round(row['Close'], 2),
                    'status': status,
                    'reasons': ", ".join(rejection_reasons) if rejection_reasons else "Passed All Rules"
                })

        # Return historical record sets ordered by newest date first
        ath_logs.reverse()
        return jsonify({'symbol': symbol, 'ath_analysis': ath_logs})

    except Exception as e:
        return jsonify({'error': f'Failed to process analysis: {str(e)}'}), 500
    
if __name__ == '__main__':
    # Render assigns a dynamic port via environment variables. 
    # If it doesn't exist, fall back to your local port 8000.
    port = int(os.environ.get("PORT", 8000))
    
    # On Render, host must be '0.0.0.0' to accept public web traffic.
    # On local, we can bind to '127.0.0.1' or '0.0.0.0' seamlessly.
    if os.environ.get("RENDER"):
        # Render Environment Settings
        app.run(debug=True, host='0.0.0.0', port=port)
    else:
        # Your Local Workspace Settings
        app.run(debug=True, host='127.0.0.1', port=port)
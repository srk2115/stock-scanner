import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
from const import INDEX_SYMBOLS
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

DATA_FOLDER = "data"

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

def find_csv_file(symbol):
    base_symbol = symbol.replace('.NS', '')
    possible_names = [
        f"{base_symbol}.csv", f"{base_symbol}.NS.csv",
        f"{symbol}.csv", f"{symbol}.NS.csv"
    ]
    if not os.path.exists(DATA_FOLDER):
        return None
    for name in possible_names:
        for actual_file in os.listdir(DATA_FOLDER):
            if actual_file.upper() == name.upper():
                return os.path.join(DATA_FOLDER, actual_file)
    return None

def process_single_stock(symbol, require_ema_dip):
    """
    Worker function task assigned to separate threads to isolate 
    and evaluate individual CSV metrics in parallel.
    """
    is_match = False
    current_close = "-"
    val_sma50 = "-"
    val_sma150 = "-"
    val_ema220 = "-"
    val_low52wk = "-"
    val_ath = "-"
    status_str = "FAILED"
    
    csv_path = find_csv_file(symbol)
    if not csv_path:
        status_str = "FILE NOT FOUND"
    else:
        try:
            df = pd.read_csv(csv_path)
            df.columns = [str(col).strip().capitalize() for col in df.columns]
            
            if 'Close' not in df.columns or 'High' not in df.columns or 'Low' not in df.columns:
                status_str = "INVALID CSV FORMAT"
            else:
                df = df.dropna(subset=['Close']).reset_index(drop=True)
                
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
        
    # Execute file calculations concurrently across worker threads
    # max_workers=10 balance overhead on smaller instances effectively
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_single_stock, symbol, require_ema_dip) for symbol in symbols]
        for future in futures:
            results.append(future.result())

    return jsonify({'success': True, 'results': results})

if __name__ == '__main__':
    app.run(debug=True)

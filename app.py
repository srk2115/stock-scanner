import time
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

COOLING_DAYS = 20

def clean_and_format_symbols(raw_input_string):
    if not raw_input_string:
        return []
    cleaned_text = raw_input_string.replace(',', ' ').replace(';', ' ').replace('\n', ' ')
    formatted_symbols = []
    for token in cleaned_text.split():
        token = token.strip().upper()
        if not token:
            continue
        if not token.endswith('.NS') and '.' not in token:
            token = f"{token}.NS"
        formatted_symbols.append(token)
    return formatted_symbols

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/scan', methods=['POST'])
def scan_stocks():
    data = request.get_json() or {}
    raw_input = data.get('symbols', '')
    
    symbols = clean_and_format_symbols(raw_input)
    if not symbols:
        return jsonify({'success': False, 'error': 'No valid symbols provided.'})
        
    results = []
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Accept': 'application/json'
    })

    current_epoch = int(time.time())

    for ticker in symbols:
        is_match = False
        current_close = "-"
        historical_ath = "-"
        current_ema = "-"
        status_str = "FAILED"
        
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            params = {
                'period1': 0,
                'period2': current_epoch,
                'interval': '1d',
                'includeAdjustedClose': 'true'
            }
            
            response = session.get(url, params=params, timeout=15)
            
            if response.status_code != 200:
                status_str = f"HTTP ERROR {response.status_code}"
            else:
                json_data = response.json()
                result_node = json_data.get('chart', {}).get('result', [])
                
                if not result_node or result_node[0] is None:
                    status_str = "INVALID TICKER"
                else:
                    indicators = result_node[0].get('indicators', {}).get('quote', [{}])[0]
                    adjclose_node = result_node[0].get('indicators', {}).get('adjclose', [{}])[0]
                    
                    raw_closes = indicators.get('close', [])
                    raw_highs = indicators.get('high', [])
                    adj_closes = adjclose_node.get('adjclose', []) if adjclose_node else []
                    
                    if not adj_closes:
                        adj_closes = raw_closes
                    
                    clean_closes = []
                    clean_highs = []
                    
                    # --- REPLICATING THE EXCLUSIVELY EXACT YFINANCE MATHEMATICAL PARSING ---
                    for i in range(len(raw_closes)):
                        rc = raw_closes[i]
                        rh = raw_highs[i]
                        ac = adj_closes[i]
                        
                        # Only accept valid, non-zero data strings
                        if rc is not None and rh is not None and ac is not None and rc > 0 and ac > 0:
                            # 1. yfinance replicates 'Close' using the native Adjusted Close array directly
                            yfinance_simulated_close = ac
                            
                            # 2. yfinance recalculates historical Highs by multiplying the raw High by the ratio of (Adj Close / Close)
                            ratio_factor = ac / rc
                            yfinance_simulated_high = rh * ratio_factor
                            
                            clean_closes.append(float(yfinance_simulated_close))
                            clean_highs.append(float(yfinance_simulated_high))

                    if len(clean_closes) < (220 + COOLING_DAYS):
                        status_str = "INSUFFICIENT DATA"
                    else:
                        # --- TECHNICAL CALCULATIONS MATRIX ---
                        k = 2 / (220 + 1)
                        ema_list = []
                        current_ema_val = sum(clean_closes[:220]) / 220
                        ema_list.append(current_ema_val)
                        
                        for price in clean_closes[220:]:
                            current_ema_val = (price * k) + (current_ema_val * (1 - k))
                            ema_list.append(current_ema_val)
                        
                        current_close = round(clean_closes[-1], 2)
                        current_ema = round(ema_list[-1], 2)
                        
                        # Isolate the historical window up to the cooling cutoff point using our mathematically perfect Highs
                        historical_highs = clean_highs[:-COOLING_DAYS]
                        historical_ath = round(max(historical_highs), 2)
                        
                        recent_closes = clean_closes[-COOLING_DAYS:-1]
                        was_below_ath = any(price < historical_ath for price in recent_closes)
                        
                        is_breaking_above_old_ath = current_close > historical_ath
                        is_above_ema = current_close > current_ema
                        
                        if is_breaking_above_old_ath and is_above_ema and was_below_ath:
                            status_str = "MATCH"
                            is_match = True
                        else:
                            status_str = "NO MATCH"
                            
        except Exception as e:
            status_str = f"ERR: {str(e)[:12]}"
            
        results.append({
            'ticker': ticker, 
            'close': current_close, 
            'ath': historical_ath, 
            'ema': current_ema, 
            'status': status_str, 
            'is_match': is_match
        })
        
        time.sleep(0.05)

    return jsonify({'success': True, 'results': results})

if __name__ == '__main__':
    app.run(debug=True)

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from statsmodels.regression.linear_model import OLS
from scipy.stats import zscore as scipy_zscore
from statsmodels.tsa.stattools import adfuller
from pykalman import KalmanFilter
import statsmodels.api as sm
import matplotlib.pyplot as plt
import time
import os
import math

# Initialize MetaTrader 5 connection
if not mt5.initialize():
    raise SystemExit("Initialization failed")

path = os.path.join(mt5.terminal_info().data_path, r'MQL5\Files')
filename = os.path.join(path, 'scores_entry.pickle')

# Initialize Kalman Filter
kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1, n_dim_state=1, em_vars=['transition_covariance', 'observation_covariance'])

def get_data(symbol, timeframe, n, start):
    """
    Retrieve historical data for a symbol.
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, start, n)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def check_cointegration(symbolY, symbolX, start):
    """
    Check cointegration between two symbols.
    """
    df_Y = get_data(symbolY, mt5.TIMEFRAME_D1, periods, start)
    df_X = get_data(symbolX, mt5.TIMEFRAME_D1, periods, start)
  
    indep = df_X['close'].dropna()
    dep = df_Y['close'].dropna()

    X = sm.add_constant(indep)
    lr_model = sm.OLS(dep, X).fit()
    hedge_ratio = lr_model.params[1]
    
    spread = dep - hedge_ratio * indep
    adf_result = adfuller(spread)

    return adf_result

def generate_regression(symbolY, symbolX, start):
    """
    Perform linear regression between the daily returns of two symbols.
    """
    df_Y = get_data(symbolY, mt5.TIMEFRAME_D1, periods, start)
    df_X = get_data(symbolX, mt5.TIMEFRAME_D1, periods, start)

    data = pd.DataFrame({
        'close_y': df_Y['close'].pct_change().dropna(),
        'close_x': df_X['close'].pct_change().dropna()
    }).dropna()

    X = np.vstack([np.ones(len(data['close_y'])), data['close_y']]).T
    model = OLS(data['close_x'], X).fit()
    hedge_ratio = model.params[1]
    
    spread = data['close_x'] - hedge_ratio * data['close_y']
    z_scores = scipy_zscore(spread)

    state_means, state_covariances = kf.smooth(z_scores)

    z_scores_lag = np.roll(z_scores, 1)
    z_scores_lag[0] = 0
    z_scores_diff = z_scores - z_scores_lag

    model_ar = sm.OLS(z_scores_diff, sm.add_constant(z_scores_lag))
    results = model_ar.fit()

    theta = results.params[1]
    half_life = -np.log(2) / theta

    return z_scores, half_life, state_means, hedge_ratio

def execute_trade(symbol, action, volume, price, comment):
    """
    Execute a trade for a given symbol.
    """
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": action,
        "price": price,
        "sl": 0.0,
        "tp": 0.0,
        "deviation": 10,
        "magic": 234000,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Failed to execute {action} on {symbol}, Error code: {result.retcode}")
    return result

def execute_trade_short_long(symbolY, symbolX, slope):
    volumeY, volume_X = volume_adjust(symbolY, symbolX, slope)
    execute_trade(symbolY, mt5.ORDER_TYPE_SELL, volumeY, mt5.symbol_info_tick(symbolY).bid, "dependent")
    execute_trade(symbolX, mt5.ORDER_TYPE_BUY, volume_X, mt5.symbol_info_tick(symbolX).ask, "independent")

def execute_trade_long_short(symbolY, symbolX, slope):
    volumeY, volume_X = volume_adjust(symbolY, symbolX, slope)
    execute_trade(symbolY, mt5.ORDER_TYPE_BUY, volumeY, mt5.symbol_info_tick(symbolY).ask, "dependent")
    execute_trade(symbolX, mt5.ORDER_TYPE_SELL, volume_X, mt5.symbol_info_tick(symbolX).bid, "independent")

def execute_trade_long_long(symbolY, symbolX, slope):
    volumeY, volume_X = volume_adjust(symbolY, symbolX, slope)
    execute_trade(symbolY, mt5.ORDER_TYPE_BUY, volumeY, mt5.symbol_info_tick(symbolY).ask, "dependent")
    execute_trade(symbolX, mt5.ORDER_TYPE_BUY, volume_X, mt5.symbol_info_tick(symbolX).ask, "independent")

def execute_trade_short_short(symbolY, symbolX, slope):
    volumeY, volume_X = volume_adjust(symbolY, symbolX, slope)
    execute_trade(symbolY, mt5.ORDER_TYPE_SELL, volumeY, mt5.symbol_info_tick(symbolY).bid, "dependent")
    execute_trade(symbolX, mt5.ORDER_TYPE_SELL, volume_X, mt5.symbol_info_tick(symbolX).bid, "independent")

def verify_pairs(min_zscore, half_life_max):
    major_pairs_y = ["USDCHF", "GBPUSD", "USDJPY", "AUDNZD", "EURCAD", "CHFJPY", "AUDCHF", "CADCHF", "GBPCHF", "EURCHF", "NZDCAD", "EURMXN"]
    major_pairs_x = ["CADJPY", "AUDJPY", "GBPJPY", "AUDUSD", "EURAUD", "EURUSD", "AUDJPY", "CADJPY", "USDMXN", "EURJPY", "NZDJPY", "SP500"]

    for i in range(len(major_pairs_y)):
        for j in range(len(major_pairs_x)):
            spread_y = mt5.symbol_info(major_pairs_y[i]).spread
            spread_x = mt5.symbol_info(major_pairs_x[j]).spread

            if spread_y > max_spread or spread_x > max_spread:
                continue

            adf_result = check_cointegration(major_pairs_y[i], major_pairs_x[j], 0)
            if adf_result[1] >= 0.05:
                continue

            z_scores, half_life, state_means, hedge_ratio = generate_regression(major_pairs_y[i], major_pairs_x[j], 0)
            if abs(z_scores[-1]) > min_zscore and half_life < half_life_max:
                return major_pairs_y[i], major_pairs_x[j], hedge_ratio, state_means[-1][0], z_scores[-1]
    
    return None, None, None, None, None

def volume_adjust(symbolY, symbolX, hedge_ratio):
    """
    Adjust volumes based on hedge ratio.
    """
    min_lot_Y = mt5.symbol_info(symbolY).volume_min
    min_lot_X = mt5.symbol_info(symbolX).volume_min

    leverage = mt5.account_info().leverage
    total_investment = mt5.account_info().equity / leverage

    investment_asset_y = total_investment * abs(hedge_ratio)
    investment_asset_x = total_investment - investment_asset_y

    volume_y = max(min_lot_Y * (investment_asset_y), min_lot_Y)
    volume_x = max(min_lot_X * (investment_asset_x), min_lot_X)

    volume_y = round(volume_y, 2)
    volume_x = round(volume_x, 2)
    
    return volume_y, volume_x

def close_all_positions():
    """
    Close all open positions.
    """
    positions = mt5.positions_get()
    if positions is None or len(positions) == 0:
        return

    for position in positions:
        symbol = position.symbol
        ticket = position.ticket
        volume = position.volume
        position_type = position.type

        if position_type == mt5.ORDER_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "Close position",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to close position {ticket} on {symbol}, Error code: {result.retcode}")
        else:
            print(f"Successfully closed position {ticket} on {symbol}")

def trailing_stop(symbol, position_type, price_current, stop_loss, ticket):
    """
    Implement trailing stop logic.
    """
    profit = mt5.account_info().profit
    symbol_info = mt5.symbol_info(symbol)
    point = symbol_info.point

    if profit >= PROFIT_THRESHOLD:
        if position_type == mt5.ORDER_TYPE_BUY:
            new_stop_loss = price_current - (TRAILING_DISTANCE_POINTS * point)
            if stop_loss == 0.0 or new_stop_loss > stop_loss:
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": symbol,
                    "position": ticket,
                    "sl": new_stop_loss,
                    "tp": 0.0,
                }
                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    print(f"Failed to update SL for BUY {symbol}, ticket {ticket}: {result.comment}")
        elif position_type == mt5.ORDER_TYPE_SELL:
            new_stop_loss = price_current + (TRAILING_DISTANCE_POINTS * point)
            if stop_loss == 0.0 or new_stop_loss < stop_loss:
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": symbol,
                    "position": ticket,
                    "sl": new_stop_loss,
                    "tp": 0.0,
                }
                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    print(f"Failed to update SL for SELL {symbol}, ticket {ticket}: {result.comment}")

def check_trading_time():
    """
    Check if the current time is within the trading window.
    """
    now = datetime.now(timezone.utc)
    return trading_time_start <= now <= trading_time_end

def is_close_trading_time():
    """
    Check if it's close to the end of the trading window.
    """
    before_end = trading_time_end - timedelta(minutes=5)
    now = datetime.now(timezone.utc)
    return now > before_end

def process_strategy(symbolY, symbolX, entry_level, z_score, slope):
    """
    Process trading strategy based on z-score and slope.
    """
    if total_positions < max_positions:
        if slope > 0:
            if z_score < -entry_level:
                execute_trade_long_short(symbolY, symbolX, slope)
            elif z_score > entry_level:
                execute_trade_short_long(symbolY, symbolX, slope)
        elif slope < 0:
            if z_score < -entry_level:
                execute_trade_long_long(symbolY, symbolX, slope)
            elif z_score > entry_level:
                execute_trade_short_short(symbolY, symbolX, slope)

# Configuration parameters
last_profit = 0
stop_level = 0
gain_level = 0
max_risk = 0.065
max_grids = 4
stop_loss = 0.80
take_profit = 0.80
grid_distance = 0.25
stop_offset = 0.05
min_z_score = 1.8
additional_grid = 0.20
min_kalman = 0.60
half_life_max = 0.9
max_positions = 6
max_spread = 8
volume = 5
periods = 60
TRAILING_DISTANCE_POINTS = 30
PROFIT_THRESHOLD = 10.0
MAGIC = 234000

# Trading time parameters
today = datetime.now(timezone.utc)
trading_time_start = today.replace(hour=4, minute=0, second=0, microsecond=0)
trading_time_end = trading_time_start + timedelta(hours=18)
trade_time = check_trading_time()

while True:
    if check_trading_time():
        equity = mt5.account_info().equity
        balance = mt5.account_info().balance
        free_margin = mt5.account_info().margin_free
        original_margin = mt5.account_info().margin
        total_positions = mt5.positions_total()

        if total_positions > 0:
            grid_count = total_positions / 2
            positions = mt5.positions_get()
            df = pd.DataFrame(list(positions), columns=positions[0]._asdict().keys())
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.drop(['time_update', 'time_msc', 'time_update_msc', 'external_id'], axis=1, inplace=True)

            for position in positions:
                if position.comment == 'independent':
                    ticket_x = position.ticket
                    position_magic = position.magic
                    open_position_x = position.symbol
                    volume_x = position.volume
                    price_current_x = position.price_current
                    stop_loss_x = position.sl
                    type_position_x = position.type
                elif position.comment == 'dependent':
                    ticket_y = position.ticket
                    open_position_y = position.symbol
                    position_magic = position.magic
                    volume_y = position.volume
                    price_open_y = position.price_open
                    price_current_y = position.price_current
                    stop_loss_y = position.sl
                    type_position_y = position.type

            if position_magic == MAGIC and ((open_position_x is None) ^ (open_position_y is None)):
                net_profit_reached = True

            if open_position_y is not None:
                trailing_stop(open_position_y, type_position_y, price_current_y, stop_loss_y, ticket_y)
            if open_position_x is not None:
                trailing_stop(open_position_x, type_position_x, price_current_x, stop_loss_x, ticket_x)

            if open_position_x is not None and open_position_y is not None:
                adf_result = check_cointegration(open_position_y, open_position_x, 0)
                z_scores, half_life, state_means, hedge_ratio = generate_regression(open_position_y, open_position_x, 0)

                current_kalman = state_means[-1][0]
                z_score = z_scores[periods - 1]
                grid_entry = min_z_score + (grid_count) * additional_grid
                process_strategy(open_position_y, open_position_x, grid_entry, z_score, hedge_ratio)
        elif total_positions == 0:
            symbolY, symbolX, slope, state_mean, zscore = verify_pairs(min_z_score, half_life_max)
            if symbolY and symbolX:
                process_strategy(symbolY, symbolX, min_z_score, zscore, slope)
            else:
                print("No pairs found")
    else:
        close_all_positions()
  
    time.sleep(10)

# Shutdown MetaTrader 5 connection
mt5.shutdown()

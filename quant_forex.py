import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from statsmodels.regression.linear_model import OLS
from scipy.stats import zscore as scipy_zscore
from statsmodels.tsa.stattools import adfuller
from pykalman import KalmanFilter
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt
import time
import os
from arch import arch_model
import math
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize MetaTrader 5 connection
if not mt5.initialize():
    logging.error("Initialization failed")
    mt5.shutdown()
    quit()

path = os.path.join(mt5.terminal_info().data_path, r'MQL5\Files')
filename = os.path.join(path, 'scores_entry.pickle')

# Initialize Kalman
kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1, n_dim_state=1, em_vars=['transition_covariance', 'observation_covariance'])

def get_data(symbol, timeframe, n, start):
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, start, n)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    except Exception as e:
        logging.error(f"Failed to get data for {symbol}: {e}")
        return pd.DataFrame()

def check_cointegration(symbolY, symbolX, start):
    df_Y = get_data(symbolY, mt5.TIMEFRAME_D1, periods, start)
    df_X = get_data(symbolX, mt5.TIMEFRAME_D1, periods, start)
    
    indep = df_X['close'].dropna()
    dep = df_Y['close'].dropna()

    X = sm.add_constant(indep)
    lr_model = sm.OLS(dep, X).fit()
    hedge_ratio = lr_model.params[1]
    
    spread = dep - hedge_ratio * indep
    adf_result = adfuller(spread)

    return adf_result, hedge_ratio

def generate_regression(symbolY, symbolX, start):
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

def execute_trade(request):
    try:
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Failed to execute trade: {result.comment}")
        return result
    except Exception as e:
        logging.error(f"Trade execution error: {e}")

def volume_adjust(symbolY, symbolX, hedge_ratio):
    min_lot_Y = mt5.symbol_info(symbolY).volume_min
    min_lot_X = mt5.symbol_info(symbolX).volume_min

    leverage = mt5.account_info().leverage
    total_investment = mt5.account_info().equity / leverage

    investment_asset_y = total_investment * abs(hedge_ratio)
    investment_asset_x = total_investment - investment_asset_y

    volume_y = max(min_lot_Y * investment_asset_y, min_lot_Y)
    volume_x = max(min_lot_X * investment_asset_x, min_lot_X)

    volume_y = round(volume_y, 2)
    volume_x = round(volume_x, 2)
    
    logging.info(f"Proportion {symbolY} volume amount is {volume_y} and {symbolX} volume amount is {volume_x} with hedge ratio {hedge_ratio}")

    return volume_y, volume_x

def main():
    while True:
        if check_trading_time():
            equity = mt5.account_info().equity
            balance = mt5.account_info().balance
            free_margin = mt5.account_info().margin_free
            original_margin = mt5.account_info().margin
            total_positions = mt5.positions_total()

            if total_positions > 0:
                positions = mt5.positions_get()
                for position in positions:
                    if position.comment == 'independent':
                        trailing_stop(position)
                    elif position.comment == 'dependent':
                        trailing_stop(position)

                adf_result, slope = check_cointegration(open_position_y, open_position_x, 0)
                z_scores, half_life, state_means, hedge_ratio = generate_regression(open_position_y, open_position_x, 0)
                process_strategy(open_position_y, open_position_x, grid_entry, z_score, slope)

            elif total_positions == 0:
                symbolY, symbolX, slope, state_mean, zscore = verify_pairs(min_z_score, half_life_max)
                if symbolY and symbolX:
                    process_strategy(symbolY, symbolX, min_z_score, zscore, slope)
                else:
                    logging.info("No pairs found")
        else:
            close_all_positions()

        time.sleep(10)

if __name__ == "__main__":
    main()
    mt5.shutdown()

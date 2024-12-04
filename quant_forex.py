import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime,timedelta, timezone, time as datetime_time
from statsmodels.regression.linear_model import OLS
from scipy.stats import zscore as scipy_zscore,pearsonr
from statsmodels.tsa.stattools import adfuller, coint
from pykalman import KalmanFilter
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt
import time
import os
from arch import arch_model
import math


# Initialize MetaTrader 5 connection
if not mt5.initialize():
    print("Initialization failed")
    mt5.shutdown()
    quit()

path=os.path.join(mt5.terminal_info().data_path,r'MQL5\Files')
filename = os.path.join(path,'scores_entry.pickle')

# Initialize Kalman    
kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1, n_dim_state=1, em_vars=['transition_covariance', 'observation_covariance'])


# Define function to get historical data
def get_data(symbol, timeframe, n,start):
     rates = mt5.copy_rates_from_pos(symbol, timeframe, start, n)
     df = pd.DataFrame(rates)
     df['time'] = pd.to_datetime(df['time'], unit='s')
     return df


def check_cointegration(symbolY,symbolX,start):
    df_Y = get_data(symbolY, mt5.TIMEFRAME_D1, periods, start)
    df_X = get_data(symbolX, mt5.TIMEFRAME_D1, periods, start)
  
    indep = df_X['close'].dropna()
    dep = df_Y['close'].dropna()

    X = sm.add_constant(indep)
    lr_model = sm.OLS(dep, X).fit()
    hedge_ratio = lr_model.params[1]
    
    # Calculate the spread (residuals)
    spread = dep - hedge_ratio * indep

    # Step 3: Run the Augmented Dickey-Fuller test on the spread
    adf_result = adfuller(spread)

    return adf_result

# Get historical data
def generate_regression(symbolY,symbolX, start):
    
    df_Y = get_data(symbolY, mt5.TIMEFRAME_D1, periods, start)
    df_X = get_data(symbolX, mt5.TIMEFRAME_D1, periods, start)

    data = pd.DataFrame({
    'close_y': df_Y['close'].pct_change().dropna(),  # Daily returns for dependent
    'close_x': df_X['close'].pct_change().dropna()  # Daily returns for independent
    }).dropna()

    # Step 3.1: Perform linear regression to find the hedge ratio
    X = np.vstack([np.ones(len(data['close_y'])), data['close_y']]).T
    model = OLS(data['close_x'], X).fit()
    hedge_ratio = model.params[1]
    
    # Step 3.2: Calculate the spread (series2 - hedge_ratio * series1)
    spread = data['close_x'] - hedge_ratio * data['close_y']
    
    # Step 3.3: Calculate and return the Z-score of the spread
    z_scores = scipy_zscore(spread)

    state_means, state_covariances = kf.smooth(z_scores)

    # Step 5: Estimate Half-life
    # Fit an AR(1) model to the Z-scores
    z_scores_lag = np.roll(z_scores, 1)
    z_scores_lag[0] = 0  # Replace the first element with 0 since there is no lag for the first element
    z_scores_diff = z_scores - z_scores_lag

    # Use statsmodels to perform the linear regression for the AR(1) process
    model_ar = sm.OLS(z_scores_diff, sm.add_constant(z_scores_lag))
    results = model_ar.fit()

      # Calculate the half-life
    theta = results.params[1]
    half_life = -np.log(2) / theta

    return z_scores,half_life,state_means,hedge_ratio


def execute_trade_short_long(symbolY,symbolX,slope):
      # prepare the Short request
        volumeY, volume_X = volume_adjust(symbolY,symbolX,slope)
        request_short = {
           "action": mt5.TRADE_ACTION_DEAL,
           "symbol": symbolY,
           "volume": volumeY,
           "type": mt5.ORDER_TYPE_SELL,
           "zscore": mt5.symbol_info_tick(symbolY).bid,
           "sl": 0.0,
           "tp": 0.0,
           "deviation": 10,
           "magic": 234000,
           "comment": "dependent",
           "type_time": mt5.ORDER_TIME_GTC,
           "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result_short_order_check = mt5.order_check(request_short)
        print("Resultado do short check order (dependente) ", result_short_order_check)       
        
        # prepare the Long request
        point=mt5.symbol_info(symbolX).point
        request_long = {
           "action": mt5.TRADE_ACTION_DEAL,
           "symbol": symbolX,
           "volume": volume_X,
           "type": mt5.ORDER_TYPE_BUY,
           "zscore": mt5.symbol_info_tick(symbolX).ask,
           "sl": 0.0,
           "tp": 0.0,
           "deviation": 10,
           "magic": 234000,
           "comment": "independent",
           "type_time": mt5.ORDER_TIME_GTC,
           "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result_long_order_check = mt5.order_check(request_long)
        print("Resultado do long check order (independente) ", result_long_order_check)

        result_sell_order = mt5.order_send(request_short)
        result_buy_order = mt5.order_send(request_long)
        print("Resultado do short (dependente) ", result_sell_order)
        print("Resultado do long (independente) ", result_buy_order)
        
        

def execute_trade_long_short(symbolY,symbolX,slope):
      # prepare the Short request
        volumeY, volume_X = volume_adjust(symbolY,symbolX,slope)
        request_short = {
           "action": mt5.TRADE_ACTION_DEAL,
           "symbol": symbolY,
           "volume": volumeY,
           "type": mt5.ORDER_TYPE_BUY,
           "zscore": mt5.symbol_info_tick(symbolY).ask,
           "sl": 0.0,
           "tp": 0.0,
           "deviation": 10,
           "magic": 234000,
           "comment": "dependent",
           "type_time": mt5.ORDER_TIME_GTC,
           "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result_short_order_check = mt5.order_check(request_short)
        print("Resultado do short check order (dependente) ", result_short_order_check)            
        
        
        # prepare the Long request
        
        request_long = {
           "action": mt5.TRADE_ACTION_DEAL,
           "symbol": symbolX,
           "volume": volume_X,
           "type": mt5.ORDER_TYPE_SELL,
           "zscore": mt5.symbol_info_tick(symbolX).bid,
           "sl": 0.0,
           "tp": 0.0,
           "deviation": 10,
           "magic": 234000,
           "comment": "independent",
           "type_time": mt5.ORDER_TIME_GTC,
           "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result_long_order_check = mt5.order_check(request_long)
        print("Resultado do long check order (dependente) ", result_long_order_check)

        result_long_independent = mt5.order_send(request_long) 
        result_short_dependent = mt5.order_send(request_short)
        print(f" order_send done, {result_long_independent} and {result_short_dependent} ") 



def execute_trade_long_long(symbolY,symbolX,slope):
      # prepare the Long request
        volumeY, volume_X = volume_adjust(symbolY,symbolX,slope)
        request_long_dependent = {
           "action": mt5.TRADE_ACTION_DEAL,
           "symbol": symbolY,
           "volume": volumeY,
           "type": mt5.ORDER_TYPE_BUY,
           "zscore": mt5.symbol_info_tick(symbolY).ask,
           "sl": 0.0,
           "tp": 0.0,
           "deviation": 10,
           "magic": 234000,
           "comment": "dependent",
           "type_time": mt5.ORDER_TIME_GTC,
           "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result_long_dependent_order_check = mt5.order_check(request_long_dependent)
        print("Resultado do long check order (dependente) ", result_long_dependent_order_check)
        
        # prepare the Long request
        point=mt5.symbol_info(symbolX).point
        request_long_independent = {
           "action": mt5.TRADE_ACTION_DEAL,
           "symbol": symbolX,
           "volume": volume_X,
           "type": mt5.ORDER_TYPE_BUY,
           "zscore": mt5.symbol_info_tick(symbolX).ask,
           "sl": 0.0,
           "tp": 0.0,
           "deviation": 10,
           "magic": 234000,
           "comment": "independent",
           "type_time": mt5.ORDER_TIME_GTC,
           "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result_long_independent_order_check = mt5.order_check(request_long_independent)
        print("Resultado do long check order (independente) ", result_long_independent_order_check)
        result_long_independent = mt5.order_send(request_long_independent)
        result_long_dependent = mt5.order_send(request_long_dependent)
        print(f" order_send done, {result_long_independent} and {result_long_dependent} ")
        

def execute_trade_short_short(symbolY,symbolX,slope):
      # prepare the Long request
        volumeY, volume_X = volume_adjust(symbolY,symbolX,slope)
        request_short_y = {
           "action": mt5.TRADE_ACTION_DEAL,
           "symbol": symbolY,
           "volume": volumeY,
           "type": mt5.ORDER_TYPE_SELL,
           "zscore": mt5.symbol_info_tick(symbolY).bid,
           "sl": 0.0,
           "tp": 0.0,
           "deviation": 10,
           "magic": 234000,
           "comment": "dependent",
           "type_time": mt5.ORDER_TIME_GTC,
           "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result_short_order_check = mt5.order_check(request_short_y)
        print("Resultado do short check order (dependente) ", result_short_order_check)
        
        
        # prepare the Long request
    
        request_short_x = {
           "action": mt5.TRADE_ACTION_DEAL,
           "symbol": symbolX,
           "volume": volume_X,
           "type": mt5.ORDER_TYPE_SELL,
           "zscore": mt5.symbol_info_tick(symbolX).bid,
           "sl": 0.0,
           "tp": 0.0,
           "deviation": 10,
           "magic": 234000,
           "comment": "independent",
           "type_time": mt5.ORDER_TIME_GTC,
           "type_filling": mt5.ORDER_FILLING_IOC,
        }
     
        request_short_x_check = mt5.order_check(request_short_x)
        print("Resultado do short check order (independente) ", request_short_x_check)

        result_short_independent = mt5.order_send(request_short_y)
        result_short_dependent = mt5.order_send(request_short_x)
        print(f" order_send done, {result_short_independent} and {result_short_dependent} ")
        
def verify_pairs(min_zscore, half_life_max):
     major_pairs_y = ["USDCHF","GBPUSD","USDJPY","AUDNZD","EURCAD","CHFJPY","AUDCHF","CADCHF","GBPCHF","EURCHF","NZDCAD","EURMXN"] #["WDOX24"] #["WDOX24"]
     major_pairs_x = ["CADJPY","AUDJPY","GBPJPY","AUDUSD","EURAUD","EURUSD","AUDJPY","CADJPY","USDMXN","EURJPY","NZDJPY","SP500"] #["WINZ24"] #["WINZ24"]

     print(f"Selecting pairs with min z score {min_zscore} and half life {half_life_max} ")
     
     is_stationary = False


     for i in range(len(major_pairs_y)):
          for j in range(len(major_pairs_x)):

                    spread_y = mt5.symbol_info(major_pairs_y[i]).spread
                    spread_x = mt5.symbol_info(major_pairs_x[j]).spread

                    is_spread_allowed = (spread_y <= max_spread) and (spread_x <= max_spread)
                    
                    adf_result = check_cointegration(major_pairs_y[i],major_pairs_x[j],0)
                    p_value = adf_result[1]                                    

                    is_stationary = p_value < 0.05

                    if not (is_spread_allowed and is_stationary):
                            continue
                    
                    #print(f"Dependente {major_pairs_y[i]} e independente {major_pairs_x[j]}")
                    z_scores,half_life,state_means,hedge_ratio = generate_regression(major_pairs_y[i],major_pairs_x[j],0)
                    volume_adjust(major_pairs_y[i],major_pairs_x[j], hedge_ratio)
                    z_score = z_scores[periods - 1]
                    state_mean = state_means[-1:][0][0]
                    hours_left = math.ceil(half_life*24)
                    print(f"Zscore {z_score} e  min zscore {min_z_score} para {major_pairs_y[i]} e independente {major_pairs_x[j]}")
                    #print("P-value ", p_value) 
                    #plot_values(z_scores,state_means,half_life,major_pairs_y[i],major_pairs_x[j])
                    #print(f" Half life {half_life} e  max half life {half_life_max}")   
                    if (abs(z_score) > min_zscore and half_life < half_life_max ):
                        return major_pairs_y[i], major_pairs_x[j], hedge_ratio, state_mean, z_score
    
     return None, None, None, None, None

                        

def calculate_max_volume(symbol, action):
    account_currency=mt5.account_info().currency
    lot = mt5.symbol_info(symbol).volume_step
    margin = mt5.account_info().margin
    margin_per_asset = margin/2
    asset_margin = 0
    margin_factor = 0

    current_margin = mt5.account_info().margin_free
    ask=mt5.symbol_info_tick(symbol).ask
    margin_for_one_lot = mt5.order_calc_margin(action,symbol,1.0,ask) 
    print(f" Current margin {current_margin} and margin for one lot {margin_for_one_lot} ")
    
    min_lot = mt5.symbol_info(symbol).volume_min
    max_volume = mt5.symbol_info(symbol).volume_max
    volume_step = mt5.symbol_info(symbol).volume_step

    lot_size = current_margin/margin_for_one_lot

    max_lot_size = max(min_lot, math.floor(lot_size / volume_step) * volume_step)

    print(f"Max lot size : {max_lot_size}")

    if (action == mt5.ORDER_TYPE_BUY):
      ask=mt5.symbol_info_tick(symbol).ask
      asset_margin=mt5.order_calc_margin(action,symbol,lot,ask) 
      if asset_margin != None:
          print("   {} buy {} lot margin: {} {}".format(symbol,lot,asset_margin,account_currency)); 
      else: 
          print("order_calc_margin failed: , error code =", mt5.last_error()) 
    
    elif (action == mt5.ORDER_TYPE_SELL):
      
      lot=mt5.symbol_info(symbol).volume_step
      bid=mt5.symbol_info_tick(symbol).bid
      asset_margin=mt5.order_calc_margin(action,symbol,lot,bid) 
      if asset_margin != None:
          print("   {} sell {} lot margin: {} {}".format(symbol,lot,asset_margin,account_currency)); 
      else: 
          print("order_calc_margin failed: , error code =", mt5.last_error())

    return asset_margin
                
def plot_values(z_scores,state_means,half_life,symbol_y,symbol_x):
     
        # Plot the original and smoothed Z-scores
        plt.figure(figsize=(14, 7))
        plt.plot(z_scores, label='Original Z-scores', alpha=0.5, linewidth=2)
        plt.plot(state_means, label='Kalman Filter Smoothed Z-scores', color='red', linewidth=2)    
        plt.axhline(0, color='black', linestyle='--')
        plt.axhline(1, color='blue', linestyle='--')
        plt.axhline(2, color='green', linestyle='--', label='+2 Std Dev')
        plt.axhline(-1, color='red', linestyle='--')
        plt.axhline(-2, color='green', linestyle='--', label='-2 Std Dev')
        plt.xlabel('Date')
        plt.ylabel('Z-score')
        plt.title(f'Z-scores for assets {symbol_y} and {symbol_x} half life {half_life}')
        plt.legend()
        plt.show()

def volume_adjust(symbolY,symbolX,hedge_ratio):
   

    min_lot_Y = mt5.symbol_info(symbolY).volume_min
    min_lot_X = mt5.symbol_info(symbolX).volume_min

    leverage = mt5.account_info().leverage

    # Total investment amount
    total_investment = mt5.account_info().equity/leverage # Example amount

    # Adjust lot sizes based on volatility
    investment_asset_y = total_investment*abs(hedge_ratio)
    investment_asset_x = total_investment - investment_asset_y

    volume_y = max(min_lot_Y*(investment_asset_y),min_lot_Y)
    volume_x = max(min_lot_X*(investment_asset_x),min_lot_X)

    volume_y = round(volume_y,2)
    volume_x = round(volume_x,2)
    
    print(f"Proportion {symbolY} volume amount is {volume_y} and {symbolX} volume amount is {volume_x} with hedge ratio {hedge_ratio} ")

    return volume_y,volume_x

def close_all_positions():
    # Get all open positions
    positions = mt5.positions_get()
    if positions is not None or len(positions) > 0:
        # Loop through each position and close it
        for position in positions:
            symbol = position.symbol
            ticket = position.ticket
            volume = position.volume
            position_type = position.type  # 0 for buy, 1 for sell

        # Determine the opposite order type to close the position
            if position_type == mt5.ORDER_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                zscore = mt5.symbol_info_tick(symbol).bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                zscore = mt5.symbol_info_tick(symbol).ask

        # Create a close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "position": ticket,
                "zscore": zscore,
                "deviation": 20,
                "magic": 234000,
                "comment": "Close position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

        # Send the close request
            result = mt5.order_send(request)

        # Check the result
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"Failed to close position {ticket} on {symbol}, Error code: {result.retcode}")
            else:
                print(f"Successfully closed position {ticket} on {symbol}")

def close_position_criteria():

          for position in positions:
            if position.comment == 'independent':
                symbolX = position.symbol
            elif position.comment == 'dependent':
                symbolY = position.symbol             
           
        
          close_trades_time = is_close_trading_time()

          close_profit_condition = False
        
          print(f" Close profit condition {close_profit_condition}")          

          risk_percentage = abs(equity/balance - 1)
          risk_limit = risk_percentage >= max_risk
                    
          
          print(f" Risk percentage {risk_percentage} max risk {max_risk} is it at the limit ? {risk_limit}")          
      
          if (close_trades_time):
              close_all_positions()

def trailing_stop(symbol,position_type,price_current,stop_loss,ticket):
    # Get all open positions
    profit = mt5.account_info().profit
    symbol_info = mt5.symbol_info(symbol)
    point = symbol_info.point
    new_stop_loss = price_current - (TRAILING_DISTANCE_POINTS * point)
      
    # Check if profit exceeds the threshold
    if profit >= PROFIT_THRESHOLD:
          
        # Calculate the new stop loss level
        if position_type == mt5.ORDER_TYPE_BUY:
            new_stop_loss = price_current - (TRAILING_DISTANCE_POINTS * point)
            
            # Only modify if the new SL is higher than the current one
            if (stop_loss == 0.0) or new_stop_loss > stop_loss:
                print(f"Updating Trailing para {symbol} com profit {profit} new stop {new_stop_loss} old stop {stop_loss} ")
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": symbol,
                    "position": ticket,
                    "sl": new_stop_loss,
                    "tp": position.tp,
                }
                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    print(f"Failed to update SL for BUY {symbol}, ticket {ticket}: {result.comment}")
                else:
                    print(f"Trailing stop updated for BUY {symbol}, ticket {ticket}. New SL: {new_stop_loss:.5f}")
        elif position_type == mt5.ORDER_TYPE_SELL:
            new_stop_loss = price_current + (TRAILING_DISTANCE_POINTS * point)
            # Only modify if the new SL is lower than the current one
            if (stop_loss == 0.0) or new_stop_loss < stop_loss:
                print(f"Updating Trailing para {symbol} com profit {profit} new stop {new_stop_loss} old stop {stop_loss} ")
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": symbol,
                    "position": ticket,
                    "sl": new_stop_loss,
                    "tp": position.tp,
                }
                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    print(f"Failed to update SL for SELL {symbol}, ticket {ticket}: {result.comment}")
                else:
                    print(f"Trailing stop updated for SELL {symbol}, ticket {ticket}. New SL: {new_stop_loss:.5f}")

    
def check_trading_time():
  now = datetime.now(timezone.utc)
  trade_time = (now > trading_time_start) and (now < trading_time_end)
  return trade_time
   

def is_close_trading_time():
  before_end = trading_time_end - timedelta(minutes=5)
  now = datetime.now(timezone.utc)
  time_to_close = now > before_end
  
  return time_to_close

def calculate_volumes(hedge_ratio, tick_value_x, tick_value_y, max_volume_x):
    # Adjust the hedge ratio based on tick values
    adjusted_ratio = abs(hedge_ratio) * (tick_value_x / tick_value_y)
    
    # Calculate volumes
    volume_x = max_volume_x
    volume_y = adjusted_ratio * volume_x
    
    return volume_x, volume_y

def process_strategy(symbolY,symbolX,entry_level,z_score,slope):
     print(f"Processando estratégia para {symbolY} e {symbolX}")

     # Logica de operacoes
     if (total_positions < max_positions):
          if (slope > 0):
            if (z_score < -entry_level):
                  z_score_open = execute_trade_long_short(symbolY,symbolX,slope)

            elif (z_score > entry_level):
                  z_score_open = execute_trade_short_long(symbolY,symbolX,slope)

          elif (slope < 0):
            if (z_score < -entry_level):
                  z_score_open = execute_trade_long_long(symbolY,symbolX,slope)

            elif (z_score > entry_level):
                  z_score_open = execute_trade_short_short(symbolY,symbolX,slope)


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
volume = 5  # Volume for trading
periods = 60
# Trailing stop parameters
TRAILING_DISTANCE_POINTS = 30
PROFIT_THRESHOLD = 10.0  # USD
MAGIC = 234000

today = datetime.now(timezone.utc)

trading_time_start = today.replace(hour=4, minute=0, second=0, microsecond=0)
trading_time_end = trading_time_start + timedelta(hours=18,minutes=0)
trade_time = check_trading_time()
print(f"Trade will start at {trading_time_start} until {trading_time_end} Trade time ? {trade_time}")
net_profit_reached = False
open_position_x = None
open_position_y = None

while True:

  if check_trading_time():
 
      equity = mt5.account_info().equity
      balance = mt5.account_info().balance
      free_margin = mt5.account_info().margin_free
      original_margin = mt5.account_info().margin
      total_positions = mt5.positions_total()

      if (total_positions > 0):
            
          grid_count = (total_positions/2)
          positions = mt5.positions_get()
          df=pd.DataFrame(list(positions),columns=positions[0]._asdict().keys())
          df['time'] = pd.to_datetime(df['time'], unit='s') 
          df.drop(['time_update', 'time_msc', 'time_update_msc', 'external_id'], axis=1, inplace=True)  
          print(f" Time trade {df['time'][0]} ")

      
          for position in positions:
              if position.comment == 'independent':
                  # Extract position details
                  ticket_x = position.ticket
                  position_magic = position.magic
                  open_position_x = position.symbol
                  volume_x = position.volume
                  price_current_x = position.price_current
                  stop_loss_x = position.sl
                  type_position_x = position.type  # 0 = BUY, 1 = SELL
                                    
              elif position.comment == 'dependent':
                  ticket_y = position.ticket
                  open_position_y = position.symbol
                  position_magic = position.magic
                  volume_y = position.volume
                  price_open_y = position.price_open
                  price_current_y = position.price_current
                  stop_loss_y = position.sl
                  type_position_y = position.type  # 0 = BUY, 1 = SELL
                  
                               
          if position_magic == MAGIC and ((open_position_x is None) ^ (open_position_y is None)):
                  net_profit_reached = True
                  
          print(f" Net profit reached {net_profit_reached}")
          if (open_position_y is not None):
              trailing_stop(open_position_y,type_position_y,price_current_y,stop_loss_y,ticket_y)
          if (open_position_x is not None):
              trailing_stop(open_position_x,type_position_x,price_current_x,stop_loss_x,ticket_x)
          
                  
          if not (open_position_x is None or open_position_y is None):        
                  
            #volume_adjust(open_position_y,open_position_x)
            #calculate_max_volume(open_position_x, type_position_x)
            adf_result = check_cointegration(open_position_y,open_position_x,0)

            z_scores,half_life,state_means,hedge_ratio = generate_regression(open_position_y,open_position_x,0)
            #plot_values(z_scores,state_means,half_life,open_position_y,open_position_x)        
      
  
            current_kalman = state_means[-1:][0][0]
            z_score = z_scores[periods-1]
            grid_entry = min_z_score + (grid_count)*additional_grid
            print(f"Current half life {half_life}")
            print(f"Hedge ratio {hedge_ratio}")
            print(f"z score is {z_score}")
            print(f"New z score entry is {grid_entry}")
            process_strategy(open_position_y,open_position_x,grid_entry,z_score,hedge_ratio)

      elif (total_positions == 0):
      
          symbolY,symbolX,slope,state_mean,zscore = verify_pairs(min_z_score,half_life_max)

          if (symbolY != None) and (symbolX != None):
              print(f"Pares selecionados {symbolY} dependente e {symbolX} independente ")
              process_strategy(symbolY,symbolX,min_z_score,zscore,slope)
          else:
              print("No pairs found ")
  else:
    close_all_positions()
  
  time.sleep(10)
  
# Shutdown MetaTrader 5 connection
mt5.shutdown()

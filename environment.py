# Modules
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from Trade import Trade
from TradeDatabase import TradeDatabase

# Global variables
START_DATE  = "2020-1-1"     # y-m-d-hr-min-sec
END_DATE    = "2024-1-1"     # y-m-d-hr-min-sec
INTERVAL    = "1d"
ASSETS      = 1000      # Current portfolio value in USD
FEE         = 0.006     # Max buy/sell fee at Coinbase
COMMISSION  = 0.1       # Commission on profits (made up)

# Class
class BTC:
    def __init__(self, 
                 start_date = START_DATE, 
                 end_date   = END_DATE,
                 interval   = INTERVAL,
                 assets     = ASSETS, 
                 fee        = FEE,
                 commission = COMMISSION):
        
        self.start_date = start_date
        self.end_date   = end_date
        self.interval   = interval
        self.t          = 0
        self.assets     = assets
        self.fee        = fee
        self.commission = commission

        self.btc_prices, self.dates = self.get_price_data()

        self.trade_history = TradeDatabase()

    def reset(self):
        self.__init__()

    def get_price_data(self):
        btc = yf.download("BTC-USD", 
                          start     = self.start_date, 
                          end       = self.end_date, 
                          interval  = self.interval)
        
        avg_prices = list(np.minimum(btc['Open'], btc['Close']) + np.abs(btc['Open'] - btc['Close'])/2)
        avg_prices = np.round(avg_prices, 2)

        dates = list(btc.index)
        dates = [time.date() for time in dates]

        return avg_prices, dates
    
    def get_price(self, t):
        return self.btc_prices[t]
    
    def buy(self, date):
        price = self.get_price(date)
        trade = Trade(price, self.fee, self.commission, date)

        trade.buy()
        
        if trade.buy_value > self.assets:
            print("ERROR: Cannot spend more than you have!")
            return -1
        elif self.trade_history:
            print("ERROR: Can only hold one trade at a time!")
            return -1
        else:
            self.trade_history.add_open_trade(trade)
            assets -= trade.cost
    
    def sell(self):
        if not self.trade_history:
            print("ERROR: You have no open trades to sell!")
            return -1
        else:
           reward = self.trade_history.close_trade(self.trade_history.open_trade.buy_date)

           return reward


    

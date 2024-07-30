# Modules
import numpy as np
import yfinance as yf

from Trade import Trade
from TradeDatabase import TradeDatabase

# Global variables
START_DATE  = "2020-1-1"     # y-m-d-hr-min-sec
END_DATE    = "2024-1-1"     # y-m-d-hr-min-sec
INTERVAL    = "1d"
CAPITAL     = 1000      # Current portfolio value in USD
FEE         = 0.006     # Max buy/sell fee at Coinbase
COMMISSION  = 0.1       # Commission on profits (made up)

# Class
class Environment:
    def __init__(self, 
                 start_date = START_DATE, 
                 end_date   = END_DATE,
                 interval   = INTERVAL,
                 capital     = CAPITAL, 
                 fee        = FEE,
                 commission = COMMISSION):
        
        self.start_date = start_date
        self.end_date   = end_date
        self.interval   = interval
        self.t          = 0
        self.capital     = capital
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
    
    def buy(self, t):
        price = self.get_price(t)
        trade = Trade(price, self.fee, self.commission, t)

        trade.buy()
        
        if trade.cost > self.capital:
            print("ERROR: Cannot spend more than you have!")
            return -1
        elif self.trade_history.open_trades:
            print("ERROR: Can only have one open trade at a time!")
            return -1
        else:
            self.trade_history.add_open_trade(trade)
            capital -= trade.cost
            return 0
    
    def sell(self):
        price = self.get_price(self.t)
        if not self.trade_history:
            print("ERROR: You have no open trades to sell!")
            return -1
        else:
           reward, trade_return = self.trade_history.close_trade(self.trade_history.open_trade,
                                                   price, self.t)
           self.capital += trade_return

           return reward
        
    def wait(self):
        return 0
    
    def update(self, action):
        self.t += 1
        reward = 0
        if action == 'buy':
            reward = self.buy(self.t)
        elif action == 'sell':
            reward = self.sell()
        elif action == 'wait':
            reward = self.wait()
        else:
            print("ERROR: An unknown action is sent!")
        
        if self.t == len(self.dates) - 1:
            pass
        if self.capital <= 0:
            pass

        return reward


    

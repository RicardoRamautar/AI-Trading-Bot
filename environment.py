# Modules
import numpy as np
import yfinance as yf

from Trade import Trade
# from TradeDatabase import TradeDatabase

# Global variables
START_DATE  = "2020-1-1"     # y-m-d-hr-min-sec
END_DATE    = "2024-1-1"     # y-m-d-hr-min-sec
INTERVAL    = "1d"
CAPITAL     = 20000     # Current portfolio value in USD
FEE         = 0.006     # Max buy/sell fee at Coinbase
COMMISSION  = 0.1       # Commission on profits (made up)

# Class
class Environment:
    def __init__(self, 
                 start_date = START_DATE, 
                 end_date   = END_DATE,
                 interval   = INTERVAL,
                 capital    = CAPITAL, 
                 fee        = FEE,
                 commission = COMMISSION):
        
        self.start_date = start_date
        self.end_date   = end_date
        self.interval   = interval
        self.t          = 0
        self.capital    = capital
        self.fee        = fee
        self.commission = commission

        self.open_trade     = None
        self.closed_trades  = {}

        self.prices, self.dates = self.get_price_data()

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
        return self.prices[t]

    def __add_open_trade(self, trade):
        self.open_trade = trade

    def __close_trade(self):
        price = self.get_price(self.t)
        reward, trade_return = self.open_trade.sell(price, self.t)
        self.closed_trades[self.open_trade.buy_date] = self.open_trade
        self.open_trade = None

        return reward, trade_return
    
    def __buy(self, amount):
        t = self.t
        price = self.get_price(t)
        trade = Trade(price, self.fee, self.commission, t)

        trade.buy(amount)
        
        if trade.cost > self.capital:
            print("ERROR: Cannot spend more than you have!")
            return -1
        elif self.open_trade:
            print("ERROR: Can only have one open trade at a time!")
            return -1
        else:
            self.__add_open_trade(trade)
            self.capital -= trade.cost
            return 0
    
    def __sell(self):
        # price = self.get_price(self.t)
        if not self.open_trade:
            print("ERROR: You have no open trades to sell!")
            return -1
        else:
           reward, trade_return = self.__close_trade()
           self.capital += trade_return

           return reward
        
    def __wait(self):
        return 0
    
    def update(self, action=0, amount=1):
        """0: Wait - 1: Buy - 2: Sell"""
        reward = 0
        if action == 1:
            reward = self.__buy(amount)
        elif action == 2:
            reward = self.__sell()
        elif action == 0:
            reward = self.__wait()
        else:
            print("ERROR: An unknown action is sent!")
        
        if self.t == len(self.dates) - 1:
            pass
        if self.capital <= 0:
            pass

        self.t += 1

        return reward


    

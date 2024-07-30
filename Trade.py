import numpy as np

class Trade:
    def __init__(self, price, fee, commission, date):
        self.buy_price  = price
        self.fee        = fee
        self.commission = commission
        self.buy_date   = date

        self.amount
        self.buy_value
        self.cost
        self.sell_date
        self.reward

    def buy(self, amount):
        self.amount     = amount
        self.buy_value  = self.buy_price * self.amount
        self.cost       = self.buy_value * (1 + self.fee)

    def sell(self, price, date):
        self.sell_date = date

        sell_value = self.amount * price
        trade_return = sell_value - max(sell_value - self.buy_value, 0) * self.commission
        self.reward = trade_return - self.cost
        return self.reward



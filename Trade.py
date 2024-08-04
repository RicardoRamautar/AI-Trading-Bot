class Trade:
    def __init__(self, price, fee, commission, date):
        self.buy_price  = price
        self.fee        = fee
        self.commission = commission
        self.buy_date   = date

        self.amount         = None
        self.buy_value      = None
        self.cost           = None
        self.sell_date      = None
        self.trade_return   = None
        self.reward         = None

    def buy(self, amount):
        self.amount     = amount
        self.buy_value  = self.buy_price * self.amount
        self.cost       = self.buy_value * (1 + self.fee)
        print(self.cost)

    def sell(self, price, t):
        self.sell_date = t

        sell_value = self.amount * price
        self.trade_return = sell_value - max(sell_value - self.buy_value, 0) * self.commission

        self.reward = self.trade_return - self.cost

        return self.reward, self.trade_return



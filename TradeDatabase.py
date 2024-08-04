from Trade import Trade

class TradeDatabase:
    def __init__(self):
        self.open_trades    = None
        self.closed_trades  = {}

    def add_open_trade(self, trade):
        self.open_trades = trade

    def close_trade(self, trade, price, date):
        # self.open_trade = None
        trade.sell(price, date)
        self.closed_trades[trade.buy_date] = trade

        return trade.reward
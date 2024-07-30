from Trade import Trade

class TradeDatabase:
    def __init__(self):
        self.open_trades    = None
        self.closed_trades  = {}

    def add_open_trade(self, trade):
        self.open_trade = trade

    def close_trade(self, trade):
        self.open_trade = None
        self.closed_trades[trade.buy_date] = trade
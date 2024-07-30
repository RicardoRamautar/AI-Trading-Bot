from Trade import Trade

class TradeDatabase:
    def __init__(self):
        self.open_trades    = {}
        self.closed_trades  = {}

    def add_open_trade(self, trade):
        self.open_trades[trade.buy_date] = trade

    def add_closed_trade(self, trade):
        del self.open_trades[trade.buy_date]
        self.closed_trades[trade.buy_date] = trade
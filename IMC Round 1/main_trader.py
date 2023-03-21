from typing import Dict, List, Tuple
from datamodel import OrderDepth, TradingState, Order, Symbol, Trade, Product, Position
import json


class Trader:
    def __init__(self) -> None:
        self.cash = 0
        self.LAST_ROUND = 99900
        # comment out a category to disable it in the logs
        self.logger = Logger([
            "format",
            "profit",
            "final_profit",
            "cash",
            "error",
            "debug",
            "mid_price",
            "orders",
            "default",
        ])

    def calculate_cash(self, own_trades: Dict[Symbol, List[Trade]], now) -> None:
        for product in own_trades:
            # trades of a product
            trades = own_trades[product]
            for trade in trades:
                # only consider trades from last iteration
                if trade.buyer == "SUBMISSION" and trade.timestamp == now-100:
                    self.logger.log_buy(trade)
                    self.cash -= trade.price * trade.quantity
                elif trade.seller == "SUBMISSION" and trade.timestamp == now-100:
                    self.logger.log_sell(trade)
                    self.cash += trade.price * trade.quantity

    def calculate_last_round_profit(self, position: Dict[Product, Position], mid_prices: Dict[Symbol, Position]):
        """what the profit would be if it is last round"""
        profit = self.cash
        for product in position:
            mid_price = mid_prices[product]
            self.logger.log(
                f"{product}'s position = {position[product]}, with mid_price = {mid_price}",
                "debug")
            profit += position[product] * mid_price
        return profit

    def calculate_mid_prices(self, order_depths):
        """returns 3 dictionaries"""
        mid_prices = {}
        best_bids = {}
        best_asks = {}
        for product in order_depths:
            order_depth = order_depths[product]
            if (order_depth.buy_orders == {} or order_depth.sell_orders == {}):
                self.logger.log_error(
                    f"buy order or sell order of {product}'s order depth is empty")
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            self.logger.log(
                f"{product} -- best bid: {best_bid}, best_ask: {best_ask}, mid_price: {mid_price}",
                "mid_price")
            mid_prices[product] = mid_price
            best_bids[product] = best_bid
            best_asks[product] = best_ask
        return mid_prices, best_bids, best_asks

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """

        mid_prices, best_bids, best_asks = self.calculate_mid_prices(
            state.order_depths)
        self.logger.log(mid_prices, "mid_price")

        # profit calculation
        self.calculate_cash(state.own_trades, state.timestamp)
        self.logger.log(f"out position now is {state.position}", "debug")
        self.logger.log(f"cash now is {self.cash}", "cash")

        profit = self.calculate_last_round_profit(state.position, mid_prices)
        self.logger.log(f"if it is last round, profit = {profit}", "profit")
        if state.timestamp == self.LAST_ROUND:
            self.logger.log(f"final profit = {profit}", "final_profit")

        print(json.dumps(state.toJSON(), indent=4, sort_keys=True))
        result = {}

        banana_orders = self.market_making(
            "BANANAS", 20, best_bids["BANANAS"], best_asks["BANANAS"], state.position)
        result['BANANAS'] = banana_orders
        self.logger.log(
            f"Making banana orders by market making:\n{banana_orders}", "orders")

        self.logger.divider_big()
        return result

    def market_making(self, product: str, pos_lim: int,
                      best_bid: int, best_ask: int,
                      position: Dict[Product, Position], history: list = [],
                      WAP: float = 0, sigma: float = 0):
        """
        Given the current orderbook and trade history, calculate the new ask/bid price and volume.

        Input arguments:
        product -- the product traded on
        best_bid -- the current best bid price
        best_ask -- the current best ask price
        pos -- the current position
        history -- list of mid prices of the last 10 timestamps
        WAP -- the current weighted average price
        sigma -- the volatility (standard deviation)

        Return values:
        orders --  a list of all orders
        """
        self.logger.log(f"{position}")
        pos = position[product] if product in position else 0

        market_spread = (best_ask - best_bid)/2  # note is may be a float
        mid_price = (best_ask + best_bid) / 2  # note this may be a float

        lot_size = 5  # the quantity of our orders
        delta = 0.8  # delta defines the ratio of our bid-ask spread to the gross market spread

        orders = []
        mm_bid_price = round(mid_price - delta * market_spread)
        mm_bid_vol = min(lot_size, pos_lim-pos)
        mm_ask_price = round(mid_price + delta * market_spread)
        mm_ask_vol = -min(lot_size, pos+pos_lim)
        orders.append(Order(product, mm_bid_price, mm_bid_vol))
        orders.append(Order(product, mm_ask_price, mm_ask_vol))

        return orders


class Logger:
    """
    ["default","profit","error","debug"]
    """

    def __init__(self, categories: List[str], print_all=False):
        self.categories = categories
        self.print_all = print_all

    def log(self, message, category="default"):
        if self.print_all or category in self.categories:
            print(f"[{category.upper()}] {message}")

    def log_error(self, message):
        self.log(message, "error")

    def log_buy(self, trade: Trade):
        self.log(
            f"we bought {trade.symbol} x {trade.quantity} for {trade.price}, cash - {trade.quantity * trade.price}.", "debug")

    def log_sell(self, trade: Trade):
        self.log(
            f"we sold {trade.symbol} x {trade.quantity} for {trade.price}, cash + {trade.quantity * trade.price}.", "debug")

    def divider_big(self):
        if "format" in self.categories:
            print("="*100)

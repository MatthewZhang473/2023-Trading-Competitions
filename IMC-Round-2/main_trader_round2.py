from typing import Dict, List, Tuple
from datamodel import OrderDepth, TradingState, Order, Symbol, Trade, Product, Position
import json
import numpy as np


class Trader:
    def __init__(self) -> None:
        self.cash = 0
        self.window = Window(10)
        self.LAST_ROUND = 99900
        self.position_limits = {
            "BANANAS": 20,
            "PEARLS": 20,
            "COCONUTS": 600,
            "PINA_COLADAS": 300
        }
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

        # mid price calculation
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

        self.logger.log(json.dumps(
            state.toJSON(), indent=4, sort_keys=True), "debug")

        ### TRADING ALGORITHM ###

        result = {}

        ### BANANAS TRADING ###
        BANANA = "BANANAS"
        banana_orders = self.window_calc(BANANA, self.window, mid_prices[BANANA],
                                         best_bids[BANANA], best_asks[BANANA],
                                         state.position[BANANA] if BANANA in state.position else 0,
                                         self.position_limits[BANANA])
        result[BANANA] = banana_orders
        self.logger.log(
            f"Making banana orders by sliding window:\n{banana_orders}", "orders")
        self.window.push(mid_prices[BANANA])

        # banana_orders = self.market_making_calc(
        #     BANANA, 20, best_bids[BANANA], best_asks[BANANA], state.position)
        # result[BANANA] = banana_orders
        # self.logger.log(
        #     f"Making banana orders by market making:\n{banana_orders}", "orders")

        ### PEARLS TRADING ###
        PEARL = "PEARLS"
        pearl_orders = self.mean_reversal_calc(
            PEARL, 20, best_bids[PEARL], best_asks[PEARL], state.position)
        result[PEARL] = pearl_orders
        self.logger.log(
            f"Making pearl orders by mena reversal:\n{pearl_orders}", "orders")
        
        ## COCONUT AND PINA COLADAS TRADING ###
        COCONUTS = "COCONUTS"
        PINA_COLADAS = "PINA_COLADAS"
        COCONUTS_OVER_PINAS_VALUE_RATIO = 8/15
        COCONUTS_LOT_SIZE = 15
        PINA_COLADAS_LOT_SIZE = 8
        COCONUTS_best_bid_price = best_bids[COCONUTS]
        COCONUTS_best_bid_volume = state.order_depths[COCONUTS].buy_orders[COCONUTS_best_bid_price]
        COCONUTS_best_ask_price = best_asks[COCONUTS]
        COCONUTS_best_ask_volume = abs(state.order_depths[COCONUTS].sell_orders[COCONUTS_best_ask_price])
        PINAS_best_bid_price = best_bids[PINA_COLADAS]
        PINAS_best_bid_volume = state.order_depths[PINA_COLADAS].buy_orders[PINAS_best_bid_price]
        PINAS_best_ask_price = best_asks[PINA_COLADAS]
        PINAS_best_ask_volume = abs(state.order_depths[PINA_COLADAS].sell_orders[PINAS_best_ask_price])
        arbitrage_COCONUTS_orders, arbitrage_PINAS_orders =\
            self.arbitrage_calc(product_1=COCONUTS, product_2=PINA_COLADAS,
                                best_bid_price_product_1=COCONUTS_best_bid_price, 
                                best_ask_price_product_1=COCONUTS_best_ask_price,
                                best_bid_volume_product_1=COCONUTS_best_bid_volume,
                                best_ask_volume_product_1=COCONUTS_best_ask_volume,
                                best_bid_price_product_2=PINAS_best_bid_price,
                                best_ask_price_product_2=PINAS_best_ask_price,
                                best_bid_volume_product_2=PINAS_best_bid_volume,
                                best_ask_volume_product_2=PINAS_best_ask_volume,
                                position_product_1=state.position[COCONUTS] if COCONUTS in state.position else 0,
                                position_product_2=state.position[PINA_COLADAS] if PINA_COLADAS in state.position else 0,
                                product_1_over_product_2_value_ratio=COCONUTS_OVER_PINAS_VALUE_RATIO,
                                lot_size_product_1=COCONUTS_LOT_SIZE,
                                lot_size_product_2=PINA_COLADAS_LOT_SIZE,
                                position_limit_product_1=self.position_limits[COCONUTS],
                                position_limit_product_2=self.position_limits[PINA_COLADAS])
        result[COCONUTS] = arbitrage_COCONUTS_orders
        result[PINA_COLADAS] = arbitrage_PINAS_orders
        self.logger.log(
            f"Making COCONUTS orders by arbitrage:\n{arbitrage_COCONUTS_orders}", "orders")
        self.logger.log(
            f"Making PINAS orders by arbitrage:\n{arbitrage_PINAS_orders}", "orders")


        ### RETURN RESULT ###
        self.logger.divider_big()
        return result

    def market_making_calc(self, product: str, pos_lim: int,
                           best_bid: int, best_ask: int,
                           position: Dict[Product, int], history: list = [],
                           WAP: float = 0, sigma: float = 0) -> List[Order]:
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
        pos = position[product] if product in position else 0

        market_spread = (best_ask - best_bid)/2  # note is may be a float
        mid_price = (best_ask + best_bid) / 2  # note this may be a float

        ## Cancel the trade if the market spread is too small ##
        if market_spread * 2 < 3:
            return []

        # #########################################################

        # ## Sigmoid Inventory Control##
        # alpha = 1 * market_spread
        # offset = 10
        # scale = 1.5
        # if pos > 0:
        #     normalized_pos = (pos - offset)/scale
        #     mid_price -= round(alpha * self.sigmoid(normalized_pos))
        # if pos < 0:
        #     normalized_pos = -(pos + offset)/scale
        #     mid_price += round(alpha * self.sigmoid(normalized_pos))
        # ###############################

        lot_size = 5  # the quantity of our orders
        # delta = 0.8  # delta defines the ratio of our bid-ask spread to the gross market spread
        orders = []

        mm_bid_price = best_bid + 1
        mm_bid_vol = min(lot_size, pos_lim-pos)
        mm_ask_price = best_ask - 1
        mm_ask_vol = -min(lot_size, pos+pos_lim)

        if pos > 2 * lot_size:
            mm_ask_price -= 1
            mm_ask_vol *= 2
        if pos < -2 * lot_size:
            mm_bid_price += 1
            mm_bid_vol *= 2

        orders.append(Order(product, mm_ask_price, mm_ask_vol))  # sell
        orders.append(Order(product, mm_bid_price, mm_bid_vol))  # buy
        return orders

    def sigmoid(self, x: float):
        """
        Sigmoid function
        """
        y = 1 / (1+np.exp(-x))
        return y

    def mean_reversal_calc(self, product: str, pos_lim: int,
                           best_bid: int, best_ask: int,
                           position: Dict[Product, int], history: list = [],
                           WAP: float = 0, sigma: float = 0) -> List[Order]:
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
        pos = position[product] if product in position else 0

        true_price = 10000
        lot_size = 15
        orders = []
        min_profit_margin = 2  # minimum acceptable price
        alpha = 1  # alpha is a parameter that scale the bid/ask volume
        margin = 0

        mr_ask_price = 0  # mr for mean reversal
        mr_bid_price = 0

        # Buy / Sell, with volume proportional to the profit margin
        buy_profit_margin = true_price - best_ask
        sell_profit_margin = best_bid - true_price
        self.logger.log(f"buy profit margin is {buy_profit_margin}", "debug")
        self.logger.log(f"sell profit margin is {sell_profit_margin}", "debug")
        if buy_profit_margin >= min_profit_margin:
            if buy_profit_margin >= 2:
                margin = 1
            mr_ask_price = best_ask + margin
            mr_ask_vol = min(alpha*buy_profit_margin*lot_size, pos_lim-pos)
        if sell_profit_margin >= min_profit_margin:
            if sell_profit_margin >= 2:
                margin = 1
            mr_bid_price = best_bid - margin
            mr_bid_vol = -min(alpha*sell_profit_margin*lot_size, pos+pos_lim)

        # send the orders
        if mr_bid_price != 0:
            # sell
            orders.append(Order(product, mr_bid_price, mr_bid_vol))
        if mr_ask_price != 0:
            # buy
            orders.append(Order(product, mr_ask_price, mr_ask_vol))

        return orders

    def window_calc(self, product: Product, window, mid_price: float,
                    best_bid, best_ask,
                    position: float, position_limit: float) -> List[Order]:
        orders = []

        margin = 1
        clear_limit = 20  # gradually clear
        force_clear = 20
        adaptive_multiplier = 2
        n = 0.7  # first bound
        m = 2  # second bound

        self.logger.log(
            f"margin = {margin}, clear_limit={clear_limit}, force_clear={force_clear}, n={n}", "debug")

        upper, lower = window.upper_lower_bounds(n)
        upper2, lower2 = window.upper_lower_bounds(m)
        self.logger.log(f"upper: {upper}, lower: {lower}")

        # has consecutive spikes prevention
        # if mid price is above upper bound, sell
        if mid_price > upper and abs(position) < force_clear:
            if mid_price > upper2:
                margin *= adaptive_multiplier
            max_sell = -(position + position_limit)
            orders.append(Order(product, best_ask - margin, max_sell))

        # if mid price is below lower bound, buy
        elif mid_price < lower and abs(position) < force_clear:
            if mid_price < lower2:
                margin *= adaptive_multiplier
            max_buy = position_limit - position  # sign accounted for
            orders.append(Order(product, best_bid + margin, max_buy))
        # do normal market making if there are no spikes
        # elif abs(position) < lot_size:
        #     if (best_bid-best_ask) < 3:
        #         return []
        #     mm_bid_price = best_bid + 1
        #     mm_bid_vol = min(lot_size, position_limit-position)
        #     mm_ask_price = best_ask - 1
        #     mm_ask_vol = -min(lot_size, position+position_limit)
        #     if position > 2 * lot_size:
        #         mm_ask_price -= 1
        #         mm_ask_vol *= 2
        #     if position < -2 * lot_size:
        #         mm_bid_price += 1
        #         mm_bid_vol *= 2
        #     orders.append(Order(product, mm_bid_price, mm_bid_vol))  # buy
        #     orders.append(Order(product, mm_ask_price, mm_ask_vol))  # sell
        # else just clear position gradually
        else:
            # price = mid_price - margin if clear_amount < 0 else mid_price + margin
            price = best_ask - margin if position > 0 else best_bid + margin
            clear_amount = min(-position, clear_limit) if position < 0 else - \
                min(position, clear_limit)
            orders.append(Order(product, price, clear_amount))

        return orders

    def arbitrage_calc(self, product_1: Product, product_2: Product,
                       best_bid_price_product_1: int, best_ask_price_product_1: int,
                       best_bid_volume_product_1:int, best_ask_volume_product_1: int,
                       best_bid_price_product_2: int, best_ask_price_product_2: int,
                       best_bid_volume_product_2:int, best_ask_volume_product_2: int,
                       position_product_1: int, position_product_2: int,
                       product_1_over_product_2_value_ratio: float,
                       lot_size_product_1: int, lot_size_product_2: int,
                       position_limit_product_1: int, position_limit_product_2: int) -> List[Order]:
        """
        Arbitrage between two related products.

        Input Arguments:
        product_1 -- the main product we trade on
        product_2 -- the related product we hedge (take reverse trades on) 
                         to reduce risk
        product_1_over_product_2_value_ratio -- the ratio of prices between two related products
        lot_size_product -- 15 for COCONAS (price at 8k), 8 for PINA (price at 15k)
        """
        
        product_1_orders = []
        product_2_orders = []

        buy_volume_product_1 = 0
        buy_volume_product_2 = 0
        sell_volume_product_1 = 0
        sell_volume_product_2 = 0

        # if normalized product 2 is cheaper than product 1
        if best_bid_price_product_1 > best_ask_price_product_2 * product_1_over_product_2_value_ratio:
            # for now we only buy / sell at the best bid and best ask prices
            # buy in product 2, sell product 1

            # how many lots of product 2 can I buy from the order book, without breaching position limit
            max_number_buy_lots = min(best_ask_volume_product_2,\
                                      position_limit_product_2 - position_product_2)//lot_size_product_2

            # how many lots of product 1 can I sell from the order book, without breaching position limit
            max_number_sell_lots = min(best_bid_volume_product_1,\
                                       position_limit_product_1 - position_product_1)//lot_size_product_1
            
            number_trade_lots = min(max_number_buy_lots, max_number_sell_lots) # the number of lots I can trade, this is the smaller of the two
            
            buy_volume_product_2 = number_trade_lots * lot_size_product_2
            sell_volume_product_1 = number_trade_lots * lot_size_product_1
            buy_price_product_2 = best_ask_price_product_2
            sell_price_product_1 = best_bid_price_product_1
    
        # if normalized product 1 is cheaper than prodcut 2
        if best_bid_price_product_2 * product_1_over_product_2_value_ratio > best_ask_price_product_1:
            # buy in product 1, sell product 2

            # how many lots of product 1 can I buy from the order book, without breaching position limit
            max_number_buy_lots = min(best_ask_volume_product_1,\
                                      position_limit_product_1 - position_product_1)// lot_size_product_1
            
            # how many lots of product 2 can I sell from the order book, without breaching position limit
            max_number_sell_lots = min(best_bid_volume_product_2,\
                                    position_limit_product_2 + position_product_2)//lot_size_product_2
            number_trade_lots = min(max_number_buy_lots, max_number_sell_lots) # the number of lots I can trade, this is the smaller of the two
            
            buy_volume_product_1 = number_trade_lots * lot_size_product_1
            sell_volume_product_2 = number_trade_lots * lot_size_product_2
            buy_price_product_1 = best_ask_price_product_1
            sell_price_product_2 = best_bid_price_product_2

        # Send orders
        if sell_volume_product_1!=0 and buy_volume_product_2!=0:
            # if we can sell product 1 and buy product 2
            
            # N.B. I assumed all the volumes given are positive
            product_1_orders.append(Order(product_1, sell_price_product_1, -sell_volume_product_1))
            product_2_orders.append(Order(product_2, buy_price_product_2, buy_volume_product_2))

        if sell_volume_product_2!=0 and buy_volume_product_1!=0:
            # if we can sell product 2 and buy product 1
            product_2_orders.append(Order(product_2, sell_price_product_2, -sell_volume_product_2))
            product_1_orders.append(Order(product_1, buy_price_product_1, buy_volume_product_1))

        return (product_1_orders,product_2_orders)
        

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


class Window:
    def __init__(self, size) -> None:
        self.size = size
        self.contents = []

    def push(self, item) -> None:
        self.contents.append(item)
        if (len(self.contents) > self.size):
            self.contents.pop(0)

    def avg(self) -> float:
        return float(sum(self.contents)/len(self.contents) if len(self.contents) > 0 else 0)

    def std(self) -> float:
        return float(np.std(self.contents))

    def upper_lower_bounds(self, n=2) -> Tuple[float, float]:
        return (self.avg()+n*self.std(), self.avg()-n*self.std())

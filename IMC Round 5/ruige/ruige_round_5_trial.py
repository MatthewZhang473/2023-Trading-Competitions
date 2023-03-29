from typing import Dict, List, Tuple
from datamodel import OrderDepth, TradingState, Order, Symbol, Trade, Product, Position, Listing
import json
import numpy as np
import pandas as pd


class Trader:
    def __init__(self) -> None:

        self.products = ["BANANAS",
            "PEARLS",
            "COCONUTS",
            "PINA_COLADAS",
            'DIVING_GEAR',
            "BERRIES",
            "BAGUETTE",
            "DIP",
            "UKULELE",
            "PICNIC_BASKET"]

        # {product : negative profit threshold to stop trading}
        self.profit_threshold = {
            "BANANAS": -10000,
            "PEARLS": -10000,
            "COCONUTS": -7500,
            "PINA_COLADAS": -7500,
            'DIVING_GEAR': -10000,
            "BERRIES": -10000,
            "BAGUETTE": -5000,
            "DIP": -5000,
            "UKULELE": -5000,
            "PICNIC_BASKET": -5000,
        }

        # init cash
        self.cash = {}
        for product in self.profit_threshold.keys():
            self.cash[product] = 0

        self.window = Window(10)
        self.mayberry_window = Window(1000)

        self.mid_window_smaller = Window(35)
        self.mid_window_larger = Window(45)
        self.window1_larger = None
        self.last_peak_or_valley = 0
        self.mid_diff_last_trade = 0

        self.mid_window = Window(35)
        self.cb_pa_window = Window(35)
        self.pb_ca_window = Window(35)
        self.coconut_window = Window(10)

        self.sell = 0
        self.buy = 0

        self.last_sell_price = 0
        self.last_bought_price = 1000000
        self.LAST_ROUND = 99900

        ###### Ruige & Matthew's Code for DIVING_GEAR #######
        # Dolphin window definitions
        dolphin_diff_term = 1
        dolphin_num_past_terms = 10
        self.dolphin_window_size = dolphin_num_past_terms + dolphin_diff_term + 1
        self.dolphin_diff_term = dolphin_diff_term
        self.dolphin_window = Window(self.dolphin_window_size)

        # Diving gear window definitions
        gear_diff_term = 1
        gear_num_past_terms = 800
        self.gear_window_size = gear_num_past_terms + gear_diff_term + 1
        self.gear_diff_term = gear_diff_term
        self.gear_window = Window(self.gear_window_size)

        # tracers
        # entrance tracer uses mean values of past 10 DOLPHIN_SIGHTINGS differences
        entrance_tracer_size = 30
        self.entrance_tracer_window = Window(entrance_tracer_size)

        # exit tracer
        exit_tracer_window_size = 50
        self.exit_tracer_window = Window(exit_tracer_window_size)

        # gear trade signal
        self.gear_buy_flag = False
        self.gear_sell_flag = False
        # berries
        self.start_done = False
        self.peak_done = False
        self.end_done = False
        self.good_clear_price = 0

        # Windows for new arbitrager for coconuts / pinas
        self.new_arbitrage_long_term_mid_diff_prices = Window(500)
        self.new_arbitrage_zero_counter_window = Window(100)
        self.new_arbitrage_short_term_mid_price_window = Window(100)
        self.new_arbitrage_short_term_flag_duration_counter = 0
        self.new_arbitrage_buy_to_limit_signal = False
        self.new_arbitrage_sell_to_limit_signal = False
        self.new_arbitrage_short_term_flag = True
        self.new_arbitrage_long_term_flag = False

        ### PICNIC_BASKET WINDOW ###
        self.difference_between_main_product_and_components_window = Window(50)
        self.clearing_flag = False

        self.position_limits = {
            "BANANAS": 20,
            "PEARLS": 20,
            "COCONUTS": 600,
            "PINA_COLADAS": 300,
            'DIVING_GEAR': 50,
            "BERRIES": 250,
            "BAGUETTE": 150,
            "DIP": 300,
            "UKULELE": 70,
            "PICNIC_BASKET": 70,
        }

        # comment out a category to disable it in the logs
        self.logger = Logger([
            "format",
            "important",
            "timestamp",
            "bananas",
            "position",
            "profit",
            "final_profit",
            "cash",
            "error",
            "debug",
            "mid_price",
            "orders",
            "default",
        ])

        self.traders = ['Peter', 
                        'Gary', 
                        'Mitch', 
                        'Penelope', 
                        'Omar', 
                        'Camilla', 
                        'Caesar', 
                        'Giulia', 
                        'Mabel',
                        'Pablo', 
                        'Charlie',
                        'Olivia',
                        'Orson',
                        'Casey',
                        'George',
                        'Mya',
                        'Max',
                        'Paris',
                        'Gina',
                        'Olga'
                        ]
        self.olivia_buy_flag = False
        self.olivia_sell_flag = False
        self.paris_buy_flag = False
        self.paris_sell_flag = False

    def calculate_cash(self, own_trades: Dict[Symbol, List[Trade]], now) -> None:
        for product in own_trades:
            # trades of a product
            trades = own_trades[product]
            for trade in trades:
                # only consider trades from last iteration
                if trade.buyer == "SUBMISSION" and trade.timestamp == now-100:
                    self.logger.log_buy(trade)
                    self.cash[product] -= trade.price * trade.quantity
                elif trade.seller == "SUBMISSION" and trade.timestamp == now-100:
                    self.logger.log_sell(trade)
                    self.cash[product] += trade.price * trade.quantity
        for product in self.cash.keys():
            self.logger.log(f"{product} -- {self.cash[product]}", "cash")

    def calculate_position(self, listing: Dict[Symbol, Listing], position: Dict[Product, Position]) -> Dict[Product, Position]:
        """add missing product with 0 to the position dictionary"""
        for product in listing.keys():
            if product not in position.keys():
                position[product] = 0
        for product in listing.keys():
            self.logger.log(
                f"{product}'s position = {position[product]}", "position")
        return position

    def calculate_last_round_profit(self, position: Dict[Product, Position], mid_prices: Dict[Symbol, Position]):
        """what the profit would be if it is last round"""
        profits = self.cash.copy()
        for product in position:
            if product in mid_prices:
                mid_price = mid_prices[product]
                profits[product] += position[product] * mid_price
                self.logger.log(f"{product} -- {profits[product]}", "profit")
        return profits

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
        print("__")
        self.logger.log(state.timestamp, "timestamp")
        self.logger.log(f'the current market trades are {state.market_trades}', 'debug')

        # mid price calculation
        mid_prices, best_bids, best_asks = self.calculate_mid_prices(
            state.order_depths)

        # calculate and print positions
        positions = self.calculate_position(state.listings, state.position)

        # profit calculation
        self.calculate_cash(state.own_trades, state.timestamp)
        self.logger.log(f"cash now is {self.cash}", "cash")

        profits = self.calculate_last_round_profit(state.position, mid_prices)
        if state.timestamp == self.LAST_ROUND:
            self.logger.log(f"final profit = {profits}", "final_profit")

        # self.logger.log(state.toJSON(), "debug")


        ### Other traders ###
        market_trades = state.market_trades # this returns a a dictionary, with keys being the products and values being lists of trade objects for that product
        
        


        









        ### TRADING ALGORITHM ###

        result = {}

        ### BANANAS TRADING ###
        BANANA = "BANANAS"
        if profits[BANANA] > self.profit_threshold[BANANA]:
            banana_orders = self.window_calc(BANANA, self.window, mid_prices[BANANA],
                                             best_bids[BANANA], best_asks[BANANA],
                                             state.position[BANANA] if BANANA in state.position else 0,
                                             self.position_limits[BANANA])
            result[BANANA] = banana_orders
            self.logger.log(
                f"BANANAS window mm: {banana_orders}", "orders")
            self.window.push(mid_prices[BANANA])
        else:
            self.logger.log(
                f"{BANANA} stop trading because of profit {profits[BANANA]} below {self.profit_threshold[BANANA]}", "important")

        ### PEARLS TRADING ###
        PEARL = "PEARLS"
        if profits[PEARL] > self.profit_threshold[PEARL]:
            pearl_orders = self.mean_reversal_calc(
                PEARL, 20, best_bids[PEARL], best_asks[PEARL], state.position)
            result[PEARL] = pearl_orders
            self.logger.log(
                f"PEARLS mean reversal: {pearl_orders}", "orders")
        else:
            self.logger.log(
                f"{PEARL} stop trading because of profit {profits[PEARL]} below {self.profit_threshold[PEARL]}", "important")

        ## COCONUT AND PINA COLADAS TRADING ###
        COCONUTS = "COCONUTS"
        PINA_COLADAS = "PINA_COLADAS"
        if profits[COCONUTS] + profits[PINA_COLADAS] > self.profit_threshold[COCONUTS] + self.profit_threshold[PINA_COLADAS]:
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
                self.arbitrage_calc(timestamp=state.timestamp,
                                    product_1=COCONUTS, product_2=PINA_COLADAS,
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
        else:
            self.logger.log(
                f"{COCONUTS} stop trading because of profit {profits[COCONUTS]} below {self.profit_threshold[COCONUTS]} OR", "important")
            self.logger.log(
                f"{PINA_COLADAS} stop trading because of profit {profits[PINA_COLADAS]} below {self.profit_threshold[PINA_COLADAS]}", "important")

        ### DIVING_GEAR trade ###
        DIVING_GEAR = 'DIVING_GEAR'
        if profits[DIVING_GEAR] > self.profit_threshold[DIVING_GEAR]:
            indicator = 'DOLPHIN_SIGHTINGS'
            indicator_mid_price = state.observations[indicator]

            DIVING_GEAR_best_bid_price = best_bids[DIVING_GEAR]
            DIVING_GEAR_best_bid_volume = state.order_depths[
                DIVING_GEAR].buy_orders[DIVING_GEAR_best_bid_price]
            DIVING_GEAR_best_ask_price = best_asks[DIVING_GEAR]
            DIVING_GEAR_best_ask_volume = abs(
                state.order_depths[DIVING_GEAR].sell_orders[DIVING_GEAR_best_ask_price])
            DIVING_GEAR_position = positions[DIVING_GEAR]
            DIVING_GEAR_position_limit = self.position_limits[DIVING_GEAR]

            diving_gear_orders = self.indicator_trade(state.timestamp, DIVING_GEAR, indicator_mid_price, DIVING_GEAR_best_bid_price,
                                                      DIVING_GEAR_best_bid_volume, DIVING_GEAR_best_ask_price, DIVING_GEAR_best_ask_volume,
                                                      DIVING_GEAR_position, DIVING_GEAR_position_limit, self.dolphin_diff_term, self.gear_diff_term)
            result[DIVING_GEAR] = diving_gear_orders
            self.logger.log(
                f'{DIVING_GEAR} window indicator trade: {diving_gear_orders}', 'orders')
        else:
            self.logger.log(
                f"{DIVING_GEAR} stop trading because of profit {profits[DIVING_GEAR]} below {self.profit_threshold[DIVING_GEAR]} OR", "important")

        ### BERRIES ###
        # Key information: OLIVIA
        BERRIES = 'BERRIES'
        OLIVIA = 'Olivia'
        for trade in market_trades[BERRIES]: 
            self.logger.log(f'the buyer is {trade.buyer}','debug')
            self.logger.log(f'the seller is {trade.seller}','debug')
            if trade.buyer == OLIVIA:
                self.olivia_buy_flag = True
                self.logger.log(f'{OLIVIA} buys, triggering {BERRIES} buy flag to {self.olivia_buy_flag}', 'debug')
            elif trade.seller == OLIVIA:
                self.olivia_sell_flag = True
                self.logger.log(f'{OLIVIA} sells, triggering {BERRIES} sell flag to {self.olivia_sell_flag}', 'debug')
        
        # testing whether the flags are raised
        PEARLS = 'PEARLS'
        PARIS = 'Paris'
        for trade in market_trades[PEARLS]: 
            self.logger.log(f'the buyer is {trade.buyer}','debug')
            self.logger.log(f'the seller is {trade.seller}','debug')
            if trade.buyer == PARIS:
                self.paris_buy_flag = True
                self.logger.log(f'{PARIS} buys, triggering {PEARLS} buy flag to {self.paris_buy_flag}', 'debug')
            elif trade.seller == PARIS:
                self.paris_sell_flag = True
                self.logger.log(f'{PARIS} sells, triggering {PEARLS} sell flag to {self.paris_sell_flag}', 'debug')


        BERRIES = "BERRIES"
        if profits[BERRIES] > self.profit_threshold[BERRIES]:
            berries_orders = self.mayberry_calc_olivia(timestamp=state.timestamp, product=BERRIES, window=self.mayberry_window,
                                                mid_price=mid_prices[BERRIES],
                                                best_bid=best_bids[BERRIES], best_ask=best_asks[BERRIES],
                                                best_bid_volume=state.order_depths[
                                                    BERRIES].buy_orders[best_bids[BERRIES]],
                                                best_ask_volume=abs(
                                                    state.order_depths[BERRIES].sell_orders[best_asks[BERRIES]]),
                                                position_limit=self.position_limits[BERRIES], position=positions[BERRIES])
            result[BERRIES] = berries_orders
            self.mayberry_window.push(mid_prices[BERRIES])
            self.logger.log(
                f"BERRIES: {berries_orders}", "orders")
        else:
            self.logger.log(
                f"{BERRIES} stop trading because of profit {profits[BERRIES]} below {self.profit_threshold[BERRIES]}", "important")
        # if profits[BERRIES] > self.profit_threshold[BERRIES]:
        #     berries_orders = self.mayberry_calc(timestamp=state.timestamp, product=BERRIES, window=self.mayberry_window,
        #                                         mid_price=mid_prices[BERRIES],
        #                                         best_bid=best_bids[BERRIES], best_ask=best_asks[BERRIES],
        #                                         best_bid_volume=state.order_depths[
        #                                             BERRIES].buy_orders[best_bids[BERRIES]],
        #                                         best_ask_volume=abs(
        #                                             state.order_depths[BERRIES].sell_orders[best_asks[BERRIES]]),
        #                                         position_limit=self.position_limits[BERRIES], position=positions[BERRIES])
        #     result[BERRIES] = berries_orders
        #     self.mayberry_window.push(mid_prices[BERRIES])
        #     self.logger.log(
        #         f"BERRIES: {berries_orders}", "orders")
        # else:
        #     self.logger.log(
        #         f"{BERRIES} stop trading because of profit {profits[BERRIES]} below {self.profit_threshold[BERRIES]}", "important")

        ### PICNIC BASKETS ###
        PICNIC_BASKET = 'PICNIC_BASKET'
        BAGUETTE = 'BAGUETTE'
        DIP = 'DIP'
        UKULELE = 'UKULELE'

        if profits[PICNIC_BASKET] + profits[BAGUETTE] + profits[DIP] + profits[UKULELE] > self.profit_threshold[PICNIC_BASKET] + self.profit_threshold[BAGUETTE] + self.profit_threshold[DIP] + self.profit_threshold[UKULELE]:
            # preprocessing for components
            components = [BAGUETTE, DIP, UKULELE]
            best_bid_prices = [best_bids[i] for i in components]
            best_ask_prices = [best_asks[i] for i in components]
            components_best_bid_prices = dict(zip(components, best_bid_prices))
            components_best_ask_prices = dict(zip(components, best_ask_prices))
            components_positions = {component:positions[component] for component in components}
            components_position_limits = {component:self.position_limits[component] for component in components}

            components_best_bid_volumes = {best_bid_price: abs(state.order_depths[component].buy_orders[best_bid_price]) for component, best_bid_price in components_best_bid_prices.items()}
            components_best_ask_volumes = {best_ask_price: abs(state.order_depths[component].sell_orders[best_ask_price]) for component, best_ask_price in components_best_ask_prices.items()}

            # define main product to component volume ratios
            ratios = [2, 4, 1]
            main_product_to_components_ratios = dict(zip(components, ratios))

            # preprocessing for the main product 
            main_product = PICNIC_BASKET
            main_product_best_bid_price = best_bids[PICNIC_BASKET]
            main_product_best_ask_price = best_asks[PICNIC_BASKET]
            main_product_best_bid_volume = abs(state.order_depths[PICNIC_BASKET].buy_orders[main_product_best_bid_price])
            main_product_best_ask_volume = abs(state.order_depths[PICNIC_BASKET].sell_orders[main_product_best_ask_price])
            main_product_position = positions[PICNIC_BASKET]
            main_product_position_limit = self.position_limits[PICNIC_BASKET]

            main_product_orders, component_1_orders, component_2_orders, component_3_orders = self.difference_mean_reversion(state.timestamp, components, main_product_to_components_ratios, 
                                    components_best_bid_prices, components_best_ask_prices, components_best_bid_volumes, 
                                    components_best_ask_volumes, components_positions, components_position_limits,
                                    main_product, main_product_best_bid_price, main_product_best_bid_volume, 
                                    main_product_best_ask_price, main_product_best_ask_volume, 
                                    main_product_position, main_product_position_limit)

            result[PICNIC_BASKET] = main_product_orders
            result[BAGUETTE] = component_1_orders
            result[DIP] = component_2_orders
            result[UKULELE] = component_3_orders
            self.logger.log(
                f"PICNIC_BASKET: {main_product_orders}", "orders")
            self.logger.log(
                f"BAGUETTE: {component_1_orders}", "orders")
            self.logger.log(
                f"DIP: {component_2_orders}", "orders")
            self.logger.log(
                f"UKULELE: {component_3_orders}", "orders")
        else:
            self.logger.log(
                f"{PICNIC_BASKET} stop trading because of profit {profits[PICNIC_BASKET] + profits[BAGUETTE] + profits[DIP] + profits[UKULELE]} below {self.profit_threshold[PICNIC_BASKET] + self.profit_threshold[BAGUETTE] + self.profit_threshold[DIP] + self.profit_threshold[UKULELE]}", "important")


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
        self.logger.log(f"upper: {upper}, lower: {lower}", "bananas")

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

        # else just clear position gradually
        else:
            price = best_ask - margin if position > 0 else best_bid + margin
            clear_amount = min(-position, clear_limit) if position < 0 else - \
                min(position, clear_limit)
            orders.append(Order(product, price, clear_amount))

        return orders

    def window_arbitrage_calc(self, timestamp, mid_window, cb_pa_window, pb_ca_window, product_1: Product, product_2: Product,
                              mid_price_product_1, mid_price_product_2,
                              best_bid_price_product_1: int, best_ask_price_product_1: int,
                              best_bid_volume_product_1: int, best_ask_volume_product_1: int,
                              best_bid_price_product_2: int, best_ask_price_product_2: int,
                              best_bid_volume_product_2: int, best_ask_volume_product_2: int,
                              position_product_1: int, position_product_2: int,
                              product_1_over_product_2_value_ratio: float,
                              lot_size_product_1: int, lot_size_product_2: int,
                              position_limit_product_1: int, position_limit_product_2: int) -> List[Order]:
        """
        Arbitrage between two related products.

        Input Arguments:
        window -- BLG window of price difference (1-2)
        product_1 -- the main product we trade on
        product_2 -- the related product we hedge (take reverse trades on)
                         to reduce risk
        product_1_over_product_2_value_ratio -- the ratio of prices between two related products
        lot_size_product -- 15 for COCONAS (price at 8k), 8 for PINA (price at 15k)
        """
        if timestamp < self.cb_pa_window.size * 100:
            return [], []

        n = 2
        m = 2
        margin = 0
        trade_amount = 16
        clear_limit = 100
        multiplier = 12

        self.logger.log(
            f"WINDOW ARBITRAGE PARAMETERS: n={n}, margin={margin}, trade_amount={trade_amount},clear_limit={clear_limit}", "params")

        product_1_orders = []
        product_2_orders = []

        buy_volume_product_1 = 0
        buy_volume_product_2 = 0
        sell_volume_product_1 = 0
        sell_volume_product_2 = 0

        mid_upper, mid_lower = mid_window.upper_lower_bounds(n)

        # cbpa_upper = cb_pa_window.avg() + 10
        cbpa_upper, cb_pa_lower = cb_pa_window.upper_lower_bounds(n)
        cbpa_upper2, cb_pa_lower = cb_pa_window.upper_lower_bounds(m)
        # pbca_upper = pb_ca_window.avg() + 10
        pbca_upper, pbca_lower = pb_ca_window.upper_lower_bounds(n)
        pbca_upper2, pbca_lower = pb_ca_window.upper_lower_bounds(m)

        # if cbpa_upper < pbca_upper:
        #     cbpa_upper += 25
        # else:
        #     pbca_upper += 25

        mid_diff = mid_price_product_1 * lot_size_product_1 - \
            mid_price_product_2 * lot_size_product_2
        cbpa_diff = best_bid_price_product_1 * lot_size_product_1 - \
            best_ask_price_product_2 * lot_size_product_2
        pbca_diff = best_bid_price_product_2 * lot_size_product_2 - \
            best_ask_price_product_1 * lot_size_product_1

        # sell product 1 and buy product 2
        # if cbpa_diff > cbpa_upper\
        if mid_diff > mid_upper \
                and -position_product_1 < position_limit_product_1 and position_product_2 < position_limit_product_2:
            self.logger.log(
                f"sell product 1 and buy product 2 because cbpa_dff {cbpa_diff} > cbpa upper {cbpa_upper}", "debug")
            self.buy += 1
            self.sell = 0
            # how many lots of product 2 can I buy from the order book, without breaching position limit
            max_number_buy_lots2 = min(
                trade_amount + multiplier * self.buy,
                best_ask_volume_product_2,
                position_limit_product_2 - position_product_2)/lot_size_product_2

            # how many lots of product 2 can I buy from the order book, without breaching position limit
            max_number_buy_lots1 = min(
                trade_amount + multiplier * self.buy,
                best_ask_volume_product_1,
                position_limit_product_1 - position_product_1)/lot_size_product_1

            # the number of lots I can trade, this is the smaller of the two
            number_trade_lots = min(max_number_buy_lots2, max_number_buy_lots1)
            buy_volume_product_2 = round(
                number_trade_lots * lot_size_product_2)
            buy_volume_product_1 = round(
                number_trade_lots * lot_size_product_1)

        # buy product 1 and sell product 2
        # elif pbca_diff > pbca_upper\
        elif mid_diff < mid_lower\
                and position_product_1 < position_limit_product_1 and -position_product_2 < position_limit_product_2:
            self.logger.log(
                f"buy product 1 and sell product 2 because pbca diff {pbca_diff} > pbca upper {pbca_upper}", "debug")
            self.sell += 1
            self.buy = 0

            # how many lots of product 1 can I sell from the order book, without breaching position limit
            max_number_sell_lots1 = min(
                trade_amount + multiplier * self.sell,
                best_bid_volume_product_1,
                position_limit_product_1 + position_product_1)/lot_size_product_1

            # how many lots of product 1 can I sell from the order book, without breaching position limit
            max_number_sell_lots2 = min(
                trade_amount + multiplier * self.sell,
                best_bid_volume_product_2,
                position_limit_product_2 + position_product_2)/lot_size_product_2

            # the number of lots I can trade, this is the smaller of the two
            number_trade_lots = min(
                max_number_sell_lots1, max_number_sell_lots2)

            sell_volume_product_2 = round(
                number_trade_lots * lot_size_product_2)
            sell_volume_product_1 = round(
                number_trade_lots * lot_size_product_1)

        # else:  # clear
        #     if position_product_1 > 0:  # need to sell
        #         sell_volume_product_1 = min(clear_limit, position_product_1)
        #     else:  # need to buy
        #         buy_volume_product_1 = min(clear_limit, -position_product_1)
        #     if position_product_2 > 0:
        #         sell_volume_product_2 = min(clear_limit, position_product_2)
        #     else:  # need to buy
        #         buy_volume_product_2 = min(clear_limit, -position_product_2)

        # MM price
        # buy_price_product_2 = best_bid_price_product_2 + \
        #     (margin * 2 if cbpa_diff > cbpa_upper2 else margin)
        # sell_price_product_1 = best_ask_price_product_1 - \
        #     (margin * 1 if cbpa_diff > cbpa_upper2 else margin)
        # buy_price_product_1 = best_bid_price_product_1 + \
        #     (margin * 1 if pbca_diff > pbca_upper2 else margin)
        # sell_price_product_2 = best_ask_price_product_2 - \
        #     (margin * 2 if pbca_diff > pbca_upper2 else margin)

        buy_price_product_2 = best_ask_price_product_2 + margin
        sell_price_product_1 = best_bid_price_product_1 - margin
        buy_price_product_1 = best_ask_price_product_1 + margin
        sell_price_product_2 = best_bid_price_product_2 - margin

        # Send orders
        # if we can sell product 1 and buy product 2
        # N.B. I assumed all the volumes given are positive
        product_1_orders.append(
            Order(product_1, sell_price_product_1, -sell_volume_product_1))
        product_2_orders.append(
            Order(product_2, buy_price_product_2, buy_volume_product_2))

        # if we can sell product 2 and buy product 1
        product_2_orders.append(
            Order(product_2, sell_price_product_2, -sell_volume_product_2))
        product_1_orders.append(
            Order(product_1, buy_price_product_1, buy_volume_product_1))

        return (product_1_orders, product_2_orders)

    def window_price_diff_calc(self, timestamp, mid_window_larger, mid_window_smaller, product_1: Product, product_2: Product,
                               mid_price_product_1, mid_price_product_2,
                               best_bid_price_product_1: int, best_ask_price_product_1: int,
                               best_bid_volume_product_1: int, best_ask_volume_product_1: int,
                               best_bid_price_product_2: int, best_ask_price_product_2: int,
                               best_bid_volume_product_2: int, best_ask_volume_product_2: int,
                               position_product_1: int, position_product_2: int,
                               product_1_over_product_2_value_ratio: float,
                               lot_size_product_1: int, lot_size_product_2: int,
                               position_limit_product_1: int, position_limit_product_2: int) -> List[Order]:
        trade_cooldown = 50
        min_diff_in_window_avg = 10  # if below, don't trade
        margin1 = 0  # coco
        margin2 = 3  # pina
        trade_amount = 160000

        if timestamp < mid_window_larger.size * 100:
            return [], []

        product_1_orders = []
        product_2_orders = []

        buy_volume_product_1 = 0
        buy_volume_product_2 = 0
        sell_volume_product_1 = 0
        sell_volume_product_2 = 0

        # buy
        if (mid_window_larger.avg() > mid_window_smaller.avg()):
            if self.window1_larger == None or not self.window1_larger:
                self.window1_larger = True
                if timestamp - self.last_peak_or_valley < trade_cooldown * 100:
                    # or abs(self.mid_diff_last_trade - mid_window_smaller.avg()) < min_diff_in_window_avg:
                    return [], []
                self.last_peak_or_valley = timestamp
                self.mid_diff_last_trade = mid_window_smaller.avg()
                buy_volume_product_2 = min(
                    trade_amount,
                    position_limit_product_2 - position_product_2)

                buy_volume_product_1 = min(
                    trade_amount,
                    position_limit_product_1 - position_product_1)

        # sell
        else:
            if self.window1_larger == None or self.window1_larger:
                self.window1_larger = False
                if timestamp - self.last_peak_or_valley < trade_cooldown * 100:
                    # or abs(self.mid_diff_last_trade - mid_window_smaller.avg()) < min_diff_in_window_avg:
                    return [], []
                self.last_peak_or_valley = timestamp
                self.mid_diff_last_trade = mid_window_smaller.avg()
                sell_volume_product_2 = min(
                    trade_amount,
                    position_limit_product_2 + position_product_2)

                sell_volume_product_1 = min(
                    trade_amount,
                    position_limit_product_1 + position_product_1)

        buy_price_product_2 = best_ask_price_product_2 + margin2
        sell_price_product_1 = best_bid_price_product_1 - margin1
        buy_price_product_1 = best_ask_price_product_1 + margin1
        sell_price_product_2 = best_bid_price_product_2 - margin2

        # Send orders

        product_1_orders.append(
            Order(product_1, sell_price_product_1, -sell_volume_product_1))
        product_2_orders.append(
            Order(product_2, buy_price_product_2, buy_volume_product_2))

        product_2_orders.append(
            Order(product_2, sell_price_product_2, -sell_volume_product_2))
        product_1_orders.append(
            Order(product_1, buy_price_product_1, buy_volume_product_1))

        return (product_1_orders, product_2_orders)

    def dumb_calc(self, product, window, mid_price, trader,
                  best_bid_price: int, best_ask_price: int,
                  best_bid_volume: int, best_ask_volume: int,
                  position: int, position_limit: int) -> List[Order]:
        orders = []
        avg = window.avg()
        bound = 3
        trade_amount = 100
        buy_amount = 0
        sell_amount = 0
        min_profit = 3

        # if above average by n, sell
        if best_bid_price >= trader.last_bought_price + min_profit and position > 0:
            self.logger.log(
                f"profit chance of {best_bid_price} - {trader.last_bought_price}")
            sell_amount = position
        elif mid_price > avg + bound and position < 100:
            sell_amount = min(position+position_limit,
                              best_bid_volume, trade_amount)
            if sell_amount > 0 and position == 0:
                trader.last_sell_price = best_bid_price
        # if below average by n, buy
        if best_ask_price <= trader.last_sell_price - min_profit and position < 0:
            self.logger.log(
                f"profit chance of {trader.last_sell_price} - {best_ask_price}")
            buy_amount = -position
        elif mid_price < avg - bound and position < -100:
            buy_amount = min(position_limit-position,
                             best_ask_volume, trade_amount)
            if buy_amount > 0 and position == 0:
                trader.last_bought_price = best_ask_price

        if sell_amount > 0:
            orders.append(Order(product, best_bid_price, -sell_amount))
        if buy_amount > 0:
            orders.append(Order(product, best_ask_price, buy_amount))

        return orders

    def arbitrage_calc(self, timestamp: int, product_1: Product, product_2: Product,
                       best_bid_price_product_1: int, best_ask_price_product_1: int,
                       best_bid_volume_product_1: int, best_ask_volume_product_1: int,
                       best_bid_price_product_2: int, best_ask_price_product_2: int,
                       best_bid_volume_product_2: int, best_ask_volume_product_2: int,
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
        buy_price_product_1 = 0
        buy_price_product_2 = 0
        sell_price_product_1 = 0
        sell_price_product_2 = 0
        buy_volume_product_1 = 0
        buy_volume_product_2 = 0
        sell_volume_product_1 = 0
        sell_volume_product_2 = 0
        product_1_orders = []
        product_2_orders = []

        long_term_profit_threshold = 100
        zero_frequency_threshold = 6
        short_term_flag_duration = 200

        normalized_mid_price_product_1 = (best_ask_price_product_1 + best_bid_price_product_1) * lot_size_product_1/2
        normalized_mid_price_product_2 = (best_ask_price_product_2 + best_bid_price_product_2) * lot_size_product_2/2
        diff_normalized_mid_price = (normalized_mid_price_product_1 - normalized_mid_price_product_2) # coco price - pinas price
        self.new_arbitrage_long_term_mid_diff_prices.push(diff_normalized_mid_price)
        self.new_arbitrage_short_term_mid_price_window.push(diff_normalized_mid_price)
        self.new_arbitrage_zero_counter_window.push(diff_normalized_mid_price)

        def long_term_trade():
            """
            if detected buy signal --> buy to limit
            if detected sell signal --> sell to limit
            if position is not zero and (no signal /passed zero) --> clear 
            """
            self.logger.log(f"We are still in long term mode", 'debug')
            long_term_average_mid_price = self.new_arbitrage_long_term_mid_diff_prices.avg()\
                if len(self.new_arbitrage_long_term_mid_diff_prices.contents) == self.new_arbitrage_long_term_mid_diff_prices.size\
                else np.nan
            self.logger.log(f"Long term Average = {long_term_average_mid_price}", 'debug')
            self.logger.log(f'diff mid price = {diff_normalized_mid_price}','debug')
            num_std = 1

            current_rise_signal = (diff_normalized_mid_price< -long_term_profit_threshold) and (self.new_arbitrage_short_term_mid_price_window.upper_lower_bounds(n=num_std)[1] > long_term_average_mid_price)
            current_fall_signal = (diff_normalized_mid_price > long_term_profit_threshold) and (self.new_arbitrage_short_term_mid_price_window.upper_lower_bounds(n=num_std)[0] < long_term_average_mid_price)
            # buy signal: price is below the threhold and rises higher than the long term average
            if (position_product_1 < position_limit_product_1 and position_product_2 < position_limit_product_2) and (self.new_arbitrage_buy_to_limit_signal or current_rise_signal):
                if current_rise_signal:
                    self.new_arbitrage_buy_to_limit_signal = True
                    self.logger.log(f'Rising signal (at bottom) detected, we now buy in, current coconut position is {position_product_1}', 'debug')

                # buy product 1, sell product 2: (i.e. buy product1 - product 2)
                max_number_buy_lots = min(best_ask_volume_product_1,\
                                         (position_limit_product_1 - position_product_1))// lot_size_product_1
                    
                # how many lots of product 2 can I sell from the order book, without breaching position limit
                max_number_sell_lots = min(best_bid_volume_product_2,\
                                        (position_limit_product_2 + position_product_2))//lot_size_product_2
                number_trade_lots = min(max_number_buy_lots, max_number_sell_lots) # the number of lots I can trade, this is the smaller of the two
                
                long_term_buy_volume_product_1 = number_trade_lots * lot_size_product_1
                long_term_sell_volume_product_2 = number_trade_lots * lot_size_product_2
                long_term_buy_price_product_1 = best_ask_price_product_1
                long_term_sell_price_product_2 = best_bid_price_product_2

                return (long_term_buy_price_product_1,0,
                        0,long_term_sell_price_product_2,
                        long_term_buy_volume_product_1,0,
                        0,long_term_sell_volume_product_2)
                
            # sell signal: price is above the threhold and falls lower than the long term average
            elif(position_product_1 > -position_limit_product_1 and position_product_2 > -position_limit_product_2) and (self.new_arbitrage_sell_to_limit_signal or current_fall_signal):
                if current_fall_signal:
                    self.new_arbitrage_sell_to_limit_signal = True
                    self.logger.log(f'Falling signal (at top) detected, we now sell out, current coconut position is {position_product_1}', 'debug')

                # buy product 2, sell product 1 (i.e. sell product 1 - product 2)
                max_number_buy_lots = min(best_ask_volume_product_2,\
                                         (position_limit_product_2 - position_product_2))//lot_size_product_2

                # how many lots of product 1 can I sell from the order book, without breaching position limit
                max_number_sell_lots = min(best_bid_volume_product_1,\
                                        (position_limit_product_1 + position_product_1))//lot_size_product_1
                
                number_trade_lots = min(max_number_buy_lots, max_number_sell_lots) # the number of lots I can trade, this is the smaller of the two
                
                long_term_buy_volume_product_2 = number_trade_lots * lot_size_product_2
                long_term_sell_volume_product_1 = number_trade_lots * lot_size_product_1
                long_term_buy_price_product_2 = best_ask_price_product_2
                long_term_sell_price_product_1 = best_bid_price_product_1

                return (0,long_term_buy_price_product_2,
                        long_term_sell_price_product_1,0,
                        0,long_term_buy_volume_product_2,
                        long_term_sell_volume_product_1,0)
            # if we started with long term trade, we need to clear the positions
            # 1.  when we just started
            # 2.  when the price diff has gone back to zero.
            elif (position_product_1!=0 or position_product_2!=0)and\
                 (not(self.new_arbitrage_buy_to_limit_signal or self.new_arbitrage_sell_to_limit_signal) or self.new_arbitrage_zero_counter_window.contents[-1] * self.new_arbitrage_zero_counter_window.contents[-2] <= 0):
                # clear the position
                if self.new_arbitrage_zero_counter_window.contents[-1] * self.new_arbitrage_zero_counter_window.contents[-2] <= 0:
                    self.logger.log(f'We are now clearing the positions from long term trade as price goes to zero, current position of coconut is {position_product_1}','debug')
                else:
                    self.logger.log(f'We are now clearing the position from long term as previous price went to zero, current position of coconut is {position_product_1}','debug')
                self.new_arbitrage_buy_to_limit_signal = False
                self.new_arbitrage_sell_to_limit_signal = False
                if position_product_1 > 0 and position_product_2 < 0:
                    # sell product 1, buy product 2
                    max_number_buy_lots = min(best_ask_volume_product_2,\
                                             (-position_product_2))//lot_size_product_2

                    # how many lots of product 1 can I sell from the order book, without breaching position limit
                    max_number_sell_lots = min(best_bid_volume_product_1,\
                                            (position_product_1))//lot_size_product_1
                    
                    number_trade_lots = min(max_number_buy_lots, max_number_sell_lots) # the number of lots I can trade, this is the smaller of the two
                    
                    long_term_buy_volume_product_2 = number_trade_lots * lot_size_product_2
                    long_term_sell_volume_product_1 = number_trade_lots * lot_size_product_1
                    long_term_buy_price_product_2 = best_ask_price_product_2
                    long_term_sell_price_product_1 = best_bid_price_product_1

                    return (0,long_term_buy_price_product_2,
                            long_term_sell_price_product_1,0,
                            0,long_term_buy_volume_product_2,
                            long_term_sell_volume_product_1,0)

                elif position_product_1 < 0 and position_product_2 >0:
                    # buy product 1, sell product 2
                    max_number_buy_lots = min(best_ask_volume_product_1,\
                                             (-position_product_1))// lot_size_product_1
                        
                    # how many lots of product 2 can I sell from the order book, without breaching position limit
                    max_number_sell_lots = min(best_bid_volume_product_2,\
                                            (position_product_2))//lot_size_product_2
                    number_trade_lots = min(max_number_buy_lots, max_number_sell_lots) # the number of lots I can trade, this is the smaller of the two
                    
                    long_term_buy_volume_product_1 = number_trade_lots * lot_size_product_1
                    long_term_sell_volume_product_2 = number_trade_lots * lot_size_product_2
                    long_term_buy_price_product_1 = best_ask_price_product_1
                    long_term_sell_price_product_2 = best_bid_price_product_2

                    return (long_term_buy_price_product_1,0,
                            0,long_term_sell_price_product_2,
                            long_term_buy_volume_product_1,0,
                            0,long_term_sell_volume_product_2)

            else:
                # Dont't do any trade otherwise
                return (0,0,0,0,
                        0,0,0,0)

        def short_term_trade():
            self.logger.log(f"btw, we are in short term mode", 'debug')
            big_diff = 25
            # if normalized product 2 is cheaper than product 1
            if best_bid_price_product_1 > best_ask_price_product_2 * product_1_over_product_2_value_ratio:
                # for now we only buy / sell at the best bid and best ask prices
                # buy in product 2, sell product 1
                if best_bid_price_product_1 * lot_size_product_1 - best_ask_price_product_2 * lot_size_product_2 > big_diff:
                    # how many lots of product 2 can I buy from the order book, without breaching position limit
                    max_number_buy_lots = min(best_ask_volume_product_2,\
                                            (position_limit_product_2 - position_product_2))//lot_size_product_2

                    # how many lots of product 1 can I sell from the order book, without breaching position limit
                    max_number_sell_lots = min(best_bid_volume_product_1,\
                                            (position_limit_product_1 + position_product_1))//lot_size_product_1
                    
                    number_trade_lots = min(max_number_buy_lots, max_number_sell_lots) # the number of lots I can trade, this is the smaller of the two
                    
                    short_term_buy_volume_product_2 = number_trade_lots * lot_size_product_2
                    short_term_sell_volume_product_1 = number_trade_lots * lot_size_product_1
                    short_term_buy_price_product_2 = best_ask_price_product_2
                    short_term_sell_price_product_1 = best_bid_price_product_1

                    return (0,short_term_buy_price_product_2,
                            short_term_sell_price_product_1,0,
                            0,short_term_buy_volume_product_2,
                            short_term_sell_volume_product_1,0)
        
            # if normalized product 1 is cheaper than prodcut 2
            if best_bid_price_product_2 * product_1_over_product_2_value_ratio > best_ask_price_product_1:
                # buy in product 1, sell product 2
                if best_bid_price_product_2 * lot_size_product_2 - best_ask_price_product_1 * lot_size_product_1 > big_diff:
                    # how many lots of product 1 can I buy from the order book, without breaching position limit
                    max_number_buy_lots = min(best_ask_volume_product_1,\
                                            (position_limit_product_1 - position_product_1))// lot_size_product_1
                    
                    # how many lots of product 2 can I sell from the order book, without breaching position limit
                    max_number_sell_lots = min(best_bid_volume_product_2,\
                                            (position_limit_product_2 + position_product_2))//lot_size_product_2
                    number_trade_lots = min(max_number_buy_lots, max_number_sell_lots) # the number of lots I can trade, this is the smaller of the two
                    
                    short_term_buy_volume_product_1 = number_trade_lots * lot_size_product_1
                    short_term_sell_volume_product_2 = number_trade_lots * lot_size_product_2
                    short_term_buy_price_product_1 = best_ask_price_product_1
                    short_term_sell_price_product_2 = best_bid_price_product_2

                    return (short_term_buy_price_product_1,0,
                            0,short_term_sell_price_product_2,
                            short_term_buy_volume_product_1,0,
                            0,short_term_sell_volume_product_2)
            
            return (0,0,0,0,
                    0,0,0,0)

        self.new_arbitrage_short_term_flag_duration_counter -=1 if self.new_arbitrage_short_term_flag_duration_counter >0 else 0
        # if not enough points to tell if we are in short term or long term
        if timestamp <= self.new_arbitrage_zero_counter_window.size*100:
            self.logger.log(f'enter short term trade because not enough 0 counter with size of zero counter: {len(self.new_arbitrage_zero_counter_window.contents)}', 'debug')
            self.new_arbitrage_short_term_flag = True
            self.new_arbitrage_long_term_flag = False
        elif self.new_arbitrage_zero_counter_window.contents[-1] * self.new_arbitrage_zero_counter_window.contents[-2] <= 0: # switch the flag if it passes through zero
            self.logger.log(f'now we have enough data distinguishing long/short', 'debug')
            # count the number of zeros in the last 50 prices
            number_of_zeros = 0
            for index in range(1,self.new_arbitrage_zero_counter_window.size):
                if self.new_arbitrage_zero_counter_window.contents[index] * self.new_arbitrage_zero_counter_window.contents[index-1] <= 0:# passed through a zero
                    number_of_zeros += 1
                    self.logger.log(f'number of zeroes over the last 50 ticks is {number_of_zeros}', 'debug')

            if self.new_arbitrage_short_term_flag_duration_counter>0 or  number_of_zeros >= zero_frequency_threshold:
                if number_of_zeros >= zero_frequency_threshold: # if we come here because of short term trade signal
                    self.new_arbitrage_buy_to_limit_signal = False
                    self.new_arbitrage_sell_to_limit_signal = False
                    self.new_arbitrage_short_term_flag_duration_counter = short_term_flag_duration
                    self.logger.log(f'entering short term trade as number of zeroes is {number_of_zeros}, current coconut position is {position_product_1}', 'debug')
                else:
                    self.logger.log(f'We are in short term mode because there was a trigger within 200 ticks','debug')
                self.new_arbitrage_short_term_flag = True
                self.new_arbitrage_long_term_flag = False
            else:
                self.logger.log(f'entering long term trade as number of zeroes is {number_of_zeros}, current coconut position is {position_product_1}', 'debug')
                self.new_arbitrage_short_term_flag = False
                self.new_arbitrage_long_term_flag = True
                
        if self.new_arbitrage_long_term_flag:
            (buy_price_product_1,
            buy_price_product_2,
            sell_price_product_1,
            sell_price_product_2,
            buy_volume_product_1,
            buy_volume_product_2,
            sell_volume_product_1,
            sell_volume_product_2) = long_term_trade()
        elif self.new_arbitrage_short_term_flag:
            (buy_price_product_1,
            buy_price_product_2,
            sell_price_product_1,
            sell_price_product_2,
            buy_volume_product_1,
            buy_volume_product_2,
            sell_volume_product_1,
            sell_volume_product_2) = short_term_trade()
        else:
            self.logger.log('There is not a valid flag to tell long term / short term, most likely because of lamda reset','important')

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

    def indicator_trade(self, timestamp, product, indicator_mid_price,
                        best_bid_price, best_bid_volume, best_ask_price,
                        best_ask_volume, product_position, product_position_limit, dolphin_diff_term, gear_diff_term):
        # push the current price into the indicator window and product window
        product_mid_price = (best_bid_price + best_ask_price)/2
        self.gear_window.push(product_mid_price)
        self.dolphin_window.push(indicator_mid_price)

        # define time spans for calulating entrace_tracer and short/long term average
        long_term = 800
        short_term = 300
        time_span_for_calculating_entrance_tracer = 10

        # create a DOLPHIN_SIGHTINGS difference series, containing the differences between i th and i+diff th elements
        dolphin_differences_series = pd.Series(
            self.dolphin_window.contents).diff(periods=dolphin_diff_term)
        # calculate entrance tracer
        entrance_tracer = dolphin_differences_series.iloc[-time_span_for_calculating_entrance_tracer:].mean()\
            if len(dolphin_differences_series) == time_span_for_calculating_entrance_tracer + dolphin_diff_term + 1 else np.nan

        # long term and short term sigma and mean for the DIVING_GEAR series
        gear_differences_series = pd.Series(
            self.gear_window.contents).diff(periods=gear_diff_term)
        long_gear_differences_series = gear_differences_series.iloc[-long_term:]
        short_gear_differences_series = gear_differences_series.iloc[-short_term:]

        gear_long_term_average_diff = long_gear_differences_series.mean()\
            if len(long_gear_differences_series) == long_term else np.nan
        gear_short_term_average_diff = short_gear_differences_series.mean()\
            if len(short_gear_differences_series) == short_term else np.nan
        exit_tracer = gear_long_term_average_diff - gear_short_term_average_diff

        absolute_threshold = 0.5
        num_std_entrance = 4
        num_std_exit = 4

        product_orders = []

        self.logger.log(
            f"entrance tracer window: {self.entrance_tracer_window.contents}", "debug")
        self.logger.log(
            f'exit tracer window: {self.exit_tracer_window.contents}', "debug")
        # when big peak & big troughts comes:
        # after enough number of entrance_tracer in recorded
        if timestamp > time_span_for_calculating_entrance_tracer*100:
            # 1. condition to buy
            if (entrance_tracer > num_std_entrance * self.entrance_tracer_window.std() and entrance_tracer > absolute_threshold)\
                    or (self.gear_buy_flag == True and product_position < product_position_limit):  # if there is a new trade signal or a flag

                self.gear_buy_flag = True
                self.gear_sell_flag = False
                buy_volume = min(
                    best_ask_volume, product_position_limit - product_position)
                if buy_volume > 0:
                    product_orders.append(
                        Order(product, best_ask_price, buy_volume))
                    self.logger.log(
                        f'buying because indicator indicates upward surge, with indicator value: {entrance_tracer} at timestamp: {timestamp}, with standard deviation {self.entrance_tracer_window.std()}', 'debug')

            # 2. condition to sell
            elif (entrance_tracer < -num_std_entrance * self.entrance_tracer_window.std() and entrance_tracer < -absolute_threshold)\
                    or (self.gear_sell_flag == True and product_position > -product_position_limit):  # if there is a new trade signal or a flag

                self.gear_sell_flag = True
                self.gear_buy_flag = False
                sell_volume = min(
                    best_bid_volume, product_position_limit + product_position)
                if sell_volume > 0:
                    product_orders.append(
                        Order(product, best_bid_price, -sell_volume))
                    self.logger.log(
                        f'selling because indicator indicates downward surge, with indicator value: {entrance_tracer} at timestamp: {timestamp}, with standard deviation {self.entrance_tracer_window.std()}', 'debug')
        # push in the current entrance tracer
        self.entrance_tracer_window.push(entrance_tracer)

        if timestamp > long_term*100:  # after enough number of gear prices is recorded
            # when big surge ends
            # 1. when a peak ends and starts to drop
            if exit_tracer > num_std_exit * self.exit_tracer_window.std():
                self.gear_buy_flag = False
                clear_volume = product_position
                if clear_volume != 0 and self.gear_sell_flag == False:
                    # note that clear volume must be the negative value of current position so that we reset to position 0 for short-term trade
                    product_orders.append(
                        Order(product, best_bid_price, -clear_volume))
                    self.logger.log(
                        f'clearing as plateau reached, with exit tracer value: {exit_tracer} at timestamp: {timestamp}, with standard deviation {self.exit_tracer_window.std()}', 'debug')

            # 2. when a big trough ends and starts to increase
            if exit_tracer < -num_std_exit * self.exit_tracer_window.std():
                self.gear_sell_flag = False
                clear_volume = product_position
                if clear_volume != 0 and self.gear_buy_flag == False:
                    product_orders.append(
                        Order(product, best_ask_price, -clear_volume))
                    self.logger.log(
                        f'clearing as plateau reached, with exit tracer value: {exit_tracer} at timestamp: {timestamp}, with standard deviation {self.exit_tracer_window.std()}', 'debug')
        # push in the current exit tracer
        self.exit_tracer_window.push(exit_tracer)

        return product_orders

    def mayberry_calc(self, timestamp, product: Product, window,
                      mid_price,
                      best_bid, best_ask,
                      best_bid_volume, best_ask_volume,
                      position_limit, position):
        orders = []
        PLATEAU_START = 3500 * 100
        UPWARD_START = 4500 * 100
        DOWNWARD_START = 7000 * 100
        renbuliaole_threshold = 20

        long_term_average_price = window.avg()

        if (timestamp >= PLATEAU_START) and (timestamp < UPWARD_START):
            if mid_price > long_term_average_price and not self.start_done:
                # buy to limit
                if position == position_limit:
                    self.start_done = True
                self.logger.log("BERRIES decide to buy to limit", "debug")
                orders.append(Order(product, best_ask, min(
                    position_limit-position, best_ask_volume)))

        elif (timestamp >= UPWARD_START) and (timestamp < DOWNWARD_START):
            if mid_price < long_term_average_price and not self.peak_done:
                # sell to limit
                if position == -position_limit:
                    self.peak_done = True
                self.logger.log("BERRIES decide to sell to limit", "debug")
                orders.append(Order(product, best_bid, -
                              min(best_bid_volume, position+position_limit)))

        elif timestamp >= DOWNWARD_START:
            if mid_price > long_term_average_price and self.good_clear_price == 0:
                # buy to clear
                self.logger.log(
                    "BERRIES decide found its good point to clear, but we wait", "debug")
                self.good_clear_price = mid_price
            if (self.good_clear_price != 0 and mid_price) > (self.good_clear_price + renbuliaole_threshold) and not self.end_done:
                self.logger.log(
                    "Renbuliaole!, now BERRIES decide to clear", "debug")
                if position == 0:
                    self.end_done = True
                orders.append(Order(product, best_ask, min(
                    position_limit-position, best_ask_volume)))

        return orders
    
    def mayberry_calc_olivia(self, timestamp, product: Product, window,
                      mid_price,
                      best_bid, best_ask,
                      best_bid_volume, best_ask_volume,
                      position_limit, position):
        
        orders = []
        PLATEAU_START = 3500 * 100
        UPWARD_START = 4500 * 100
        # DOWNWARD_START = 7000 * 100
        # renbuliaole_threshold = 20

        long_term_average_price = window.avg()


        # determine when we can sit back and wait
        if position == position_limit and self.olivia_buy_flag == True:
            self.olivia_buy_flag = False
            self.logger.log(f'we can stop buying now, setting olivia buy flag to {self.olivia_buy_flag} and sit back for the next time olivia sells', 'debug')

        elif position == -position_limit and self.olivia_sell_flag == True:
            self.olivia_sell_flag = False
            self.logger.log(f'we can stop selling now, setting olivia sell flag to {self.olivia_sell_flag} and sit back for the next time olivia buys', 'debug')

        # three action scenarios
        if (timestamp >= PLATEAU_START) and (timestamp < UPWARD_START):
            if mid_price > long_term_average_price and not self.start_done:
                # buy to limit
                if position == position_limit:
                    self.start_done = True
                self.logger.log("BERRIES decide to buy to limit", "debug")
                orders.append(Order(product, best_ask, min(
                    position_limit-position, best_ask_volume)))
        
        elif self.olivia_sell_flag == True:
            # sell to limit
            self.logger.log(f"BERRIES decide to sell to limit because olivia sell flag is {self.olivia_sell_flag}", "debug")
            orders.append(Order(product, best_bid, -
                            min(best_bid_volume, position+position_limit)))
        
        elif self.olivia_buy_flag == True:
            # buy to limit
            self.logger.log(f"BERRIES decide to sell to limit because olivia sell flag is {self.olivia_buy_flag}", "debug")
            orders.append(Order(product, best_ask, min(
                    position_limit-position, best_ask_volume)))
            
        

        
        return orders





    
    def difference_mean_reversion(self, timestamp, components:list, main_product_to_components_ratios:dict, 
                                 components_best_bid_prices:dict, components_best_ask_prices:dict, components_best_bid_volumes:dict, 
                                 components_best_ask_volumes:dict, components_positions:dict, components_position_limits:dict,
                                 main_product, main_product_best_bid_price, main_product_best_bid_volume, 
                                 main_product_best_ask_price, main_product_best_ask_volume, main_product_position, main_product_position_limit):
        main_product_orders = []
        component_1_orders = []
        component_2_orders = []
        component_3_orders = []
        
        # get components
        component_1, component_2, component_3 = components

        # calculate the mid price of the components
        component_1_mid_price = (components_best_ask_prices[component_1] + components_best_bid_prices[component_1])/2
        component_2_mid_price = (components_best_ask_prices[component_2] + components_best_bid_prices[component_2])/2
        component_3_mid_price = (components_best_ask_prices[component_3] + components_best_bid_prices[component_3])/2

        components_mid_price = component_1_mid_price * main_product_to_components_ratios[component_1] + component_2_mid_price * main_product_to_components_ratios[component_2] + component_3_mid_price * main_product_to_components_ratios[component_3]
        
        self.logger.log(f'the current weighted mid price of components is {components_mid_price}', 'debug')

        # calculate the mid price of the main product, i.e. picnic basket
        main_product_mid_price = (main_product_best_bid_price + main_product_best_ask_price)/2
        current_difference = main_product_mid_price - components_mid_price

        self.logger.log(f'the current difference between main product and components is {current_difference}', 'debug')
        
        # two options: 
        # 1. use absolute mean since the differences are relatively constant;
        # 2. use rolling mean from the past differences from the difference window
        past_differences_sigma = self.difference_between_main_product_and_components_window.std()
        past_differences_mean = self.difference_between_main_product_and_components_window.avg()
        absolute_threshold = 100

        constant_differences_mean = 400
        constant_differences_sigma = 125
        num_std = 1

        # suppose we are using the constant_difference_mean
        if current_difference - num_std * constant_differences_sigma > constant_differences_mean:
            # we consider this to be a high point and hence we sell PICNIC_BASKET and buy EQUAL WORTH of COMPONENTS
            self.logger.log(f'selling main product because {current_difference} is much greater than {constant_differences_mean}', 'debug')
            main_product_sell_price = main_product_best_bid_price
            main_product_max_sell_volume = min(main_product_best_bid_volume, main_product_position_limit + main_product_position)
            
            # we must do the opposite to the components
            component_1_buy_price = components_best_ask_prices[component_1]
            component_1_max_buy_volume_normalized = min(components_best_ask_volumes[component_1_buy_price], components_position_limits[component_1] - components_positions[component_1])//main_product_to_components_ratios[component_1]
            component_2_buy_price = components_best_ask_prices[component_2]
            component_2_max_buy_volume_normalized = min(components_best_ask_volumes[component_2_buy_price], components_position_limits[component_2] - components_positions[component_2])//main_product_to_components_ratios[component_2]
            component_3_buy_price = components_best_ask_prices[component_3]
            component_3_max_buy_volume_normalized = min(components_best_ask_volumes[component_3_buy_price], components_position_limits[component_3] - components_positions[component_3])//main_product_to_components_ratios[component_3]
            
            normalized_min_volume = min([main_product_max_sell_volume, component_1_max_buy_volume_normalized, component_2_max_buy_volume_normalized, component_3_max_buy_volume_normalized])
            main_product_sell_volume = normalized_min_volume
            component_1_buy_volume = normalized_min_volume * main_product_to_components_ratios[component_1]
            component_2_buy_volume = normalized_min_volume * main_product_to_components_ratios[component_2]
            component_3_buy_volume = normalized_min_volume * main_product_to_components_ratios[component_3]

            self.logger.log(f'the selling volume is {main_product_sell_volume}, and the buying volumes are {component_1_buy_volume, component_2_buy_volume, component_3_buy_volume}', 'debug')

            if normalized_min_volume != 0:
                main_product_orders.append(Order(main_product, main_product_sell_price, -main_product_sell_volume))
                component_1_orders.append(Order(component_1, component_1_buy_price, component_1_buy_volume))
                component_2_orders.append(Order(component_2, component_2_buy_price, component_2_buy_volume))
                component_3_orders.append(Order(component_3, component_3_buy_price, component_3_buy_volume))
    
        elif current_difference + num_std * constant_differences_sigma < constant_differences_mean:
            # we consider this to be a low point and hence we buy PICNIC_BASKET and sell EQUAL WORTH of COMPONENTS
            self.logger.log(f'buying main product because {current_difference} is much less than {constant_differences_mean}', 'debug')
            main_product_buy_price = main_product_best_ask_price
            main_product_max_buy_volume = min(main_product_best_ask_volume, main_product_position_limit - main_product_position)
            
            # we must do the opposite to the component
            component_1_sell_price = components_best_bid_prices[component_1]
            component_1_max_sell_volume_normalized = min(components_best_bid_volumes[component_1_sell_price], components_position_limits[component_1] + components_positions[component_1])//main_product_to_components_ratios[component_1]
            component_2_sell_price = components_best_bid_prices[component_2]
            component_2_max_sell_volume_normalized = min(components_best_bid_volumes[component_2_sell_price], components_position_limits[component_2] + components_positions[component_2])//main_product_to_components_ratios[component_2]
            component_3_sell_price = components_best_bid_prices[component_3]
            component_3_max_sell_volume_normalized = min(components_best_bid_volumes[component_3_sell_price], components_position_limits[component_3] + components_positions[component_3])//main_product_to_components_ratios[component_3]
            
            normalized_min_volume = min([main_product_max_buy_volume, component_1_max_sell_volume_normalized, component_2_max_sell_volume_normalized, component_3_max_sell_volume_normalized])
            main_product_buy_volume = normalized_min_volume
            component_1_sell_volume = normalized_min_volume * main_product_to_components_ratios[component_1]
            component_2_sell_volume = normalized_min_volume * main_product_to_components_ratios[component_2]
            component_3_sell_volume = normalized_min_volume * main_product_to_components_ratios[component_3]


            self.logger.log(f'the buying volume is {main_product_buy_volume}, and the selling volumes are {component_1_sell_volume, component_2_sell_volume, component_3_sell_volume}', 'debug')
            if normalized_min_volume != 0:
                main_product_orders.append(Order(main_product, main_product_buy_price, main_product_buy_volume))
                component_1_orders.append(Order(component_1, component_1_sell_price, -component_1_sell_volume))
                component_2_orders.append(Order(component_2, component_2_sell_price, -component_2_sell_volume))
                component_3_orders.append(Order(component_3, component_3_sell_price, -component_3_sell_volume))

        # clearing
        clearing_threshold = 10
        if ((abs(current_difference - constant_differences_mean) < clearing_threshold) and (main_product_position != 0 )) or (self.clearing_flag == True):
            self.logger.log(f'clearing main product because current difference {current_difference} returned to constant difference mean {constant_differences_mean}', 'debug')
            
            if main_product_position > 0:
                # we wish to sell the main product and buy the components to return to the 0 position
                self.logger.log(f'selling main product because current position {main_product_position} is positive full', 'debug')
                main_product_sell_price = main_product_best_bid_price
                main_product_max_sell_volume = min(abs(main_product_position), main_product_best_bid_volume)
                
                # we must do the opposite to the components
                component_1_buy_price = components_best_ask_prices[component_1]
                component_1_max_buy_volume_normalized = min(abs(components_positions[component_1]), components_best_ask_volumes[component_1_buy_price])//main_product_to_components_ratios[component_1]
                component_2_buy_price = components_best_ask_prices[component_2]
                component_2_max_buy_volume_normalized = min(abs(components_positions[component_2]), components_best_ask_volumes[component_2_buy_price])//main_product_to_components_ratios[component_2]
                component_3_buy_price = components_best_ask_prices[component_3]
                component_3_max_buy_volume_normalized = min(abs(components_positions[component_3]), components_best_ask_volumes[component_3_buy_price])//main_product_to_components_ratios[component_3]
                
                normalized_min_volume = min([main_product_max_sell_volume, component_1_max_buy_volume_normalized, component_2_max_buy_volume_normalized, component_3_max_buy_volume_normalized])
                main_product_sell_volume = normalized_min_volume
                component_1_buy_volume = normalized_min_volume * main_product_to_components_ratios[component_1]
                component_2_buy_volume = normalized_min_volume * main_product_to_components_ratios[component_2]
                component_3_buy_volume = normalized_min_volume * main_product_to_components_ratios[component_3]

                self.logger.log(f'the selling volume is {main_product_sell_volume}, and the buying volumes are {[component_1_buy_volume, component_2_buy_volume, component_3_buy_volume]}', 'debug')

                if normalized_min_volume != 0:
                    main_product_orders.append(Order(main_product, main_product_sell_price, -main_product_sell_volume))
                    component_1_orders.append(Order(component_1, component_1_buy_price, component_1_buy_volume))
                    component_2_orders.append(Order(component_2, component_2_buy_price, component_2_buy_volume))
                    component_3_orders.append(Order(component_3, component_3_buy_price, component_3_buy_volume))
            
            if main_product_position < 0:
                # we wish to buy the main product and sell the components to return to the 0 position
                self.logger.log(f'buying main product because current position {main_product_position} is negative full', 'debug')
                main_product_buy_price = main_product_best_ask_price
                main_product_max_buy_volume =  min(abs(main_product_position), main_product_best_ask_volume)
                
                # we must do the opposite to the components
                component_1_sell_price = components_best_bid_prices[component_1]
                component_1_max_sell_volume_normalized = min(abs(components_positions[component_1]), components_best_bid_volumes[component_1_sell_price])//main_product_to_components_ratios[component_1]
                component_2_sell_price = components_best_bid_prices[component_2]
                component_2_max_sell_volume_normalized = min(abs(components_positions[component_2]), components_best_bid_volumes[component_2_sell_price])//main_product_to_components_ratios[component_2]
                component_3_sell_price = components_best_bid_prices[component_3]
                component_3_max_sell_volume_normalized = min(abs(components_positions[component_3]), components_best_bid_volumes[component_3_sell_price])//main_product_to_components_ratios[component_3]
                
                normalized_min_volume = min([main_product_max_buy_volume, component_1_max_sell_volume_normalized, component_2_max_sell_volume_normalized, component_3_max_sell_volume_normalized])
                main_product_buy_volume = normalized_min_volume
                component_1_sell_volume = normalized_min_volume * main_product_to_components_ratios[component_1]
                component_2_sell_volume = normalized_min_volume * main_product_to_components_ratios[component_2]
                component_3_sell_volume = normalized_min_volume * main_product_to_components_ratios[component_3]

                self.logger.log(f'the selling volume is {main_product_buy_volume}, and the buying volumes are {[component_1_sell_volume, component_2_sell_volume, component_3_sell_volume]}', 'debug')

                if normalized_min_volume != 0:
                    main_product_orders.append(Order(main_product, main_product_buy_price, main_product_buy_volume))
                    component_1_orders.append(Order(component_1, component_1_sell_price, -component_1_sell_volume))
                    component_2_orders.append(Order(component_2, component_2_sell_price, -component_2_sell_volume))
                    component_3_orders.append(Order(component_3, component_3_sell_price, -component_3_sell_volume))
            if normalized_min_volume < abs(main_product_position):
                self.clearing_flag = True
            else:
                self.clearing_flag = False
            
        # push in the current difference into differences window
        self.difference_between_main_product_and_components_window.push(current_difference)

        return main_product_orders, component_1_orders, component_2_orders, component_3_orders


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

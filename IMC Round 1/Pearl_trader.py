# This is the example program from the IMC official doc.

from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order


class Trader:

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {}

        # Iterate over all the keys (the available products) contained in the order depths
        for product in state.order_depths.keys():

            # Check if the current product is the 'PEARLS' product, only then run the order logic
            if product == 'PEARLS':

                # Retrieve the Order Depth containing all the market BUY and SELL orders for PEARLS
                order_depth: OrderDepth = state.order_depths[product]

                # Initialize the list of Orders to be sent as an empty list
                orders: list[Order] = []

                # Define a fair value for the PEARLS.
                # Note that this value of 1 is just a dummy value, you should likely change it!
                acceptable_price = 10

                # If statement checks if there are any SELL orders in the PEARLS market
                if len(order_depth.sell_orders) > 0:

                    # Sort all the available sell orders by their price,
                    # and select only the sell order with the lowest price
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_volume = order_depth.sell_orders[best_ask]

                    # Check if the lowest ask (sell order) is lower than the above defined fair value
                    if best_ask < acceptable_price:

                        # In case the lowest ask is lower than our fair value,
                        # This presents an opportunity for us to buy cheaply
                        # The code below therefore sends a BUY order at the price level of the ask,
                        # with the same quantity
                        # We expect this order to trade with the sell order
                        print("BUY", str(-best_ask_volume) + "x", best_ask)
                        orders.append(Order(product, best_ask, -best_ask_volume))

                # The below code block is similar to the one above,
                # the difference is that it finds the highest bid (buy order)
                # If the price of the order is higher than the fair value
                # This is an opportunity to sell at a premium
                if len(order_depth.buy_orders) != 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_volume = order_depth.buy_orders[best_bid]
                    if best_bid > acceptable_price:
                        print("SELL", str(best_bid_volume) + "x", best_bid)
                        orders.append(Order(product, best_bid, -best_bid_volume))

                # Add all the above orders to the result dict
                result[product] = orders

                # Return the dict of orders
                # These possibly contain buy or sell orders for PEARLS
                # Depending on the logic above
        return result
    
    def mean_reversal_calc(self, product:str, pos_lim:int,
                         best_bid:int, best_ask:int,
                         pos:float, history:list = [], 
                         WAP:float = 0,sigma:float = 0):
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
        true_price = 10000
        lot_size = 2
        orders = []
        min_profit_margin = 1 # minimum acceptable price
        alpha = 1 # alpha is a parameter that scale the bid/ask volume

        mr_ask_price = 0 # mr for mean reversal
        mr_bid_price = 0

        # Buy / Sell, with volume proportional to the profit margin
        buy_profit_margin = true_price - best_ask
        if buy_profit_margin > min_profit_margin:
            mr_ask_price =  best_ask
            mr_ask_vol = min(alpha*buy_profit_margin*lot_size, pos_lim-pos)
        sell_profit_margin = best_bid - true_price
        if sell_profit_margin > min_profit_margin:
            mr_bid_price = best_bid
            mr_bid_vol = -min(alpha*sell_profit_margin*lot_size, pos+pos_lim)

        # send the orders
        if mr_bid_price != 0:
            orders.append(Order(product, mr_bid_price, mr_bid_vol))
        if mr_ask_price != 0:
            orders.append(Order(product, mr_ask_price, mr_ask_vol))

        return orders

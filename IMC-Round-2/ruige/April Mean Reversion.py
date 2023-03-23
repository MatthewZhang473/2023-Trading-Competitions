from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import statistics as stats


class Trader:

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {}

        products = ['COCONUTS', 'PINA_COLADAS']
        product_limits = {'COCONUTS':600, 'PINA_COLADAS':300}

        # Iterate over all the keys (the available products) contained in the order depths
        for product in products:

            if product in state.position.keys():
                print(f'Current position of {product} are {state.position[product]}')

            # Retrieve the Order Depth containing all the market BUY and SELL orders for PEARLS
            order_depth: OrderDepth = state.order_depths[product]

            # Initialize the list of Orders to be sent as an empty list
            orders: list[Order] = []

            # Define a fair value for the PEARLS.
            buy_prices = list(order_depth.buy_orders.keys())
            sell_prices = list(order_depth.sell_orders.keys())
            acceptable_price = (sum(buy_prices) + sum(sell_prices))/(len(buy_prices) + len(sell_prices))

            # If statement checks if there are any SELL orders in the PEARLS market
            if len(list(order_depth.sell_orders.keys())) > 0:

                # Sort all the available sell orders by their price,
                # and select only the sell order with the lowest price
                best_ask = min(order_depth.sell_orders.keys())
                if product not in state.position.keys():
                    best_ask_volume = product_limits[product]
                else: 
                    best_ask_volume = abs(product_limits[product]-state.position[product])
                order_depth.sell_orders[best_ask]
                    
                # Check if the lowest ask (sell order) is lower than the above defined fair value
                if best_ask < acceptable_price:

                    # In case the lowest ask is lower than our fair value,
                    # This presents an opportunity for us to buy cheaply
                    # The code below therefore sends a BUY order at the price level of the ask,
                    # with the same quantity
                    # We expect this order to trade with the sell order
                    print("BUY", product, str(-best_ask_volume) + "x", best_ask)
                    orders.append(Order(product, best_ask, best_ask_volume))

            # The below code block is similar to the one above,
            # the difference is that it finds the highest bid (buy order)
            # If the price of the order is higher than the fair value
            # This is an opportunity to sell at a premium
            if len(list(order_depth.buy_orders.keys())) != 0:
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_volume = order_depth.buy_orders[best_bid]
                if best_bid > acceptable_price:
                    print("SELL", product, str(best_bid_volume) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_volume))

            # Add all the above orders to the result dict
            result[product] = orders
            

                # Return the dict of orders
                # These possibly contain buy or sell orders for PEARLS
                # Depending on the logic above
        return result
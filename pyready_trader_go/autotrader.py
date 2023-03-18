# Copyright 2021 Optiver Asia Pacific Pty. Ltd.
#
# This file is part of Ready Trader Go.
#
#     Ready Trader Go is free software: you can redistribute it and/or
#     modify it under the terms of the GNU Affero General Public License
#     as published by the Free Software Foundation, either version 3 of
#     the License, or (at your option) any later version.
#
#     Ready Trader Go is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public
#     License along with Ready Trader Go.  If not, see
#     <https://www.gnu.org/licenses/>.


#########Customized import for local use, MUST remove for submission #########
#import csv
##############################################################################

import asyncio
import itertools
import numpy as np


from typing import List

from ready_trader_go import BaseAutoTrader, Instrument, Lifespan, MAXIMUM_ASK, MINIMUM_BID, Side


LOT_SIZE = 33
POSITION_LIMIT = 100
TICK_SIZE_IN_CENTS = 100 # MAtt: Tick size is the minimum change in price allowed

# Matthew: ensures bid / ask are at least 1 tick away from the minimum / maximum,
# // is integer division (4 //3 = 1)
MIN_BID_NEAREST_TICK = (MINIMUM_BID + TICK_SIZE_IN_CENTS) // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS
MAX_ASK_NEAREST_TICK = MAXIMUM_ASK // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS

ETF_HISTORIES = []
FUTURE_HISTORIES = []

class AutoTrader(BaseAutoTrader):
    """Example Auto-trader.

    When it starts this auto-trader places ten-lot bid and ask orders at the
    current best-bid and best-ask prices respectively. Thereafter, if it has
    a long position (it has bought more lots than it has sold) it reduces its
    bid and ask prices. Conversely, if it has a short position (it has sold
    more lots than it has bought) then it increases its bid and ask prices.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, team_name: str, secret: str):
        """Initialise a new instance of the AutoTrader class."""
        super().__init__(loop, team_name, secret)
        #Matthew: this order_ids is an iterator, we uses it to generate new (unique) order ids.
        self.order_ids = itertools.count(1)
        # Matthew: The seld.bids are the bid orders you currently have on the market,
        # it is a set, so you cannot change the element inside, but you can remove and put in new elements
        self.bids = set()
        self.asks = set()
        # Matt: self.ask_id is used to give an id to all the orders you send later,
        # once sent, the id will be stored in self.asks
        self.ask_id = self.ask_price = self.bid_id = self.bid_price = self.position = 0
        self.TRADING_HISTORIES = []
        self.first_it = True
        self.prev_pos = 0

    def on_error_message(self, client_order_id: int, error_message: bytes) -> None:
        """Called when the exchange detects an error.

        If the error pertains to a particular order, then the client_order_id
        will identify that order, otherwise the client_order_id will be zero.
        """
        self.logger.warning("error with order %d: %s", client_order_id, error_message.decode())
        if client_order_id != 0 and (client_order_id in self.bids or client_order_id in self.asks):
            self.on_order_status_message(client_order_id, 0, 0, 0)

    def on_hedge_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when one of your hedge orders is filled.

        The price is the average price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        self.logger.info("received hedge filled for order %d with average price %d and volume %d", client_order_id,
                         price, volume)

    def on_order_book_update_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                                     ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically to report the status of an order book.

        The sequence number can be used to detect missed or out-of-order
        messages. The five best available ask (i.e. sell) and bid (i.e. buy)
        prices are reported along with the volume available at each of those
        price levels.
        """
        self.logger.info("received order book for instrument %d with sequence number %d", instrument,
                         sequence_number)
        if instrument == Instrument.FUTURE:

            # We don't operate for future orderbook

            ################ Matthew's code for storing the price changes ################
            curr_time = self.event_loop.time()
            price_log = HistoryLog(instrument=instrument, time=curr_time,
                                   ask_prices=ask_prices, ask_volumes=ask_volumes,
                                   bid_prices=bid_prices, bid_volumes=bid_volumes)
            #FUTURE_HISTORIES.append(price_log)
            self.logger.info("(FUTURE) Added price info to log. Time = %.1f, WAP = %.2f", price_log.time, price_log.WAP)

            # write the time, best_ask, best_bid, WAP into the csv file
            #write2file(instrument=instrument, row_list=[curr_time, ask_prices[0], bid_prices[0], price_log.WAP])

            ##############################################################################

        elif instrument == Instrument.ETF:

            # We recalculate the order when receiving the ETF orderbook

            ################ Matthew's code for storing the price changes ################
            curr_time = self.event_loop.time()
            price_log = HistoryLog(instrument=instrument, time=curr_time,
                                   ask_prices=ask_prices, ask_volumes=ask_volumes,
                                   bid_prices=bid_prices, bid_volumes=bid_volumes)
            #ETF_HISTORIES.append(price_log)
            self.logger.info("(ETF) Added price info to log. Time = %.1f, WAP = %.2f, total_WAP = %.2f, best_bid = %d, best_ask = %d",price_log.time, price_log.WAP, price_log.total_WAP, bid_prices[0], ask_prices[0])

            # write the time, best_ask, best_bid, WAP into the csv file
            # write2file(instrument=instrument, row_list=[curr_time, ask_prices[0], bid_prices[0], price_log.WAP])

            ##############################################################################

            ################ Payton's code for recording the WAP_time ##################
            # WAP_TIME_HISTORIES = []
            # for trade in self.TRADING_HISTORIES:
            #     WAP_TIME_HISTORIES.append(trade.WAP_time)
            # self.TRADING_HISTORIES = []
            # self.logger.info("(TRADE LOG) %d trade ticks occuered between orderbook updates and is recorded.", len(WAP_TIME_HISTORIES))
            #write2file(instrument=instrument, row_list=[price_log.time, ask_prices[0], bid_prices[0], price_log.WAP] + WAP_TIME_HISTORIES)
            ##############################################################################
            
            ############# Operate when orderbook of ETF updates ############

            # Matthew: Very hard code
            ######skip first iteration#########
            # if self.first_it:
            #     self.first_it = False
            #     return
            ###################################

            # Previous calculation for volatlity
            # if len(ETF_HISTORIES) >= 50:
            #     partial_sigma = FindVolatility(ETF_HISTORIES[-5:])
            #     total_sigma = FindVolatility(ETF_HISTORIES[-50:])
            # else:
            #     partial_sigma = 1
            #     total_sigma = 1
            # sigma_ratio = partial_sigma / total_sigma if total_sigma != 0 else 3
            # #write2file(instrument=instrument, row_list=[sigma_ratio,partial_sigma,total_sigma])

            self.OrderCalc(WAP=price_log.WAP,
                           volatility=1, total_volatility=1, volatility_ratio=1, # Matt:we are not using volatility at the moment.
                           tick_size_in_cents=TICK_SIZE_IN_CENTS,
                           best_ask=ask_prices[0], best_bid=bid_prices[0])

            ################################################################

    def on_order_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when one of your orders is filled, partially or fully.

        The price is the price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        self.logger.info("received order filled for order %d with price %d and volume %d", client_order_id,
                         price, volume)
        if client_order_id in self.bids:
            self.position += volume
            self.send_hedge_order(next(self.order_ids), Side.ASK, MIN_BID_NEAREST_TICK, volume)
        elif client_order_id in self.asks:
            self.position -= volume
            self.send_hedge_order(next(self.order_ids), Side.BID, MAX_ASK_NEAREST_TICK, volume)

    def on_order_status_message(self, client_order_id: int, fill_volume: int, remaining_volume: int,
                                fees: int) -> None:
        """Called when the status of one of your orders changes.

        The fill_volume is the number of lots already traded, remaining_volume
        is the number of lots yet to be traded and fees is the total fees for
        this order. Remember that you pay fees for being a market taker, but
        you receive fees for being a market maker, so fees can be negative.

        If an order is cancelled its remaining volume will be zero.
        """
        self.logger.info("received order status for order %d with fill volume %d remaining %d and fees %d",
                         client_order_id, fill_volume, remaining_volume, fees)
        if remaining_volume == 0:
            if client_order_id == self.bid_id:
                self.bid_id = 0
            elif client_order_id == self.ask_id:
                self.ask_id = 0

            # It could be either a bid or an ask
            self.bids.discard(client_order_id)
            self.asks.discard(client_order_id)

    def on_trade_ticks_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                               ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically when there is trading activity on the market.

        The five best ask (i.e. sell) and bid (i.e. buy) prices at which there
        has been trading activity are reported along with the aggregated volume
        traded at each of those price levels.

        If there are less than five prices on a side, then zeros will appear at
        the end of both the prices and volumes arrays.
        """
        self.logger.info("received trade ticks for instrument %d with sequence number %d", instrument,
                         sequence_number)

        ####### Payton's code to record trading info during the time interval between order book updated ######       
        # if instrument == Instrument.ETF:
        #         curr_time = self.event_loop.time()
        #         trade_price_log = TradingLog(instrument=instrument, time=curr_time,
        #                             ask_prices=ask_prices, ask_volumes=ask_volumes,
        #                             bid_prices=bid_prices, bid_volumes=bid_volumes)
        #         self.TRADING_HISTORIES.append(trade_price_log)
        ########################################################################################################

    def OrderCalc(self, WAP: float, 
                  volatility: float, total_volatility: float,volatility_ratio: float,
                  tick_size_in_cents: int,best_ask:float, best_bid:float):
            """
            Given the current position and market parameters, amend the current orders.
            """
            market_spread = (best_ask - best_bid)/2

            ## Dont send order if WAP is zero ##
            if WAP == 0:
                return
            ####################################

            ##### Cancel ordered when the spread is too small #####
            # beta = 2
            # if market_spread < beta*tick_size_in_cents:
            #     if self.bid_id != 0:
            #         self.send_cancel_order(self.bid_id)
            #         self.bid_id = 0
            #     if self.ask_id != 0:
            #         self.send_cancel_order(self.ask_id)
            #         self.ask_id = 0
            #     return
            #######################################################

            ######### Use WAP +- delta * spread  to determine the price change ###########
            delta = 1.2# manually specify this parameter delta
            WAP_tick = (WAP // tick_size_in_cents) * tick_size_in_cents # try to make WAP times as tick_size_in_cents
            # self.logger.info("(Price Calc) WAP_tick = %f, best_bid = %f, best_ask = %f", WAP_tick, best_bid, best_ask)
            price_change = (delta * market_spread// tick_size_in_cents) * tick_size_in_cents
            price_change = max(2*tick_size_in_cents, price_change) # ensure the price change is not too small

            ##############################################################################

            ######## Change the delta by market spread ############
            beta = 2
            if self.prev_pos == 0:
                self.prev_pos = self.position
            else:
                # position change from the last orderbook update
                pos_change = self.position - self.prev_pos
                self.prev_pos = self.position
                if market_spread < beta*tick_size_in_cents:
                    self.logger.info("(Protect) Position change = %d, enlarge spread to %d", pos_change, price_change/tick_size_in_cents + 5)
                    price_change += 5*tick_size_in_cents
            #################################################################

            ##### Sigmoid Inventory Control######
            alpha = 1 * market_spread
            offset = 30
            scale = 5
            if self.position > 0:
                normalized_pos = (self.position - offset)/scale
                WAP_tick -= (alpha * (sigmoid(normalized_pos))// tick_size_in_cents) * tick_size_in_cents
                self.logger.info("(INVENT) Position = %d, Price change = %d",self.position,
                            -(alpha*(sigmoid(normalized_pos))// tick_size_in_cents) * tick_size_in_cents)
            if self.position < 0:
                normalized_pos = -(self.position + offset)/scale
                WAP_tick += (alpha * (sigmoid(normalized_pos))// tick_size_in_cents) * tick_size_in_cents
                self.logger.info("(INVENT) Position = %d, Price change = %d",self.position,
                            (alpha*(sigmoid(normalized_pos))// tick_size_in_cents) * tick_size_in_cents)
            #############################


            ##### calculate total price######
            new_bid_price = WAP_tick - price_change
            new_ask_price = WAP_tick + price_change
            #################################

            ########### Send the order while maintaining position limit #########
            if self.bid_id != 0 and new_bid_price not in (self.bid_price, 0):
                self.send_cancel_order(self.bid_id)
                self.bid_id = 0
            if self.ask_id != 0 and new_ask_price not in (self.ask_price, 0):
                self.send_cancel_order(self.ask_id)
                self.ask_id = 0
            if self.bid_id == 0 and new_bid_price != 0 and self.position < POSITION_LIMIT - LOT_SIZE:
                self.bid_id = next(self.order_ids)
                self.bid_price = new_bid_price
                self.logger.info("(Bid) order inserted, price = %f (%s)", new_bid_price, type(new_bid_price))
                self.send_insert_order(self.bid_id, Side.BUY, int(new_bid_price), LOT_SIZE, Lifespan.GOOD_FOR_DAY)
                self.bids.add(self.bid_id)
            if self.ask_id == 0 and new_ask_price != 0 and self.position > -POSITION_LIMIT + LOT_SIZE:
                self.ask_id = next(self.order_ids)
                self.ask_price = new_ask_price
                self.logger.info("(Ask) order inserted, price = %f (%s)", new_ask_price, type(new_ask_price))
                self.send_insert_order(self.ask_id, Side.SELL, int(new_ask_price), LOT_SIZE, Lifespan.GOOD_FOR_DAY)
                self.asks.add(self.ask_id)

            #####################################################################

############# Payton's Class for storing trading during time interval ################

class TradingLog:
    '''A class that records the trades on market in small time interval.'''
    def __init__(self, instrument: int,time,
                 ask_prices: List[int],ask_volumes: List[int], 
                 bid_prices: List[int], bid_volumes: List[int]):
        self.instrument = instrument
        self.time = time
        self.ask_prices = ask_prices
        self.ask_volumes = ask_volumes
        self.bid_prices = bid_prices
        self.bid_volumes = bid_volumes
        self.WAP_time = self.find_WAP_time() # WAP_time is the trade based WAP
    
    def find_WAP_time(self):
        calc =[]
        for bid_p, ask_p, bid_v, ask_v in zip(self.bid_prices,self.ask_prices,self.bid_volumes,self.ask_volumes):
            result = bid_p * bid_v+ ask_p * ask_v
            calc.append(result)
        result_time = sum(calc)
        bid_volumes_time = sum(self.bid_volumes) 
        ask_volumes_time = sum(self.ask_volumes)
        return  result_time / (bid_volumes_time + ask_volumes_time)    

######################################################################################
    
############# Matthew's Class for storing prices #######################

class HistoryLog:
    """
    A class used to store the Weigted-average price at a given time.
    """
    def __init__(self, instrument: int, time,
                 ask_prices: List[int], ask_volumes: List[int],
                 bid_prices: List[int], bid_volumes: List[int]):
        self.instrument = instrument
        self.time = time
        self.ask_prices = ask_prices
        self.ask_volumes = ask_volumes
        self.bid_prices = bid_prices
        self.bid_volumes = bid_volumes
        self.WAP =  self.find_WAP()# The WAP is the volume weigted average price, it is an index to show the market price
        self.total_WAP = self.find_total_WAP()

    def find_WAP(self):
        best_ask_price = self.ask_prices[0]
        best_ask_vol = self.ask_volumes[0]
        best_bid_price = self.bid_prices[0]
        best_bid_vol = self.bid_volumes[0]

        # if there is no order in the market
        if best_ask_price == 0 and best_bid_price == 0:
            return 0
        else:
            return (best_ask_price * best_bid_vol + best_bid_price * best_ask_vol) / (best_ask_vol + best_bid_vol)

    def find_total_WAP(self):
        """
        Return the total_WAP, as numpy float.
        Total_WAP is the weighted sum of all the orders in the orderbook, 
        it reveals information of the total demand & supply.
        """
        total_volume = sum(self.ask_volumes) + sum(self.bid_volumes)
        if total_volume == 0:
            return 0
        else:
            return (np.dot(self.ask_prices, self.ask_volumes) + np.dot(self.bid_prices,self.bid_volumes)) / total_volume 


##############################################################################




################Matthew's code for write to csv##################

# def write2file(instrument: int, row_list):
#     if instrument == 0:
#         csv_name = 'Future.csv'
#     elif instrument == 1:
#         csv_name = 'ETF.csv'
#     else:
#         raise ValueError("Wrong instrument number = %d", instrument)

#     with open(csv_name, 'a', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(row_list)

##################################################################

###########Volatility Calculator############
def FindVolatility(history: List[HistoryLog]):
    """
    Calculate the volatility of a section of history prices.
    """
    n = len(history)
    sum_log_returns = 0
    for i in range(1,n):
        # avoid division by zero that might occur at the beginning.
        log_return = np.log(history[i].WAP/history[i-1].WAP) if history[i-1].WAP != 0 else 0
        sum_log_returns += log_return ** 2
    sigma = np.sqrt(sum_log_returns / n)
    return sigma

###########################################

#######sigmoid#########

def sigmoid(x:float):
    """
    Sigmoid function
    """
    y = 1 / (1+np.exp(-x))
    return y
######################
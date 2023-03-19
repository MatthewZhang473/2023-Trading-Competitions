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
import asyncio
import itertools
import numpy as np
import scipy as sp

from typing import List

from ready_trader_go import BaseAutoTrader, Instrument, Lifespan, MAXIMUM_ASK, MINIMUM_BID, Side


LOT_SIZE = 10
POSITION_LIMIT = 100
TICK_SIZE_IN_CENTS = 100
MIN_BID_NEAREST_TICK = (
    MINIMUM_BID + TICK_SIZE_IN_CENTS) // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS
MAX_ASK_NEAREST_TICK = MAXIMUM_ASK // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS


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
        self.order_ids = itertools.count(1)
        self.bids = set()
        self.asks = set()
        self.ask_id = self.ask_price = self.bid_id = self.bid_price = self.position = 0

        self.Inst_Averages = []
        self.Inst_Spreads = []
        self.Inst_Ratios = []
        self.ETF_Price_list = []
        self.Future_Price_list = []

        self.WindowSize = 50
        self.RollingAverage = []
        self.RollingStandardDeviation = []
        self.StockAverage = []
        self.StockSTD = []
        self.UpperBollingerBand = []
        self.LowerBollingerBand = []
        self.NumberOfSTD = 1  # For calculating Bollinger bands

        self.isTrendUp = False
        self.isTrendDown = False
        self.trend_slope_thresh = 20
        self.slope_window = 50
        self.trend_multiplier = 1.5

    def on_error_message(self, client_order_id: int, error_message: bytes) -> None:
        """Called when the exchange detects an error.

        If the error pertains to a particular order, then the client_order_id
        will identify that order, otherwise the client_order_id will be zero.
        """
        self.logger.warning("error with order %d: %s",
                            client_order_id, error_message.decode())
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

        self.Track_Data(instrument, ask_prices[0], bid_prices[0])
        self.CalculateRolling()
        # self.CheckTrend()
        self.BollingerBand()
        self.TradeETFUsingBollinger(
            ask_prices, bid_prices, ask_volumes, bid_volumes)
        # if instrument == Instrument.FUTURE:
        #     price_adjustment = - \
        #         (self.position // LOT_SIZE) * TICK_SIZE_IN_CENTS
        #     new_bid_price = bid_prices[0] + \
        #         price_adjustment if bid_prices[0] != 0 else 0
        #     new_ask_price = ask_prices[0] + \
        #         price_adjustment if ask_prices[0] != 0 else 0

        #     if self.bid_id != 0 and new_bid_price not in (self.bid_price, 0):
        #         self.send_cancel_order(self.bid_id)
        #         self.bid_id = 0
        #     if self.ask_id != 0 and new_ask_price not in (self.ask_price, 0):
        #         self.send_cancel_order(self.ask_id)
        #         self.ask_id = 0

        #     if self.bid_id == 0 and new_bid_price != 0 and self.position < POSITION_LIMIT:
        #         self.bid_id = next(self.order_ids)
        #         self.bid_price = new_bid_price
        #         self.send_insert_order(
        #             self.bid_id, Side.BUY, new_bid_price, LOT_SIZE, Lifespan.GOOD_FOR_DAY)
        #         self.bids.add(self.bid_id)

        #     if self.ask_id == 0 and new_ask_price != 0 and self.position > -POSITION_LIMIT:
        #         self.ask_id = next(self.order_ids)
        #         self.ask_price = new_ask_price
        #         self.send_insert_order(
        #             self.ask_id, Side.SELL, new_ask_price, LOT_SIZE, Lifespan.GOOD_FOR_DAY)
        #         self.asks.add(self.ask_id)

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
            self.send_hedge_order(next(self.order_ids),
                                  Side.ASK, MIN_BID_NEAREST_TICK, volume)
        elif client_order_id in self.asks:
            self.position -= volume
            self.send_hedge_order(next(self.order_ids),
                                  Side.BID, MAX_ASK_NEAREST_TICK, volume)

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

    def Track_Data(self, instrument, ask_price, bid_price):
        if ask_price == 0 or bid_price == 0:
            return
        else:
            ratio = 1
            midpoint_val = (ask_price + bid_price) / 2
            Average_stock = 0
            Stock_Spread = 0
            if instrument == Instrument.FUTURE:
                self.Future_Price_list.append(midpoint_val)
                if not self.ETF_Price_list:
                    ratio = 1
                    self.ETF_Price_list.append(midpoint_val)
                    Average_stock = midpoint_val
                    Stock_Spread = 0
                else:
                    self.ETF_Price_list.append(self.ETF_Price_list[-1])
                    ratio = self.ETF_Price_list[-1] / midpoint_val
                    Average_stock = (
                        self.ETF_Price_list[-1] + midpoint_val) / 2
                    Stock_Spread = self.ETF_Price_list[-1] - midpoint_val
            else:
                self.ETF_Price_list.append(midpoint_val)
                if not self.Future_Price_list:
                    ratio = 1
                    self.Future_Price_list.append(midpoint_val)
                    Average_stock = midpoint_val
                    Stock_Spread = 0
                else:
                    self.Future_Price_list.append(self.Future_Price_list[-1])
                    ratio = midpoint_val / self.Future_Price_list[-1]
                    Average_stock = (
                        self.Future_Price_list[-1] + midpoint_val) / 2
                    Stock_Spread = midpoint_val - self.Future_Price_list[-1]
            self.Inst_Ratios.append(ratio)
            self.Inst_Averages.append(Average_stock)
            self.Inst_Spreads.append(Stock_Spread)
            self.logger.info(
                "Ratio: %f, Average_stock: %f, Stock_Spread: %f", ratio, Average_stock, Stock_Spread)

    def CalculateRolling(self):
        if len(self.Inst_Ratios) == 0:
            # expected when programme first begins
            self.logger.error("Empty Inst_Ratios list")
            return
        Inst_Ratios_nparray = np.array(self.Inst_Ratios)
        Inst_Averages_nparray = np.array(self.Inst_Averages)
        if len(Inst_Ratios_nparray) > self.WindowSize:
            Inst_Ratios_nparray = Inst_Ratios_nparray[-self.WindowSize:]
            Inst_Averages_nparray = Inst_Averages_nparray[-self.WindowSize:]
        self.RollingAverage.append(np.average(Inst_Ratios_nparray))
        self.RollingStandardDeviation.append(np.std(Inst_Ratios_nparray))
        self.StockAverage.append(np.average(Inst_Averages_nparray))
        self.StockSTD.append(np.std(Inst_Averages_nparray))
        self.logger.info("StockAverage: %f", self.StockAverage[-1])
        # self.logger.info(
        #     "RollingAverage: %f, RollingStandardDeviation: %f, WindowSize: %d", self.RollingAverage[-1], self.RollingStandardDeviation[-1], self.WindowSize)

    def CheckTrend(self):
        if not self.StockAverage:
            return
        Stock_average_nparray = np.array(self.StockAverage)
        if len(self.StockAverage) > self.slope_window:
            Stock_average_nparray = Stock_average_nparray[-self.slope_window:]
        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(np.array(range(len(Stock_average_nparray))),
                                                                          Stock_average_nparray)
        if slope > self.trend_slope_thresh:
            self.isTrendUp = True
            self.isTrendDown = False
        elif slope < -self.trend_slope_thresh:
            self.isTrendUp = False
            self.isTrendDown = True
        else:
            self.isTrendUp = False
            self.isTrendDown = False
        self.logger.info("Slope: %f, isTrendUp: %s, isTrendDown: %s",
                         slope, self.isTrendUp, self.isTrendDown)

    def TradeETFUsingBollinger(self, ask_prices, bid_prices, ask_volumes, bid_volumes):
        if not self.Inst_Ratios or not self.UpperBollingerBand or not self.LowerBollingerBand:
            self.logger.info("Empty Trader list")
            return
        self.logger.info("Ratio list exists")
        ratio = self.Inst_Ratios[-1]
        # self.logger.info("Ratio: %f, UpperBollingerBand: %f, LowerBollingerBand: %f",
        #                  ratio, self.UpperBollingerBand, self.LowerBollingerBand)
        if (ratio > self.UpperBollingerBand[-1]):
            prob_ratio = sp.stats.norm.cdf(
                -(ratio-self.RollingAverage[-1])/self.RollingStandardDeviation[-1])
            bollinger_prob = sp.stats.norm.cdf(-self.NumberOfSTD)
            sell_bias = prob_ratio/bollinger_prob
            # if self.isTrendDown == True:
            #     sell_bias = (1-sell_bias) * self.trend_multiplier + sell_bias
            # maybe want to put this slightly higher??
            self.ask_price = ask_prices[0]
            SellVolume = round((sell_bias)
                               * (self.position - (-POSITION_LIMIT)))
            self.ask_id = next(self.order_ids)
            self.send_insert_order(
                self.ask_id, Side.SELL, self.ask_price, SellVolume, Lifespan.FILL_AND_KILL)
            self.asks.add(self.ask_id)
        elif (ratio < self.LowerBollingerBand[-1]):
            prob_ratio = sp.stats.norm.cdf(
                (ratio-self.RollingAverage[-1])/self.RollingStandardDeviation[-1])
            bollinger_prob = sp.stats.norm.cdf(-self.NumberOfSTD)
            buy_bias = prob_ratio/bollinger_prob
            # if self.isTrnendUp == True:
            #     prob_ratio = (1-buy_bias) * self.trend_multiplier + buy_bias
            # maybe want to put this slightly lower??
            self.bid_price = bid_prices[0]
            BuyVolume = round((buy_bias) *
                              (POSITION_LIMIT - self.position))
            self.bid_id = next(self.order_ids)
            self.send_insert_order(
                self.bid_id, Side.BUY, self.bid_price, BuyVolume, Lifespan.FILL_AND_KILL)
            self.bids.add(self.bid_id)

    # Experimental Trend Trading Strategy

    def Min_Trend_Trade(self, ask_prices, bid_prices):
        if self.isTrendUp == True:
            self.bid_price = bid_prices[0]
            BuyVolume = LOT_SIZE
            if self.position + BuyVolume > POSITION_LIMIT:
                BuyVolume = POSITION_LIMIT - self.position
            if BuyVolume != 0:
                self.bid_id = next(self.order_ids)
                self.send_insert_order(
                    self.bid_id, Side.BUY, self.bid_price, BuyVolume, Lifespan.FILL_AND_KILL)
                self.bids.add(self.bid_id)

        elif self.isTrendDown == True:
            self.ask_price = ask_prices[0]
            SellVolume = LOT_SIZE
            if self.position - SellVolume < -POSITION_LIMIT:
                SellVolume = -POSITION_LIMIT - self.position
            if SellVolume != 0:
                self.ask_id = next(self.order_ids)
                self.send_insert_order(
                    self.ask_id, Side.SELL, self.ask_price, SellVolume, Lifespan.FILL_AND_KILL)
                self.asks.add(self.ask_id)

    def BollingerBand(self):
        if len(self.Inst_Ratios) == 0:
            # expected when programme first begins
            self.logger.error("Empty Inst_Ratios list")
            return
        else:
            self.UpperBollingerBand.append(
                self.RollingAverage[-1] + self.NumberOfSTD * self.RollingStandardDeviation[-1])
            self.LowerBollingerBand.append(
                self.RollingAverage[-1] - self.NumberOfSTD * self.RollingStandardDeviation[-1])
            self.logger.info(
                "UpperBBand: %f, LowerBBand: %f", self.UpperBollingerBand[-1], self.LowerBollingerBand[-1])

    # Modelling ARIMA model work

    def pearson_correlation(self, x, y):
        return np.mean((x - np.mean(x)) * (y - np.mean(y))) / np.std(x) / np.std(y)

    def acf(self, x, lag=10):
        if len(x) < lag:
            return
        else:
            self.logger.info("ACF coefficients")
            self.logger.info(np.array(
                [1]+[self.pearson_correlation(x[:-i], x[i:]) for i in range(1, lag+1)]))

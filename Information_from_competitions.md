# Optiver

## Objective: 

"make market"

By: write an autotrader to price the ETF, execute trades, and manage risk.

## Things need to check:
1. Learn about the various market elements and fundamentals of market making.

2. The code provided: a basic Autotrader, an exchange simulator, and some test data.
3. Data provided: historical time-series order and trade data, which you can use to verify your pricing and trading algorithm
4. *The Glossay on the [Optiver page](https://readytradergo.optiver.com/how-to-play/)

## Code implementation information:
- Backtrader: a popular Python library for backtesting trading strategies.

## Glossay check list:

- Aggressor: Who identify bid / sell in market and trade with it. (It also removes liquidity of the market)
- Market maker: As apposed to aggressor, market maker actively quote two-sided markets(bid / ask) in the market. They aim to earn profit from **bid-ask-spread** (difference between bid price and ask price), and provide liquidity to the market.

- Counterparty: the seller is the counter party of the buyer

- Hedge:  Uses financial instruments or market strategies to offset the risk of any adverse price movements. 

    ------------------
- Order: An order is a set of instructions to a broker to buy or sell an asset on a trader's behalf. **The execution of an order occurs when it gets filled**, not when the investor places it.

- Order Execution: refers to messages between you and the exchange relating to your orders; e.g. you place an order by sending an “order insert” execution message, you are notified that you have traded via a “order filled” execution message etc.

- Lots: number of units / order volume.

- Hit: means to “sell” – e.g. “hitting the bid” means crossing the spread to sell an existing buy order.

- Lift: Buy.

- Order types:
    1. **market order**: trade now! (Optiver recommends that you never use a market order.)
    2. **limit order**: buy / sell at a specific price or higher
    3. stop order: sell when price is below a level
    4. ...
- **Ask**:

    i. a SELL order, or     
    ii. refers to the price of the lowest sell order in the market (i.e. the ask is $10.13).

- **Bid**: 
    i. a BUY order, or 
    ii. refers to the price of the highest buy order in the market (i.e. the bid is $10.12).

    -------------------

- Exchange Traded Fund (**ETF**): a popular tool for investors to access a range of markets and asset classes. (e.g.S&P 500) 

- Financial instrument: a tradable asset, or a negotiable item, such as a ETF, loan, bond, derivative contract...

- Derivative (衍生产品): a type of financial contract whose value is dependent on an underlying asset

- Lock / Option: There are two classes of derivative products: "lock" and "option." Lock products (e.g., futures, forwards, or swaps) bind the respective parties from the outset to the agreed-upon terms over the life of the contract. Option products (e.g., stock options) offer the holder the right, but not the obligation, to buy or sell the underlying asset or security at a specific price on or before the option's expiration date. 

- Futures (期货)： (is a derivative) A contract that **obligate** the owner (buyer) to buy a certain amount of goods with a predetermined future date and price. The future contracts can be "cash-settled" by closing the contract before expiary date.

- options(期权): similar to future, the owner of the options has **no obligation** to buy or sell. Imagine an investor owns 100 shares of a stock worth $50 per share. They believe the stock's value will rise in the future. However, this investor is concerned about potential risks and decides to hedge their position with an option. The investor could buy a **put option** that gives them the right to sell 100 shares of the underlying stock for $50 per share—known as the strike price—until a specific day in the future—known as the expiration date.

- swap: another type of derivative,often used to exchange one kind of cash flow with another. (E.g., fixed/variable intereste rate exchange)

    ---

- index:  a measurement of the value of a particular segment of the market, used for monitoring returns and benchmarking the performance of portfolios. For example, the EURO STOXX 50 is an index comprised of 50 of the largest and most liquid stocks in Europe.

- index futures: Use the idea of futures to trade and speculate index prices.

    ---

- Liquidity: How quickly and easily an asset can be bought or sold in large quantities for a fair price.

- Volatility: a statistical measure of the dispersion of returns for a given security or market index. In most cases, the higher the volatility, the riskier the security. 

    ---
- Market Information: refers to public data disseminated by the matching engine about the state of the market, such as what orders are in the book and what trades have occurred.

- Order book: a collection of orders arranged according to the rule of price-time priority.

## Some confusing rules:

1. **Positions** will be limited to 100 lots: traders are allowed to hold a maximum of 100 lots of a particular asset at any given time during the trading round.
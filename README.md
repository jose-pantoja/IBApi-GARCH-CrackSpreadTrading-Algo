# Purpose
Crack spread trading strategy algorithm based on GARCH volatility implemented in Python with IBApi.

# Requirements
- Python 3.6 or later
- Interactive Brokers API access credentials for TWS or IB Gateway 
- IB live or paper trading account

# Key Concepts and How To Use
- Simple 3:2:1 crack spread trading strategy with long and or short futures positions
- Crack spreads, z-scores, and GARCH volatility with optimized p and q are automatically and periodically calculated from requested historical data
- Manually input thresholds for z-score, low/high volatility, and adjusted volatility
- Open crack spread position (either) from trigger built on crack spreads, z-score, any open position and their length, returns, and GARCH volatility: comparison between last returned crack spread and GARCH volatility
- Close crack spread position (either) from trigger built on crack spreads, z-score, any open position and their length, returns, GARCH volatility, and low/high volatility threshold: comparison between last returned crack spread and product of GARCH volatility and low/high volatility threshold
- Broad error handling and logging of unexpected events

# Helpful Resources
- [Forecasting energy market volatility using GARCH models: Can multivariate models beat univariate models?](https://www.sciencedirect.com/science/article/abs/pii/S0140988312000540)
- [An empirical model comparison for valuing crack spread options](https://www.sciencedirect.com/science/article/abs/pii/S0140988315001917)

# Important Considerations
- Implement an alert system via Telegram that sends notification when certain conditions are met (i.e. order triggered)
- Run trading algo in a cloud service such as Microsoft Azure and Docker for continuous connection
- Seasonal periods may be better incorporated by doing items such as widening z-score threshold to trigger trades where z-score deviates from the historical average by a large margin in certain months
- Deal with data in a more exhaustive manner by incorporating a validation set
- Comprehensively test strategy using historical data as well as backtesting methods for the purpose of maximizing performance and risk
- Housekeeping items can be additionally incorporated to keep track of trading strategy parameters such as ibapi EWrapper's managedAccounts
- Stay inform of external factors that the trading algo does not take into account for possibly manual intervention

# Financial Disclaimer
Listing of securities does not entail soliticiation to trade. Trading strategy alogrithm has not been tested to the point of being provable of producing profit. Ultimately, I am not liable for any undesired results of your trades.

# Afterword
This crack spread trading strategy algo is created with hypothethical ideal parameters and conditions, ultimately there is still lots to fine-tune; however, this algo is a good starting place for someone to implement an intuitive crack spread trading strategy.

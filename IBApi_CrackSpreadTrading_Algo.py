# Import necessary libraries and packages
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import *
import pandas as pd
import numpy as np
import arch
import threading
import time
from sklearn.model_selection import TimeSeriesSplit
import logging
import datetime

# Define class that inherits from EWrapper and EClient
class IBapi(EWrapper, EClient):
    def __init__(self):
        # Initialize EClient superclass
        EClient.__init__(self, self)

        # Initialize data storage
        self.df_gasoline = pd.DataFrame()
        self.df_heating_oil = pd.DataFrame()
        self.df_crude_oil = pd.DataFrame()

        # Initialize rollover dates of contracts
        self.gasoline_rollover_date = None
        self.heating_oil_rollover_date = None
        self.crude_oil_rollover_date = None

        # Initialize contract objects of crack spread commodities
        self.gasoline_contract = None
        self.heating_oil_contract = None
        self.crude_oil_contract = None

        # Initialize position and order tracking variables
        self.is_position_open = False
        self.bar_count = 0
        self.bardata = {}
        self.nextOrderId = None
        self.opening_crack_spread = None
        self.opening_action = None
        self.open_orders = []
        self.current_pnl = 0.0
        self.cumulative_pnl = 0.0
        self.cumulative_paper_pnl = 0.0

        # Initialize logger for logging trading moves
        self.logger = logging.getLogger("CrackSpreadStrategy")
        self.logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # Initialize rollover and contract expiry dates
        self.calculate_rollover_dates()
        self.calculate_contract_expiry_dates()

        # Initialize expired contract prices dictionary
        self.expired_contract_prices = {
            "CL": None,
            "RB": None,
            "HO": None
        }

    def calculate_rollover_dates(self):
        """
        Calculate rollover dates of contracts for crack spread commodities given now and 30 days from now
        """
        current_date = datetime.datetime.now()
        self.gasoline_rollover_date = (current_date + datetime.timedelta(days=30)).strftime("%Y%m%d")
        self.heating_oil_rollover_date = (current_date + datetime.timedelta(days=30)).strftime("%Y%m%d")
        self.crude_oil_rollover_date = (current_date + datetime.timedelta(days=30)).strftime("%Y%m%d")

    def calculate_contract_expiry_dates(self):
        """
        Calculate end date and contract's last trading day or contract month given now and 30 days from now
        """
        current_date = datetime.datetime.now()
        self.end_date = (current_date).strftime("%Y%m%d")
        self.lastTradeDayOrContractMonth = (current_date + datetime.timedelta(days=30)).strftime("%Y%m")

    def historicalData(self, reqId, bar):
        """
        Receive and store historical price data for crack spread commodities
        """
        if reqId == 1:
            self.df_crude_oil = self.df_crude_oil.append({'datetime': bar.date, 'close': bar.close}, ignore_index=True)
            self.bardata["CL"] = bar.close
            if self.expired_contract_prices["CL"] is None:
                self.expired_contract_prices["CL"] = bar.close
        elif reqId == 2:
            self.df_gasoline = self.df_gasoline.append({'datetime': bar.date, 'close': bar.close}, ignore_index=True)
            self.bardata["RB"] = bar.close
            if self.expired_contract_prices["RB"] is None:
                self.expired_contract_prices["RB"] = bar.close
        elif reqId == 3:
            self.df_heating_oil = self.df_heating_oil.append({'datetime': bar.date, 'close': bar.close}, ignore_index=True)
            self.bardata["HO"] = bar.close
            if self.expired_contract_prices["HO"] is None:
                self.expired_contract_prices["HO"] = bar.close

        self.bar_count += 1

    def calculate_garch_volatility(self, returns_series, p=1, q=1):
        """
        Calculate GARCH volatility for given returns using specific parameters
        """
        model = arch.arch_model(returns_series, vol='Garch', p=p, q=q)
        result = model.fit(disp='off')
        return np.sqrt(result.conditional_volatility[-1])

    def calculate_crack_spreads(self):
        """
        Calculate crack spreads based on historical price of crack spread commodities
        """
        if self.df_gasoline.empty or self.df_crude_oil.empty or self.df_heating_oil.empty:
            return None

        crack_spreads = (3 * self.df_crude_oil['close'] -
                         2 * self.df_gasoline['close'] -
                         1 * self.df_heating_oil['close']) / 3

        return crack_spreads

    def calculate_z_score(self, crack_spreads):
        """
        Calculate z-score for the crack spreads given historical returns
        """
        returns = crack_spreads.diff().dropna()
        z_score = (returns.iloc[-1] - returns.mean()) / returns.std()
        return z_score

    def submit_order(self, contract, direction, qty=100, ordertype='MKT', transmit=True):
        """
        Submit an order for trading with specific parameters
        """
        order = Order()
        order.action = direction
        order.totalQuantity = qty
        order.orderType = ordertype
        order.transmit = transmit

        self.open_orders.append(order)

        self.placeOrder(self.nextOrderId, contract, order)
        self.nextOrderId += 1

    def optimize_parameters_cv(self, returns_series):
        """
        Optimize GARCH parameters using time series splitting and cross-validation
        """
        tscv = TimeSeriesSplit(n_splits=5)
        best_p = 0
        best_q = 0
        min_avg_mse = float('inf')

        for p in range(1, 6):
            for q in range(1, 6):
                avg_mse = 0
                for train_idx, test_idx in tscv.split(returns_series):
                    train_data = returns_series[train_idx]
                    test_data = returns_series[test_idx]

                    model = arch.arch_model(train_data, vol='Garch', p=p, q=q)
                    result = model.fit(disp='off')
                    conditional_volatility = result.conditional_volatility[-len(test_data):]

                    mse = np.mean((test_data - conditional_volatility) ** 2)
                    avg_mse += mse

                avg_mse /= tscv.n_splits

                if avg_mse < min_avg_mse:
                    min_avg_mse = avg_mse
                    best_p = p
                    best_q = q

        return best_p, best_q

    def get_num_contracts(self, contract_symbol):
        """
        Get the number of contracts for the specific commodity
        """
        num_contracts = {
            "CL": 3,
            "RB": 2,
            "HO": 1
        }

        return num_contracts.get(contract_symbol, 0)

    def calculate_rollover_cost(self, contract_symbol):
        """
        Calculate rollover cost based on contract symbol
        """
        if contract_symbol in self.expired_contract_prices:
            new_contract_price = self.bardata[contract_symbol]
            expired_contract_price = self.expired_contract_prices[contract_symbol]

            num_contracts = self.get_num_contracts(contract_symbol)

            if self.opening_action == 'BUY':
                rollover_cost = (new_contract_price - expired_contract_price) * num_contracts
            elif self.opening_action == 'SELL':
                rollover_cost = (expired_contract_price - new_contract_price) * num_contracts
            else:
                rollover_cost = 0.0

            return rollover_cost
        else:
            return 0.0

    def rollover_contract(self, contract, rollover_date):
        """
        Rollover contract if rollover date is reached
        """
        today = datetime.datetime.now().strftime("%Y%m%d")
        if today >= rollover_date:
            new_rollover_date = (rollover_date + datetime.timedelta(days=30)).strftime("%Y%m%d")
            new_contract = self.create_contract(
                contract.symbol,
                contract.secType,
                contract.exchange,
                new_rollover_date,
                contract.multiplier
            )

            if self.is_position_open:
                rollover_cost = self.calculate_rollover_cost(contract.symbol)
                self.cumulative_paper_pnl -= rollover_cost
                self.cumulative_pnl -= rollover_cost

            self.gasoline_rollover_date = new_rollover_date
            self.heating_oil_rollover_date = new_rollover_date
            self.crude_oil_rollover_date = new_rollover_date

            return new_contract
        return contract

    def open_position_zscore_trigger(self):
        """
        Trigger opening a trading position based on z-score and GARCH volatility
        """
        self.gasoline_contract = self.rollover_contract(self.gasoline_contract, self.gasoline_rollover_date)
        self.heating_oil_contract = self.rollover_contract(self.heating_oil_contract, self.heating_oil_rollover_date)
        self.crude_oil_contract = self.rollover_contract(self.crude_oil_contract, self.crude_oil_rollover_date)

        crack_spreads = self.calculate_crack_spreads()

        if crack_spreads is not None:
            z_score = self.calculate_z_score(crack_spreads)
            z_score_threshold = 1.0

            if abs(z_score) > z_score_threshold and not self.is_position_open:
                returns = crack_spreads.diff().dropna()
                best_p, best_q = self.optimize_parameters_cv(returns)
                garch_volatility = self.calculate_garch_volatility(returns, p=best_p, q=best_q)
                self.open_position(crack_spreads, garch_volatility, best_p, best_q)
        else:
            self.logger.info("Crack spreads data is not available for trading or no ideal entry.")

    def open_position(self, crack_spreads, garch_volatility, p, q):
        """
        Open a trading position based on calculated crack spreads and GARCH parameters
        """
        last_crack_spread = crack_spreads.iloc[-1]

        if last_crack_spread > garch_volatility:
            self.opening_action = 'BUY'
            self.submit_order(self.crude_oil_contract, 'BUY', 3, 'MKT', False)
            self.submit_order(self.gasoline_contract, 'SELL', 2, 'MKT', False)
            self.submit_order(self.heating_oil_contract, 'SELL', 1, 'MKT', True)
        else:
            self.opening_action = 'SELL'
            self.submit_order(self.crude_oil_contract, 'SELL', 3, 'MKT', False)
            self.submit_order(self.gasoline_contract, 'BUY', 2, 'MKT', False)
            self.submit_order(self.heating_oil_contract, 'BUY', 1, 'MKT', True)

        self.opening_crack_spread = last_crack_spread
        self.is_position_open = True
        self.bar_count = 0
        self.logger.info(f"Position opened. Opening Crack Spread: {self.opening_crack_spread:.2f}")

    def close_position_zscore_trigger(self):
        """
        Trigger closing a trading position based on z-score and GARCH volatility
        """
        crack_spreads = self.calculate_crack_spreads()

        if crack_spreads is not None and self.is_position_open:
            z_score = self.calculate_z_score(crack_spreads)
            volatility_adjusted_threshold = 2.0

            if abs(z_score) < volatility_adjusted_threshold:
                self.close_position(crack_spreads)

            else:
                last_crack_spread = crack_spreads.iloc[-1]
                closing_crack_spread = last_crack_spread
                position_pnl = 0.0

                if self.opening_action == 'BUY':
                    position_pnl = (closing_crack_spread - self.opening_crack_spread)
                    self.current_pnl = position_pnl
                    self.cumulative_paper_pnl += position_pnl
                    self.logger.info(f"Position kept open. PnL: {position_pnl:.2f} | Cumulative Paper PnL: {self.cumulative_paper_pnl:.2f}")

                else:
                    position_pnl = (self.opening_crack_spread - closing_crack_spread)
                    self.current_pnl = position_pnl
                    self.cumulative_paper_pnl += position_pnl
                    self.logger.info(f"Position kept open. PnL: {position_pnl:.2f} | Cumulative Paper PnL: {self.cumulative_paper_pnl:.2f}")
        else:
            self.logger.info("Crack spreads data is not available for trading or have no position.")

    def close_position(self, crack_spreads):
        """
        Close a trading position based on calculated crack spreads, GARCH parameters, and GARCH volatility threshold
        """
        returns = crack_spreads.diff().dropna()
        best_p, best_q = self.optimize_parameters_cv(returns)
        garch_volatility = self.calculate_garch_volatility(returns, p=best_p, q=best_q)
        last_crack_spread = crack_spreads.iloc[-1]

        closing_crack_spread = last_crack_spread
        position_pnl = 0.0

        if self.bar_count >= 5:
            low_vol_threshold = 0.5
            high_vol_threshold = 1.5

            if self.opening_action == 'BUY':
                if last_crack_spread > garch_volatility * high_vol_threshold:
                    self.submit_order(self.crude_oil_contract, 'SELL', 3, 'MKT', False)
                    self.submit_order(self.gasoline_contract, 'BUY', 2, 'MKT', False)
                    self.submit_order(self.heating_oil_contract, 'BUY', 1, 'MKT', True)

                    position_pnl = max(closing_crack_spread - self.opening_crack_spread, 0)
                    self.current_pnl = position_pnl
                    self.cumulative_paper_pnl += position_pnl
                    self.cumulative_pnl += position_pnl
                    self.logger.info(f"Position closed. PnL: {position_pnl:.2f} | Cumulative Paper PnL: {self.cumulative_paper_pnl:.2f} | Cumulative PnL: {self.cumulative_pnl:.2f}")

                    self.opening_crack_spread = None
                    self.is_position_open = False
                    self.bar_count = 0
                    self.opening_action = None

                elif last_crack_spread < garch_volatility * low_vol_threshold:
                    self.submit_order(self.crude_oil_contract, 'SELL', 3, 'MKT', False)
                    self.submit_order(self.gasoline_contract, 'BUY', 2, 'MKT', False)
                    self.submit_order(self.heating_oil_contract, 'BUY', 1, 'MKT', True)

                    position_pnl = max(closing_crack_spread - self.opening_crack_spread, 0)
                    self.current_pnl = position_pnl
                    self.cumulative_paper_pnl += position_pnl
                    self.cumulative_pnl += position_pnl
                    self.logger.info(f"Position closed. PnL: {position_pnl:.2f} | Cumulative Paper PnL: {self.cumulative_paper_pnl:.2f} | Cumulative PnL: {self.cumulative_pnl:.2f}")

                    self.opening_crack_spread = None
                    self.is_position_open = False
                    self.bar_count = 0
                    self.opening_action = None

                else:
                    position_pnl = (closing_crack_spread - self.opening_crack_spread)
                    self.current_pnl = position_pnl
                    self.cumulative_paper_pnl += position_pnl
                    self.logger.info(f"Position kept open. PnL: {position_pnl:.2f} | Cumulative Paper PnL: {self.cumulative_paper_pnl:.2f}")
            else:
                if last_crack_spread > garch_volatility * high_vol_threshold:
                    self.submit_order(self.crude_oil_contract, 'BUY', 3, 'MKT', False)
                    self.submit_order(self.gasoline_contract, 'SELL', 2, 'MKT', False)
                    self.submit_order(self.heating_oil_contract, 'SELL', 1, 'MKT', True)

                    position_pnl = (self.opening_crack_spread - closing_crack_spread)
                    self.current_pnl = position_pnl
                    self.cumulative_paper_pnl += position_pnl
                    self.cumulative_pnl += position_pnl
                    self.logger.info(f"Position closed. PnL: {position_pnl:.2f} | Cumulative Paper PnL: {self.cumulative_paper_pnl:.2f} | Cumulative PnL: {self.cumulative_pnl:.2f}")

                    self.opening_crack_spread = None
                    self.is_position_open = False
                    self.bar_count = 0
                    self.opening_action = None

                elif last_crack_spread < garch_volatility * low_vol_threshold:
                    self.submit_order(self.crude_oil_contract, 'BUY', 3, 'MKT', False)
                    self.submit_order(self.gasoline_contract, 'SELL', 2, 'MKT', False)
                    self.submit_order(self.heating_oil_contract, 'SELL', 1, 'MKT', True)

                    position_pnl = (self.opening_crack_spread - closing_crack_spread)
                    self.current_pnl = position_pnl
                    self.cumulative_paper_pnl += position_pnl
                    self.cumulative_pnl += position_pnl
                    self.logger.info(f"Position closed. PnL: {position_pnl:.2f} | Cumulative Paper PnL: {self.cumulative_paper_pnl:.2f} | Cumulative PnL: {self.cumulative_pnl:.2f}")

                    self.opening_crack_spread = None
                    self.is_position_open = False
                    self.bar_count = 0
                    self.opening_action = None

                else:
                    position_pnl = (self.opening_crack_spread - closing_crack_spread)
                    self.current_pnl = position_pnl
                    self.cumulative_paper_pnl += position_pnl
                    self.logger.info(f"Position kept open. PnL: {position_pnl:.2f} | Cumulative Paper PnL: {self.cumulative_paper_pnl:.2f}")
        else:
            if self.opening_action == 'BUY':
                position_pnl = (closing_crack_spread - self.opening_crack_spread)
                self.current_pnl = position_pnl
                self.cumulative_paper_pnl += position_pnl
                self.logger.info(f"Position kept open. PnL: {position_pnl:.2f} | Cumulative Paper PnL: {self.cumulative_paper_pnl:.2f}")

            else:
                position_pnl = (self.opening_crack_spread - closing_crack_spread)
                self.current_pnl = position_pnl
                self.cumulative_paper_pnl += position_pnl
                self.logger.info(f"Position kept open. PnL: {position_pnl:.2f} | Cumulative Paper PnL: {self.cumulative_paper_pnl:.2f}")

    def create_contract(self, symbol, secType='FUT', exchange='NYMEX', lastTradeDayOrContractMonth='202309', multiplier='100'):
        """
        Create a contract with specific parameters
        """
        contract = Contract()
        contract.symbol = symbol
        contract.secType = secType
        contract.exchange = exchange
        contract.lastTradeDayOrContractMonth = lastTradeDayOrContractMonth
        contract.multiplier = multiplier
        return contract

    def nextValidId(self, orderId: int):
        """
        Receive next valid order ID from the API
        """
        super().nextValidId(orderId)
        self.nextOrderId = orderId
        self.logger.info(f'The next valid order id is: {self.nextOrderId}')

    def run_loop(self):
        """
        Start event loop for the IBApi
        """
        self.run()

def main():
    # Instantiate IBApi class
    app = IBapi()

    # Specify parameters to create a connection
    app.connect("127.0.0.1", 7497, 1)

    # Create seperate thread for API event loop
    api_thread = threading.Thread(target=app.run_loop, daemon=True)
    api_thread.start()

    # Wait for nextOrderId to be set before proceeding
    while True:
        if isinstance(app.nextOrderId, int):
            app.logger.info("Connected")
            app.logger.info("")
            break
        else:
            app.logger.info("Waiting for connection")
            time.sleep(1)

    time.sleep(1)

    app.reqIds(1)

    app.calculate_rollover_dates()
    app.calculate_contract_expiry_dates()

    app.gasoline_contract = app.create_contract("RB", "FUT", "NYMEX", app.lastTradeDayOrContractMonth, "100")
    app.heating_oil_contract = app.create_contract("HO", "FUT", "NYMEX", app.lastTradeDayOrContractMonth, "100")
    app.crude_oil_contract = app.create_contract("CL", "FUT", "NYMEX", app.lastTradeDayOrContractMonth, "100")

    app.df_gasoline = pd.DataFrame(columns=["datetime", "close"])
    app.df_heating_oil = pd.DataFrame(columns=["datetime", "close"])
    app.df_crude_oil = pd.DataFrame(columns=["datetime", "close"])

    app.reqHistoricalData(1, app.gasoline_contract, app.end_date, "30 D", "1 day", "TRADES", 1, 1, False, [])
    app.reqHistoricalData(2, app.heating_oil_contract, app.end_date, "30 D", "1 day", "TRADES", 1, 1, False, [])
    app.reqHistoricalData(3, app.crude_oil_contract, app.end_date, "30 D", "1 day", "TRADES", 1, 1, False, [])

    # Trading loop
    try:
        while True:
            app.open_position_zscore_trigger()
            app.close_position_zscore_trigger()
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Stopping the trading strategy...")
    finally:
        app.disconnect()
        app.logger.info("Trading strategy has been stopped and disconnected.")

if __name__ == '__main__':
    main()
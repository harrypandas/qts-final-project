import pandas as pd
import numpy as np
import nasdaqdatalink as ndl
import quandl
import wrds
import os
import datetime
import pandas as pd
import numpy as np
import nasdaqdatalink as ndl
import quandl
import wrds
from dotenv import load_dotenv
import os
import datetime
from datetime import timedelta
from scipy.stats import norm
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import warnings
from mpl_toolkits.mplot3d import Axes3D
import functools
from scipy.stats import zscore
from sklearn.decomposition import PCA
from functools import lru_cache
import pytz
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import statsmodels.api as sm
import matplotlib.ticker as mticker

def thousands_formatter(x, pos):
    return f'{int(x/1000)}K'  # Convert to thousands and append 'K'

def grab_quandl_table(
    table_path,
    avoid_download=False,
    replace_existing=False,
    date_override=None,
    allow_old_file=False,
    **kwargs,
):
    root_data_dir = os.path.join(os.path.expanduser("~"), "quandl_data_table_downloads")
    if not os.path.exists(root_data_dir):
        print(f"Directory does not exist. Creating: {root_data_dir}")
        os.makedirs(root_data_dir, exist_ok=True)
    else:
        print(f"Directory already exists: {root_data_dir}")

    data_symlink = os.path.normpath(os.path.join(root_data_dir, f"{table_path}_latest.zip"))

    if avoid_download and os.path.exists(data_symlink):
        print(f"Skipping any possible download of {table_path}")
        return data_symlink
    
    table_dir = os.path.dirname(data_symlink)

    print(table_dir)
    if not os.path.isdir(table_dir):
        print(f'Creating new data dir {table_dir}')
        os.mkdir(table_dir)

    if date_override is None:
        my_date = datetime.datetime.now().strftime("%Y%m%d")
    else:
        my_date = date_override
    data_file = os.path.normpath(os.path.join(root_data_dir, f"{table_path}_{my_date}.zip"))

    if os.path.exists(data_file):
        file_size = os.stat(data_file).st_size
        if replace_existing or not file_size > 0:
            print(f"Removing old file {data_file} size {file_size}")
        else:
            print(
                f"Data file {data_file} size {file_size} exists already, no need to download"
            )
            return data_file

    dl = quandl.export_table(
        table_path, 
        filename=data_file, 
        api_key=os.getenv('NDL_API_KEY'), 
        **kwargs
    )
    file_size = os.stat(data_file).st_size
    if os.path.exists(data_file) and file_size > 0:
        print(f"Download finished: {file_size} bytes")
        print(data_symlink)
        if not date_override:
            if os.path.exists(data_symlink):
                print(f"Removing old symlink")
                os.unlink(data_symlink)
            print(f"Creating symlink: {data_file} -> {data_symlink}")
            os.symlink(
                data_file, data_symlink,
            )
    else:
        print(f"Data file {data_file} failed download")
        return
    return data_symlink if (date_override is None or allow_old_file) else "NoFileAvailable"


def fetch_quandl_table(table_path, avoid_download=True, **kwargs):
    return pd.read_csv(
        grab_quandl_table(table_path, avoid_download=avoid_download, **kwargs)
    )


class QTS:
    def __init__(
            self, 
            data,
            benchmark_data,
            rank_features=['suescore'],
            quantile_n=5,
            holding_period=20, 
            dynamic_sizing='equal',
            stop_loss_threshold=-0.05,
            initial_funding_rate=0.043, 
            repo_spread=0.01, 
            initial_capital=1_000_000
        ):
        """
        PEAD-Based Quant Trading Strategy with Full Price Data.
        """
        self.data = data.sort_values(['date', 'ticker']).copy().reset_index(drop=True)
        self.benchmark = benchmark_data.sort_values('date').copy().reset_index(drop=True)
        self.rank_metrics = rank_features
        self.quantile_n = quantile_n
        self.holding_period = holding_period
        self.dynamic_sizing = dynamic_sizing
        self.stop_loss_threshold = stop_loss_threshold
        self.funding_rate = initial_funding_rate
        self.repo_rate = self.funding_rate - repo_spread  
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.pnl_data = None
        self.trade_log = []
        self.strategy_name = f"{self.holding_period}day_holding-{self.dynamic_sizing}_size-{self.quantile_n}_qtl"

    def _calculate_quantiles(self):
        """ Assign quantiles to stocks only on earnings days while keeping full price data. """

        for metric in self.rank_metrics:
            if metric not in self.data.columns:
                raise ValueError(f"Rank metric '{metric}' not found in data.")
        
        self.data['long_signal'] = False
        self.data['short_signal'] = False

        rank_df = self.data[self.rank_metrics].apply(lambda x: (x - x.mean()) / x.std(), axis=0).copy()
        self.data['rank_metric'] = rank_df.mean(axis=1)

        earnings_days = self.data[self.data['day_after_earnings'] == self.data['date']]['date'].unique()
        for date in earnings_days:
            subset = self.data[(self.data['date'] == date) & (self.data['day_after_earnings']==date)].copy()
            # Assign quantiles only for stocks reporting earnings
            subset['quantile'] = pd.qcut(subset['rank_metric'], self.quantile_n, labels=False, duplicates='drop')

            # Define Long and Short signals
            subset['long_signal'] = subset['quantile'] == subset['quantile'].max()
            # subset['short_signal'] = subset['quantile'] == subset['quantile'].min()

            # Merge updated signals into full dataset
            self.data.loc[self.data['date'] == date, ['long_signal', 'short_signal']] = subset[['long_signal', 'short_signal']]
            self.data[['long_signal', 'short_signal']] = self.data[['long_signal', 'short_signal']].fillna(False)

        self.data = self._calculate_positions(self.data)

    def _calculate_positions(self, subset_df):
        """ Assign position sizes based on quantile ranks for each earnings date separately. """

        positions_df = subset_df.copy()
        positions_df['position_size'] = 0

        # Get unique earnings announcement days
        earnings_days = positions_df['day_after_earnings'].unique()

        for date in earnings_days:
            # Select only stocks that had earnings on this date
            daily_subset = positions_df[positions_df['date'] == date].copy()

            if daily_subset.empty:
                continue

            # Apply position sizing only within stocks that reported earnings on this date
            if self.dynamic_sizing == 'equal':
                daily_subset['position_size'] = np.select(
                    [daily_subset['long_signal'], daily_subset['short_signal']],
                    [1, -1], default=0
                )

                total_symbols = daily_subset['long_signal'].sum() + daily_subset['short_signal'].sum()
                if total_symbols > 0:
                    daily_subset['position_size'] /= total_symbols  # Normalize across earnings day stocks

            elif self.dynamic_sizing == 'dynamic':
                num_long = daily_subset['long_signal'].sum()
                num_short = daily_subset['short_signal'].sum()

                if num_long > 1:
                    daily_subset.loc[daily_subset['long_signal'], 'position_size'] = daily_subset.loc[daily_subset['long_signal'], 'rank_metric'].rank(method='dense', ascending=False)

                if num_short > 1:
                    daily_subset.loc[daily_subset['short_signal'], 'position_size'] = -daily_subset.loc[daily_subset['short_signal'], 'rank_metric'].rank(method='dense', ascending=True)

                # Normalize position sizes within the earnings day
                max_pos = daily_subset['position_size'].abs().max()
                if max_pos > 0:
                    daily_subset['position_size'] /= max_pos  # Scale between -1 and 1

            # Merge updated positions back into the main dataset
            positions_df.loc[positions_df['date'] == date, 'position_size'] = daily_subset['position_size']*0.05

        return positions_df

    def _track_trades(self):
        """ Track entry and exit details for each trade, looping through daily price data to check stop-loss. """

        trading_days = sorted(self.data['date'].unique())
        trade_entries = self.data[self.data['long_signal'] | self.data['short_signal']].copy()

        # Initialize exit date as the planned holding period exit
        trade_entries['exit_date'] = trade_entries['date'].apply(
            lambda x: next((d for d in trading_days if d > x and (d - x).days >= self.holding_period), np.nan)
        )

        # Merge with exit prices
        exit_prices = self.data[['date', 'ticker', 'adj_close']].rename(columns={'date': 'exit_date', 'adj_close': 'exit_price'})
        trade_entries = trade_entries.merge(exit_prices, on=['exit_date', 'ticker'], how='left')

        trade_entries['trade_type'] = np.where(trade_entries['long_signal'], 'Long', 'Short')
        trade_entries = trade_entries.rename(columns={'date': 'entry_date', 'adj_close': 'entry_price'})
        trade_entries['notional'] = trade_entries['position_size'] * self.initial_capital

        # --- Apply Stop-Loss on a Daily Basis ---
        stop_loss_dates = []  # Store new exit dates for trades that hit stop-loss

        for index, row in trade_entries.iterrows():
            entry_date = row['entry_date']
            exit_date = row['exit_date']
            ticker = row['ticker']
            entry_price = row['entry_price']
            trade_type = row['trade_type']

            # Get price data for this ticker after entry date
            price_data = self.data[(self.data['ticker'] == ticker) & (self.data['date'] > entry_date) & (self.data['date']<=exit_date)].copy()

            for i, price_row in price_data.iterrows():
                current_date = price_row['date']
                current_price = price_row['adj_close']
                return_pct = (current_price / entry_price - 1) if trade_type == 'Long' else (entry_price / current_price - 1)

                # If stop-loss is hit, exit immediately
                if return_pct < self.stop_loss_threshold:
                    stop_loss_dates.append({'ticker': ticker, 'entry_date': entry_date, 'exit_date': current_date,'exit_price': current_price})
                    break  # Stop checking further dates once stop-loss is hit
            
            if stop_loss_dates is None:
                stop_loss_dates.append({'ticker': ticker, 'entry_date': entry_date, 'exit_date': exit_date, 'exit_price': row['exit_price']})

        # Convert stop-loss exits to DataFrame and merge with trade entries
        stop_loss_df = pd.DataFrame(stop_loss_dates)
        trade_entries = trade_entries.merge(stop_loss_df, on=['ticker','entry_date'], how='left')

        # Replace exit_date with stop-loss exit if applicable
        trade_entries['exit_date'] = trade_entries['exit_date_y'].combine_first(trade_entries['exit_date_x'])
        trade_entries['exit_price'] = trade_entries['exit_price_y'].combine_first(trade_entries['exit_price_x'])
        trade_entries.drop(columns=['exit_date_x', 'exit_date_y'], inplace=True)
        trade_entries.drop(columns=['exit_price_x', 'exit_price_y'], inplace=True)
        
        trade_entries['return'] = np.where(
            trade_entries['trade_type'] == 'Long',
            (trade_entries['exit_price'] - trade_entries['entry_price']) / trade_entries['entry_price'],
            (trade_entries['entry_price'] - trade_entries['exit_price']) / trade_entries['entry_price']
        )

        # Store updated trade log
        self.trade_log = trade_entries[['ticker', 'entry_date', 'entry_price', 'exit_date', 
                                        'exit_price', 'trade_type', 'position_size', 'notional','return']].to_dict('records')

    def _calculate_accounting(self):
        """ Compute detailed PnL metrics using trade logs, including repo costs and cumulative returns. """
    
        trade_df = pd.DataFrame(self.trade_log)

        if trade_df.empty:
            print("Error: No trades recorded. Check signal generation.")
            return

        # Compute trade returns and PnL
        trade_df['trade_return'] = np.where(
            trade_df['trade_type'] == 'Long',
            (trade_df['exit_price'] - trade_df['entry_price']) / trade_df['entry_price'],
            (trade_df['entry_price'] - trade_df['exit_price']) / trade_df['entry_price']
        )

        # Compute PnL per trade
        trade_df['pnl'] = trade_df['trade_return'] * abs(trade_df['notional'])

        # Calculate repo cost (for short positions)
        trade_df['repo_cost'] = -self.repo_rate / 252 * trade_df['position_size'].clip(upper=0) * trade_df['entry_price']

        # Compute net PnL (including repo cost)
        trade_df['net_pnl'] = trade_df['pnl'] + trade_df['repo_cost']

        # Compute net return based on initial capital
        trade_df['net_ret'] = trade_df['net_pnl'] / self.initial_capital

        # Aggregate cumulative PnL over time
        self.pnl_data = trade_df.groupby('exit_date')[['pnl', 'repo_cost', 'net_pnl', 'net_ret']].sum().reset_index()
        self.pnl_data['cumulative_pnl'] = self.pnl_data['net_pnl'].cumsum()
        self.pnl_data['cumulative_ret'] = self.pnl_data['net_ret'].cumsum()
        self.pnl_data.rename(columns={'exit_date':'date'}, inplace=True)

    def backtest(self):
        """ Run backtest while preserving full price data. """
        print(f"Starting Backtest for {self.strategy_name}...")
        self._calculate_quantiles()
        self._track_trades()
        self._calculate_accounting()
        self.performance_metrics()

    def get_trade_log(self):
        """ Display trade history. """
        trade_df = pd.DataFrame(self.trade_log)
        return trade_df

    def plot_pnl(self):
        """ Plot cumulative PnL over time. """
        plt.figure(figsize=(12, 4))
        plt.plot(self.pnl_data['date'], self.pnl_data['cumulative_pnl'], label='Cumulative Net PnL')
        plt.title(f"Strategy Cumulative Daily PnL Over Time")
        plt.xlabel("Date")
        plt.ylabel("PnL ($)")
        plt.legend()
        plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(thousands_formatter))
        plt.grid(axis='y', linestyle='-', alpha=0.7)  
        plt.grid(axis='x', linestyle='-', alpha=0.7) 
        plt.tight_layout() 
        plt.show()

    def performance_metrics(self, annualization_factor=252):
        """ Compute strategy performance metrics. """
        daily_returns = self.pnl_data.groupby('date')['net_ret'].sum().reset_index()['net_ret']
        merged_df = self.pnl_data.merge(self.benchmark[['date', 'adj_close']], on='date', how='left', suffixes=('', '_benchmark'))
        merged_df['benchmark_ret'] = merged_df['adj_close'].pct_change()
        merged_df['excess_ret'] = merged_df['net_ret'] - merged_df['benchmark_ret']

        self.perf_metrics =  pd.DataFrame({'Mean': [daily_returns.mean() * annualization_factor]})
        self.perf_metrics['Volatility'] = daily_returns.std() * np.sqrt(annualization_factor)
        self.perf_metrics['Sharpe Ratio'] = (daily_returns.mean()) / daily_returns.std() * np.sqrt(annualization_factor)
        
        mean_excess_ret = merged_df['excess_ret'].mean() * annualization_factor
        std_excess_ret = merged_df['excess_ret'].std() * np.sqrt(annualization_factor)
        information_ratio = mean_excess_ret / std_excess_ret if std_excess_ret != 0 else np.nan
        self.perf_metrics['Information Ratio'] = information_ratio

        X = sm.add_constant(merged_df['benchmark_ret']) 
        Y = merged_df['net_ret']
        
        model = sm.OLS(Y, X, missing='drop').fit() 
        beta = model.params[1] 
        alpha = model.params[0] * annualization_factor 

        self.perf_metrics['Beta'] = beta
        self.perf_metrics['Alpha'] = alpha
        self.perf_metrics['Skewness'] = daily_returns.skew()
        self.perf_metrics['Kurtosis'] = daily_returns.kurtosis()
        self.perf_metrics['Var (0.05)'] = daily_returns.quantile(0.05)
        self.perf_metrics['CVaR (0.05)'] = daily_returns[daily_returns <= daily_returns.quantile(0.05)].mean()

        wealth_index = 1000 * (1 + daily_returns).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        self.perf_metrics['Max Drawdown'] = drawdowns.min()
        
        self.perf_metrics.rename(index={0: self.strategy_name}, inplace=True)
        self.perf_metrics = self.perf_metrics.round(4)
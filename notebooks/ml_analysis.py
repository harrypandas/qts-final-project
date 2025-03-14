import itertools
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

import xgboost as xgb
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

import pandas as pd
import numpy as np
import nasdaqdatalink as ndl
import quandl
import wrds
from dotenv import load_dotenv
import os
import datetime
from datetime import timedelta
from plotnine import ggplot, aes, geom_line, scale_color_manual, geom_hline, labs, theme, element_text, facet_wrap, geom_histogram, ggtitle, geom_boxplot, stat_function, geom_density
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
from sklearn.metrics import r2_score, mean_absolute_error
import shap
import statsmodels.api as sm

proj_dir = Path.cwd().parent
base_dir = proj_dir.parent
import sys
sys.path.append(str(proj_dir))
from helper.quandl_table import fetch_quandl_table, grab_quandl_table
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    ConfusionMatrixDisplay
)



warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

start_filter_date = '2018-01-01'
end_filter_date = '2024-12-31'

data_start_date = '2019-01-01'
data_end_date = '2024-12-31'

start_filing_date = '2014-01-01'
end_filing_date = '2024-12-31'

year_beginning_filing = pd.to_datetime(start_filing_date).year
year_end_filing = pd.to_datetime(end_filing_date).year
year_beginning_filter = pd.to_datetime(start_filter_date).year
year_end_filter = pd.to_datetime(end_filter_date).year

# db = wrds.Connection(wrds_username=wrds_username, verbose = False)

def prep_data(merged_df, top_eod_data):
    merged_df['days_since_announcement'] = (merged_df['date'] - merged_df['anndats']).dt.days
    merged_df['close_after_announcement'] = merged_df.groupby('ticker')['adj_close'].shift(-1)
    merged_df['close_after_announcement'] = merged_df['close_after_announcement'].where(merged_df['days_since_announcement'] == 0, np.nan)
    merged_df['close_after_announcement'] = merged_df['close_after_announcement'].groupby(merged_df['ticker']).shift(1)
    merged_df['close_after_announcement'] = merged_df['close_after_announcement'].groupby(merged_df['ticker']).ffill()
    merged_df['return_since_announcement'] = (merged_df['adj_close'] - merged_df['close_after_announcement']) / merged_df['close_after_announcement']
    merged_df[(merged_df['ticker']=='AAPL') & (merged_df['date']>='2024-10-30') & (merged_df['date']<'2024-11-05')]
        # Define window for reference price
    window = 65

    # Compute rolling average price (proxy for purchase price)
    merged_df['ref_price'] = merged_df.groupby('ticker')['adj_close'].transform(lambda x: x.rolling(window).mean())

    # Compute Capital Gains Overhang
    merged_df['cgo'] = (merged_df['adj_close'] - merged_df['ref_price']) / merged_df['adj_close']

        # Define rolling window lengths
    momentum_windows = [21, 126, 252]  # 1-month, 6-month, 12-month

    # Compute momentum factors (log returns)
    for window in momentum_windows:
        merged_df[f'mom_{window}'] = merged_df.groupby('ticker')['adj_close'].transform(lambda x: np.log(x / x.shift(window)))

    # Compute short-term reversal (1-week return)
    merged_df['short_term_reversal'] = merged_df.groupby('ticker')['adj_close'].transform(lambda x: np.log(x / x.shift(5)))
        # Compute cumulative abnormal return (CAR) post earnings
    windows = [5, 20]  # 5-day and 30-day drift

    for w in windows:
        merged_df[f'pead_{w}d'] = merged_df.groupby('ticker')['adj_close'].transform(lambda x: x.pct_change(w))

    merged_df['day_after_earnings']= merged_df.groupby(['ticker', 'anndats'])['date'].nth(1)
    merged_df['day_after_earnings'] = merged_df.groupby('ticker')['day_after_earnings'].ffill()
    merged_df['previous_day_return'] = merged_df.groupby('ticker')['adj_close'].pct_change()
    # merged_df[merged_df['ticker']=='AAPL'][['date','anndats', 'day_after_earnings']].tail(20)


def gradient_regression_model(final_df, w=20, N=20):
    feature_importance_dict = {}
    req_cols = [
    'ticker',
    'date',
    'adj_close'
    ]
    features = [
        'suescore',
        'cgo', 
        'mom_21', 'mom_126', 'mom_252','short_term_reversal', 
        'pead_5d', 'pead_20d', 
        'ps_ratio','earnings_yield', 'fcf_yield', 'ebitda_margin', 'roi_scaled',
        'net_debt_to_ebitda', 'debt_to_revenue', 'revenue_growth',
        'ebitda_growth', 'mkt_cap_growth', 'log_mkt_cap', 'fcf_growth', 'days_since_announcement', 'return_since_announcement', 'previous_day_return'
    ]
    final_df[f'target_return_{w}_day'] = final_df.groupby('ticker')['adj_close'].pct_change(w).shift(-w)
    final_df = final_df.dropna(subset=[f'target_return_{w}_day']).copy()
    validation_set = final_df.loc[final_df['date']>=pd.to_datetime('2024-01-01')].copy()
    in_sample = final_df.loc[final_df['date']<pd.to_datetime('2024-01-01')].copy()
    test_size = int(np.floor(len(in_sample) * 0.1))
    test_set = in_sample.iloc[-test_size:]
    training_set = in_sample.iloc[:-test_size]
    
    test_size = int(np.floor(len(in_sample) * 0.1))
    X_train = training_set[features]
    y_train = training_set[f'target_return_{w}_day']
    X_test = test_set[features]
    y_test = test_set[f'target_return_{w}_day']
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    xgb_model = xgb.XGBRegressor(n_estimators=25, 
                                 learning_rate=0.1, 
                                 max_depth=6, 
                                 alpha = 2,
                                 random_state=42)
    xgb_model.fit(X_train_scaled, y_train)

    feature_importance = pd.Series(xgb_model.feature_importances_, index=features).sort_values(ascending=False)

    feature_importance_dict[w] = feature_importance

    plt.figure(figsize=(10, 4))
    feature_importance.plot(kind='bar')
    plt.xticks(rotation=-75) 
    plt.title(f"Feature Importance (XGBoost - {w} Day Return)")
    plt.show()
    
    #plot actual vs predicted returns
    y_pred = xgb_model.predict(X_test_scaled)
    plt.figure(figsize=(7, 7)) 
    colors = ['green' if np.sign(actual) == np.sign(pred) else 'red'
          for actual, pred in zip(y_test, y_pred)]

    plt.scatter(y_test, y_pred, label='Actual vs Predicted Returns', c = colors)
    plt.xlabel('Actual Returns')
    plt.ylabel('Predicted Returns')
    plt.legend()
    plt.title(f"Actual vs Predicted Returns ({w} Day Return)")
    plt.show()

    # stacked histograms of actual vs predicted returns
    plt.figure(figsize=(10, 4))
    sns.histplot(y_test, color='blue', alpha=0.5, bins=100, label='Actual Returns')
    sns.histplot(y_pred, color='red', alpha=0.5, bins=100, label='Predicted Returns')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title(f"Actual vs Predicted Returns ({w} Day Return)")
    plt.show()

        # Assuming you have a model and X_train for background data:
    explainer = shap.Explainer(xgb_model, X_train)
    shap_values = explainer(X_test)

    # Choose an instance (e.g., the first instance)
    instance_index = 0
    shap_val_instance = shap_values[instance_index]

    # Round the values to 2 decimal places:
    shap_val_instance.values = np.round(shap_val_instance.values, 6)
    shap_val_instance.base_values = round(shap_val_instance.base_values, 6)
    shap_val_instance.data = np.round(shap_val_instance.data, 6)

    shap.plots.waterfall(shap_val_instance)

        # Assuming features is a list of your actual feature names
    xgb_model.get_booster().feature_names = features
        # Plot the first tree (num_trees=0 for the first tree)
    xgb.plot_tree(xgb_model, num_trees=0)
    plt.rcParams['figure.figsize'] = [100, 10]  # Adjust the size if needed
    plt.show()


    # Print out of sample metrics
    r2 = r2_score(y_test, xgb_model.predict(X_test_scaled))
    mae = mean_absolute_error(y_test, xgb_model.predict(X_test_scaled))
    signs = [1 if np.sign(actual) == np.sign(pred) else 0
        for actual, pred in zip(y_test, y_pred)]
    print("Percentage of Correct Signs:", sum(signs)/len(signs))
    print(f"R2 Score for {w} Day Return: {r2}")
    print(f"Mean Absolute Error for {w} Day Return: {mae}")

    top_features = feature_importance.index[:10].tolist()
    print("Top 10 Important Features:", top_features)


def gradient_clf_model(final_df, spy_df, w=20, N=20):

    target_col = f'cat_target_return_{w}_day'  # Change this if needed
    if target_col not in final_df.columns:
        final_df = final_df.dropna(subset= 'adj_close').copy()
        final_df[f'target_return_{w}_day'] = final_df.groupby('ticker')['adj_close'].pct_change(w)
        final_df[f'target_return_{w}_day'] = final_df.groupby('ticker')[f'target_return_{w}_day'].shift(-w)
        final_df[f'lower_quantile_{w}_day'] = final_df[f'target_return_{w}_day'].expanding().quantile(0.25)
        final_df[f'upper_quantile_{w}_day'] = final_df[f'target_return_{w}_day'].expanding().quantile(0.75)
        final_df[f"cat_target_return_{w}_day"] = np.where(
        (final_df[f'lower_quantile_{w}_day'] > 0) | (final_df[f'upper_quantile_{w}_day'] < 0),
        0,
        np.where(
            final_df[f'target_return_{w}_day'] < final_df[f'lower_quantile_{w}_day'],
            1,
            np.where(
                final_df[f'target_return_{w}_day'] > final_df[f'upper_quantile_{w}_day'],
                2,
                0
            )
        )
        )
    final_df= final_df.loc[(final_df['date']>=data_start_date) & (final_df['date']<=data_end_date)].copy().reset_index(drop=True)
    final_df = final_df.loc[final_df['date']==final_df['day_after_earnings']].copy()


    # 1) Split your data into training, test, and validation sets
    validation_set = final_df.loc[final_df['date'] >= pd.to_datetime('2024-01-01')].copy()
    in_sample = final_df.loc[final_df['date'] < pd.to_datetime('2024-01-01')].copy()

    test_set = in_sample.loc[in_sample['date'] >= pd.to_datetime('2023-01-01')].copy()
    training_set = in_sample.loc[in_sample['date'] < pd.to_datetime('2023-01-01')].copy()

    # 2) Define your columns
    req_cols = [
        'ticker',
        'date',
        'adj_close'
    ]

    features = [
        'suescore', 'cgo', 
        'mom_21', 'mom_126', 'mom_252','short_term_reversal', 
        'pead_5d', 'pead_20d', 
        'ps_ratio','earnings_yield', 'fcf_yield', 'ebitda_margin', 'roi_scaled',
        'net_debt_to_ebitda', 'debt_to_revenue', 'revenue_growth',
        'ebitda_growth', 'mkt_cap_growth', 'log_mkt_cap', 'fcf_growth', 
        'days_since_announcement', 'return_since_announcement', 'previous_day_return'
    ]

    # prediction_target_windows = [1, 5]  # Example
    feature_importance_dict = {}

    predictions_df = pd.DataFrame({'date':sorted(test_set['date'].unique())})
    for col in sorted(test_set['ticker'].unique()):
        predictions_df[col] = 0

    prices_df = test_set[['date', 'ticker', 'adj_close']].pivot_table(
        index='date',
        columns='ticker',
        values='adj_close',
        aggfunc='first'
    )
    prices_df = pd.merge(prices_df, spy_df[['date', 'adj_close']], how='left', on='date')
    prices_df.rename(columns={'adj_close':'SPY'}, inplace=True)




    # 3) Prepare X and y for classification
    #    * Instead of 'target_return_{w}_day', we use 'cat_target_return_{w}_day'


    X_train = training_set[features]
    y_train = training_set[f'cat_target_return_{w}_day']  # e.g. -1, 0, 1
    X_test = test_set[features]
    y_test = test_set[f'cat_target_return_{w}_day']

    # 4) Scale your features (optional but often helpful)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    class_counts = y_train.value_counts()
    # e.g. class_counts might be: {0: 10000, 1: 6000, 2: 4000}

    # Invert them so smaller classes get higher weights:
    weights = (1.0 / class_counts)
    # e.g. => {0: 0.0001, 1: 0.000166..., 2: 0.00025}

    # Then map each sample in y_train to its class weight:
    class_weight_map = {0: 1, 1: 10, 2: 10}

    sample_weight = y_train.map(class_weight_map)

    # 5) Create and train an XGBClassifier
    #    If you have three classes (-1, 0, 1), use 'multi:softmax' with num_class=3
    xgb_model = xgb.XGBClassifier(
        tree_method='hist',  # Use 'hist' for faster training
        n_estimators=10,
        learning_rate=0.1,
        max_depth=10,
        random_state=42,
        objective='multi:softprob',  # multi-class objective
        num_class=3,
        sample_weight=sample_weight  # Pass the sample weights here
    )
    xgb_model.fit(X_train_scaled, y_train)

    # 6) Feature importance (from XGBClassifier)
    feature_importance = pd.Series(
        xgb_model.feature_importances_,
        index=features
    ).sort_values(ascending=False)

    feature_importance_dict[w] = feature_importance

    plt.figure(figsize=(10, 4))
    feature_importance.plot(kind='bar')
    plt.xticks(rotation=-75)
    plt.title(f"Feature Importance (XGBoost - {w}-Day Classification)")
    plt.show()

    # 7) Predictions and classification metrics
    y_pred = xgb_model.predict(X_test_scaled)
    test_set['prediction'] = y_pred

    for idx, row in test_set.iterrows():
        # Convert the date to a Timestamp
        pred_date = pd.to_datetime(row['date'])
        ticker = row['ticker']
        pred = row['prediction']
        
        # Map prediction: 1 -> short (-1), 2 -> long (1), 0 -> flat (0)
        if pred == 1:
            pos = -1
        elif pred == 2:
            pos = 1
        else:
            pos = 0

        # Find the index (or indices) in predictions_df corresponding to the prediction date.
        # (We assume your predictions_df has a 'date' column that is a datetime or convertible to one.)
        matching_index = predictions_df.index[predictions_df['date'] == pred_date]
        if len(matching_index) == 0:
            continue  # Skip if the prediction date is not found
        start_index = matching_index[0]
        
        # Determine the end index (making sure not to go out of bounds)
        end_index = start_index + N
        if end_index > len(predictions_df):
            end_index = len(predictions_df)
        
        # For each day from the prediction date to N days after, update the position for this ticker.
        predictions_df.loc[start_index:end_index-1, ticker] = pos

    # -- Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy ({w}-Day): {accuracy:.3f}")

    # -- Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix ({w}-Day):\n{cm}")

    # -- Display confusion matrix nicely
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                    display_labels=xgb_model.classes_)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix ({w}-Day)")
    plt.show()

    # -- Classification Report
    report = classification_report(y_test, y_pred, zero_division=0)
    print(f"Classification Report ({w}-Day):\n{report}")

    # 8) Plot distribution of actual vs predicted classes
    plt.figure(figsize=(10, 4))
    sns.histplot(y_test, color='blue', alpha=0.5, bins=3, label='Actual', discrete=True)
    sns.histplot(y_pred, color='red', alpha=0.5, bins=3, label='Predicted', discrete=True)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.legend()
    plt.title(f"Distribution of Actual vs Predicted Classes ({w}-Day)")
    plt.show()

class XGBRFClassifier(BaseEstimator, ClassifierMixin):
    """
    A scikit-learn estimator wrapper for XGBoost random forest mode.
    This uses xgb.train with:
    - learning_rate=1, n_estimators=1 (dummy, since num_parallel_tree controls the ensemble size)
    - objective: multi:softprob
    - num_class: 3 (for three-class classification)
    """
    def __init__(self, max_depth=4, num_parallel_tree=500, subsample=0.8, colsample_bynode=0.8, seed=42, tree_method='gpu_hist'):
        self.max_depth = max_depth
        self.num_parallel_tree = num_parallel_tree
        self.subsample = subsample
        self.colsample_bynode = colsample_bynode
        self.seed = seed
        self.tree_method = tree_method
        self.bst = None  # Will hold the trained booster

    def fit(self, X, y, sample_weight=None):
        dtrain = xgb.DMatrix(X, label=y, weight=sample_weight)
        params = {
            "colsample_bynode": self.colsample_bynode,
            "learning_rate": 1,           # fixed for RF mode
            "max_depth": self.max_depth,
            "num_parallel_tree": self.num_parallel_tree,
            "objective": "multi:softprob",
            "num_class": 3,
            "subsample": self.subsample,
            "tree_method": self.tree_method,
            "seed": self.seed
        }
        self.bst = xgb.train(params, dtrain, num_boost_round=1)
        return self

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        preds = self.bst.predict(dtest)  # shape: (n_samples, num_class)
        return preds

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

def run_backtest(merged_df, 
                 spy_df,
                 ff, 
                 w = 60, 
                 N=60, 
                 capital_per_trade=10000, 
                 trading_cost = 0.001, 
                 depth = 5, 
                 num_parallel_tree = 200):
    # 1) Split your data into training, test, and validation sets
    starting_portfolio_value = capital_per_trade * len(merged_df['ticker'].unique())



    final_df = merged_df.dropna(subset= 'adj_close').copy()
    final_df[f'target_return_{w}_day'] = final_df.groupby('ticker')['adj_close'].pct_change(w)
    final_df[f'target_return_{w}_day'] = final_df.groupby('ticker')[f'target_return_{w}_day'].shift(-w)
    final_df[f'lower_quantile_{w}_day'] = final_df[f'target_return_{w}_day'].expanding().quantile(0.25)
    final_df[f'upper_quantile_{w}_day'] = final_df[f'target_return_{w}_day'].expanding().quantile(0.75)
    final_df[f"cat_target_return_{w}_day"] = np.where(
    (final_df[f'lower_quantile_{w}_day'] > 0) | (final_df[f'upper_quantile_{w}_day'] < 0),
    0,
    np.where(
        final_df[f'target_return_{w}_day'] < final_df[f'lower_quantile_{w}_day'],
        1,
        np.where(
            final_df[f'target_return_{w}_day'] > final_df[f'upper_quantile_{w}_day'],
            2,
            0
        )
    )
    )
    final_df= final_df.loc[(final_df['date']>=data_start_date) & (final_df['date']<=data_end_date)].copy().reset_index(drop=True)

    test_set = final_df.loc[(final_df['date'] >= pd.to_datetime('2023-01-01')) & (final_df['date']< '2024-01-01')].copy()

    # 2) Define your columns
    req_cols = [
        'ticker',
        'date',
        'adj_close'
    ]

    features = [
        'suescore', 'cgo', 
        'mom_21', 'mom_126', 'mom_252','short_term_reversal', 
        'pead_5d', 'pead_20d', 
        'ps_ratio','earnings_yield', 'fcf_yield', 'ebitda_margin', 'roi_scaled',
        'net_debt_to_ebitda', 'debt_to_revenue', 'revenue_growth',
        'ebitda_growth', 'mkt_cap_growth', 'log_mkt_cap', 'fcf_growth', 
        'days_since_announcement', 'return_since_announcement', 'previous_day_return'
    ]

    # prediction_target_windows = [1, 5]  # Example
    feature_importance_dict = {}

    predictions_df = pd.DataFrame({'date':sorted(test_set['date'].unique())})
    for col in sorted(test_set['ticker'].unique()):
        predictions_df[col] = 0

    prices_df = test_set[['date', 'ticker', 'adj_close']].pivot_table(
        index='date',
        columns='ticker',
        values='adj_close',
        aggfunc='first'
    )
    prices_df = pd.merge(prices_df, spy_df[['date', 'adj_close']], how='left', on='date')
    prices_df.rename(columns={'adj_close':'SPY'}, inplace=True)

    # 1) Split your data into training, test, and validation sets
    final_df = final_df.loc[final_df['date']==final_df['day_after_earnings']].copy()

    # Remove duplicate ticker/date rows
    final_df = final_df.groupby(['ticker', 'date']).first().reset_index()

    validation_set = final_df.loc[final_df['date'] >= pd.to_datetime('2024-01-01')].copy()
    in_sample = final_df.loc[final_df['date'] < pd.to_datetime('2024-01-01')].copy()

    test_set = in_sample.loc[in_sample['date'] >= pd.to_datetime('2023-01-01')].copy()
    training_set = in_sample.loc[in_sample['date'] < pd.to_datetime('2023-01-01')].copy()

    # 2) Visualize class distributions in the training set and test set
    target_col = f"cat_target_return_{w}_day"  # Change this if needed


    # 3) Define your features and target for the 5-day classification
    features = [
        'suescore', 'cgo', 
        'mom_21', 'mom_126', 'mom_252', 'short_term_reversal', 
        'pead_5d', 'pead_20d', 
        'ps_ratio', 'earnings_yield', 'fcf_yield', 'ebitda_margin', 'roi_scaled',
        'net_debt_to_ebitda', 'debt_to_revenue', 'revenue_growth',
        'ebitda_growth', 'mkt_cap_growth', 'log_mkt_cap', 'fcf_growth', 
        'days_since_announcement', 'return_since_announcement', 'previous_day_return'
    ]

    X_train = training_set[features]
    y_train = training_set[target_col]
    X_test = test_set[features]
    y_test = test_set[target_col]

    # 4) Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5) Define sample weights based on class to boost underrepresented classes
    # For example, if you want to weight classes 1 and 2 higher:
    # Get the class counts from y_train
    class_counts = y_train.value_counts()
    total_samples = len(y_train)

    # Compute class percentages
    class_percentages = class_counts / total_samples

    # Option A: Use inverse of percentage (may produce large numbers if some classes are very rare)
    class_weight_map = {cls: 1.0 / perc for cls, perc in class_percentages.items()}
    sample_weight = y_train.map(class_weight_map).values  # Create an array of weights

    # 6) Create DMatrix objects for XGBoost training and testing
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train, weight=sample_weight)
    dtest = xgb.DMatrix(X_test_scaled, label=y_test)

    # 7) Set up the parameter dictionary for Random Forest using the XGBoost native API.
    #    Note: We use num_parallel_tree to set the number of trees, and num_boost_round=1 so that XGBoost trains
    #    a single random forest instead of boosting multiple forests.
    params = {
        "colsample_bynode": 0.8,         # Randomly sample columns for each split
        "learning_rate": 1,              # Must be set to 1 for random forest training
        "max_depth": depth,
        "num_parallel_tree": num_parallel_tree,        # Number of trees in the forest
        "objective": "multi:softprob",   # Multi-class classification with probabilities
        "num_class": 3,                  # Number of classes
        "subsample": 0.8,                # Randomly sample rows (cases)
        "tree_method": "gpu_hist",       # Use GPU acceleration; for CPU use "hist"
        "gpu_id": 0,
        "seed": 42
    }

    num_boost_round = 1  # Train a single random forest

    # 8) Train the model
    bst = xgb.train(params, dtrain, num_boost_round=num_boost_round)

    # 9) Make predictions on the test set
    y_pred_prob = bst.predict(dtest)  # Returns probabilities for each class
    y_pred = np.argmax(y_pred_prob, axis=1)
    


    test_set['prediction'] = y_pred

    for idx, row in test_set.iterrows():
        # Convert the date to a Timestamp
        pred_date = pd.to_datetime(row['date'])
        ticker = row['ticker']
        pred = row['prediction']
        
        # Map prediction: 1 -> short (-1), 2 -> long (1), 0 -> flat (0)
        if pred == 1:
            pos = -1
        elif pred == 2:
            pos = 1
        else:
            pos = 0

        # Find the index (or indices) in predictions_df corresponding to the prediction date.
        # (We assume your predictions_df has a 'date' column that is a datetime or convertible to one.)
        matching_index = predictions_df.index[predictions_df['date'] == pred_date]
        if len(matching_index) == 0:
            continue  # Skip if the prediction date is not found
        start_index = matching_index[0]
        
        # Determine the end index (making sure not to go out of bounds)
        end_index = start_index + N
        if end_index > len(predictions_df):
            end_index = len(predictions_df)
        
        # For each day from the prediction date to N days after, update the position for this ticker.
        predictions_df.loc[start_index:end_index-1, ticker] = pos


    # 10) Evaluate the model
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy:.3f}")

    cm = confusion_matrix(y_test, y_pred)
    # print("Confusion Matrix:")
    # print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Extract importance scores from the model
    # The keys are like 'f0', 'f1', ... corresponding to feature indices.
    score_dict = bst.get_score(importance_type='gain')

    # Map these keys back to your feature names (assuming the order of features matches)
    mapped_scores = {features[int(k[1:])]: v for k, v in score_dict.items()}

    # Create a pandas Series and sort the values
    importance_series = pd.Series(mapped_scores).sort_values(ascending=False)

    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    importance_series.plot(kind='bar')
    plt.title(f"Feature Importances (Gain) {w} Day Return")
    plt.ylabel("Gain")
    plt.xlabel("Feature")
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    plt.show()

    predictions_df.set_index('date', inplace=True)

    # positions_df will store our share positions (not just the signal) for each ticker and SPY.
    # For each ticker trade, we allocate exactly 'capital_per_trade' dollars.
    # Partial shares are allowed.
    # Create an empty DataFrame for share positions (same shape as predictions_df)
    positions_df = pd.DataFrame(index=predictions_df.index, columns=predictions_df.columns)
    positions_df = positions_df.astype(float)
    prices_df.set_index('date', inplace=True)

    # For each ticker, determine the fixed number of shares at entry and hold that throughout the trade.
    for ticker in predictions_df.columns:
        # Create a series to hold the share positions for this ticker
        shares = pd.Series(index=predictions_df.index, dtype=float)
        current_signal = 0  # tracks whether we're in a trade
        entry_shares = 0.0  # the number of shares determined at entry
        
        # Loop over each day (date) in the predictions
        for date in predictions_df.index:
            signal = predictions_df.loc[date, ticker]
            
            if signal == 0:
                # No position on this day.
                shares.loc[date] = 0
                current_signal = 0  # reset the trade status
            else:
                # If we're starting a new trade (or signal reverses), compute entry shares.
                if current_signal == 0 or signal != current_signal:
                    current_signal = signal
                    entry_price = prices_df.loc[date, ticker]
                    # Calculate number of shares using the entry price:
                    entry_shares = (capital_per_trade / entry_price) * signal
                # Hold the same number of shares for this day (and future days until trade ends)
                shares.loc[date] = entry_shares
                
        # Save the computed share positions back into positions_df
        positions_df[ticker] = shares

    # Now, compute the net dollar exposure from the ticker trades.
    # For each ticker, the dollar exposure on a given day is: shares * current price.

    net_exposure = (positions_df* prices_df).sum(axis=1)

    # Hedge the overall net exposure using SPY.
    # The required SPY share position = - (net exposure) / (SPY price).
    positions_df['SPY'] = - net_exposure / prices_df['SPY']

    # Now, positions_df holds the share positions for each ticker (held at fixed share count) and SPY.
    # You can then compute daily PnL by multiplying positions by daily returns.
    # For example, if you have a returns_df computed as:
    returns_df = prices_df.pct_change().fillna(0)

    daily_pnl_unhedged = (positions_df * prices_df * returns_df).drop(columns=['SPY']).sum(axis=1)
    daily_pnl_hedged = (positions_df * prices_df * returns_df).sum(axis=1)


    # Compute daily change in share positions.
    position_changes = positions_df.diff().fillna(0).abs()
    # Nominal value traded = change in positions * asset price.
    nominal_traded = position_changes * prices_df
    # Sum trading costs across all assets for each day.
    daily_trading_cost = trading_cost * nominal_traded.sum(axis=1)

    # 7) Adjust daily PnL for trading costs.
    daily_pnl_net_hedged = daily_pnl_hedged - daily_trading_cost
    daily_pnl_net_unhedged = daily_pnl_unhedged - daily_trading_cost

    cumulative_pnl_hedged = daily_pnl_net_hedged.cumsum()
    cumulative_pnl_unhedged = daily_pnl_net_unhedged.cumsum()

    # # Output the first few rows to inspect
    # print("Share Positions (first 5 rows):")
    # print(positions_df.head())

    # print("\nDaily PnL (first 5 rows):")
    # print(daily_pnl.head())

    # print("\nCumulative PnL (first 5 rows):")
    # print(cumulative_pnl.head())

    # Analyze returns:


    # (You can then plot the cumulative PnL or further analyze the backtest performance.)
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_pnl_hedged.index, cumulative_pnl_hedged, label='Cumulative PnL Hedged')
    plt.plot(cumulative_pnl_unhedged.index, cumulative_pnl_unhedged, label='Cumulative PnL Unhedged')
    # plt.plot(cumulative_pnl.index, net_exposure, label='Net Exposure')
    plt.xlabel('Date')
    plt.ylabel('Cumulative PnL ($)')
    plt.title(f'Backtest Portfolio Performance {w} Day Prediction {N} holding period')
    plt.legend()
    plt.show()

    # Count the number of long (1) and short (-1) positions for each day:
    long_counts = predictions_df.apply(lambda row: (row == 1).sum(), axis=1)
    short_counts = predictions_df.apply(lambda row: (row == -1).sum(), axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(long_counts.index, long_counts, label='Long Positions')
    plt.plot(short_counts.index, short_counts, label='Short Positions')
    plt.xlabel('Date')
    plt.ylabel('Number of Positions')
    plt.title('Number of Long and Short Positions Over Time')
    plt.legend()
    plt.show()

    total_notional_hedged = positions_df.abs().mul(prices_df).sum(axis=1)
    total_notional_unhedged = positions_df.drop(columns='SPY').abs().mul(prices_df).sum(axis=1)
    plt.figure(figsize=(10, 6))     
    plt.plot(total_notional_hedged.index, total_notional_hedged, label='Total Notional Hedged')
    plt.plot(total_notional_unhedged.index, total_notional_unhedged, label='Total Notional Unhedged')  
    plt.xlabel('Date')
    plt.ylabel('Total Notional Value ($)')
    plt.title('Total Notional Value Over Time')
    plt.legend()
    plt.show()

    returns_hedged = cumulative_pnl_hedged.pct_change().dropna()
    returns_hedged = returns_hedged[np.isfinite(returns_hedged)]
    returns_unhedged = cumulative_pnl_unhedged.pct_change().dropna()
    returns_unhedged = returns_unhedged[np.isfinite(returns_unhedged)]
    # print(returns_hedged.head())

    sharpe_ratio = returns_hedged.mean() / returns_hedged.std() * np.sqrt(252)
    print(f"Sharpe Ratio (Hedged): {sharpe_ratio:.4f}")

    sharpe_ratio = returns_unhedged.mean() / returns_unhedged.std() * np.sqrt(252)
    print(f"Sharpe Ratio (Unhedged): {sharpe_ratio:.4f}")

    reg = sm.OLS(returns_hedged, sm.add_constant(ff.loc[ff['date'].isin(returns_hedged.index)].set_index('date'))).fit()
    print(reg.summary())
    return returns_hedged, returns_unhedged

def rf_gridsearch(param_grid_xgb, merged_df):

    # Create a list of parameter combinations using itertools.product
    param_names = list(param_grid_xgb.keys())
    param_values = list(param_grid_xgb.values())
    param_combinations = list(itertools.product(*param_values))

    # Set up a 3-fold stratified cross validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # To store the results for each parameter combination
    results = []
    final_df = merged_df.dropna(subset= 'adj_close').copy()



    for w in param_grid_xgb['holding_period']:

        final_df[f'target_return_{w}_day'] = final_df.groupby('ticker')['adj_close'].pct_change(w)
        final_df[f'target_return_{w}_day'] = final_df.groupby('ticker')[f'target_return_{w}_day'].shift(-w)
        final_df[f'lower_quantile_{w}_day'] = final_df[f'target_return_{w}_day'].expanding().quantile(0.25)
        final_df[f'upper_quantile_{w}_day'] = final_df[f'target_return_{w}_day'].expanding().quantile(0.75)
        final_df[f"cat_target_return_{w}_day"] = np.where(
        (final_df[f'lower_quantile_{w}_day'] > 0) | (final_df[f'upper_quantile_{w}_day'] < 0),
        0,
        np.where(
            final_df[f'target_return_{w}_day'] < final_df[f'lower_quantile_{w}_day'],
            1,
            np.where(
                final_df[f'target_return_{w}_day'] > final_df[f'upper_quantile_{w}_day'],
                2,
                0
            )
        )
        )
    final_df= final_df.loc[(final_df['date']>=data_start_date) & (final_df['date']<=data_end_date)].copy().reset_index(drop=True)

    test_set = final_df.loc[(final_df['date'] >= pd.to_datetime('2023-01-01')) & (final_df['date']< '2024-01-01')].copy()

    # 2) Define your columns
    req_cols = [
        'ticker',
        'date',
        'adj_close'
    ]

    # 1) Split your data into training, test, and validation sets
    final_df = final_df.loc[final_df['date']==final_df['day_after_earnings']].copy()

    # Remove duplicate ticker/date rows
    final_df = final_df.groupby(['ticker', 'date']).first().reset_index()
    #     validation_set = final_df.loc[final_df['date'] >= pd.to_datetime('2024-01-01')].copy()
    in_sample = final_df.loc[final_df['date'] < pd.to_datetime('2024-01-01')].copy()

    test_set = in_sample.loc[in_sample['date'] >= pd.to_datetime('2023-01-01')].copy()
    training_set = in_sample.loc[in_sample['date'] < pd.to_datetime('2023-01-01')].copy()


    # 3) Define your features and target for the 5-day classification
    features = [
        'suescore', 'cgo', 
        'mom_21', 'mom_126', 'mom_252', 'short_term_reversal', 
        'pead_5d', 'pead_20d', 
        'ps_ratio', 'earnings_yield', 'fcf_yield', 'ebitda_margin', 'roi_scaled',
        'net_debt_to_ebitda', 'debt_to_revenue', 'revenue_growth',
        'ebitda_growth', 'mkt_cap_growth', 'log_mkt_cap', 'fcf_growth', 
        'days_since_announcement', 'return_since_announcement', 'previous_day_return'
    ]

    X_train = training_set[features]
        # 4) Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Loop over every combination in the grid
    for comb in param_combinations:
        params = dict(zip(param_names, comb))
        fold_f1_scores = []
        y_train = training_set[f'cat_target_return_{params["holding_period"]}_day']
        # 5) Define sample weights based on class to boost underrepresented classes
        # For example, if you want to weight classes 1 and 2 higher:
        # Get the class counts from y_train
        class_counts = y_train.value_counts()
        total_samples = len(y_train)

        # Compute class percentages
        class_percentages = class_counts / total_samples

        # Option A: Use inverse of percentage (may produce large numbers if some classes are very rare)
        class_weight_map = {cls: 1.0 / perc for cls, perc in class_percentages.items()}
        sample_weight = y_train.map(class_weight_map).values  # Create an array of weights
        
        # Perform CV manually
        for train_idx, val_idx in skf.split(X_train_scaled, y_train):

            X_tr = X_train_scaled[train_idx]
            y_tr = y_train.iloc[train_idx]
            sw_tr = sample_weight[train_idx]
            X_val = X_train_scaled[val_idx]
            y_val = y_train.iloc[val_idx]
            
            # Initialize the custom XGBoost Random Forest estimator with current parameters
            model = XGBRFClassifier(
                max_depth=params['max_depth'],
                num_parallel_tree=params['num_parallel_tree'],
                # subsample=params['subsample'],
                # colsample_bynode=params['colsample_bynode'],
                tree_method='gpu_hist',  # Change to 'hist' if you're using CPU
                seed=42
            )
            
            # Fit the model
            model.fit(X_tr, y_tr, sample_weight=sw_tr)
            # Predict on the validation fold
            y_pred = model.predict(X_val)
            # Compute macro F1 score for this fold
            fold_f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
            fold_f1_scores.append(fold_f1)
        
        # Average the score across folds for this parameter combination
        avg_f1 = np.mean(fold_f1_scores)
        results.append((params, avg_f1))
        print("Parameters:", params, "-> Avg Macro F1:", avg_f1)

    # Identify the best parameter combination
    best_params, best_score = max(results, key=lambda x: x[1])
    print("\nBest parameters:", best_params)
    print("Best macro F1 score:", best_score)
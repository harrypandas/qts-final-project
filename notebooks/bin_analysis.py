"""Here is some bulk of the code that consists of functions used to calculate the bins and the 
lists/dictionaries with their varying parameters """




# The names used to label each approach in ranking.
approach_names = [
    "Surprise Weight = 0.9",
    "Equal Weighting (all factors = 0.2)",
    "Surprise = 0.6, Others = 0.1",
    "Lookback Period = 1 Month",
    "Lookback Period = 2 Months",
    "Base 3-Month Quantile Approach",
]




# This list contains all the initial quantile variations
param_list_quantile_bin = [
    # 1) Top 10% / Bot 10%, Bin-Aware
    {
        "top_quantile_threshold": 0.10,
        "bottom_quantile_threshold": 0.10,
        "bin_aware": True,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
        "lookback_months": 3,
        "weight_surprise": 1,
        "weight_ps": 0,
        "weight_netdebt": 0,
        "weight_revgrowth": 0,
        "weight_ebitdagrowth": 0
    },
    # 2) Top 10% / Bot 10%, No Bin
    {
        "top_quantile_threshold": 0.10,
        "bottom_quantile_threshold": 0.10,
        "bin_aware": False,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
        "lookback_months": 3,
        "weight_surprise": 1,
        "weight_ps": 0,
        "weight_netdebt": 0,
        "weight_revgrowth": 0,
        "weight_ebitdagrowth": 0
    },
    # 3) Top 20% / Bot 20%, Bin-Aware
    {
        "top_quantile_threshold": 0.20,
        "bottom_quantile_threshold": 0.20,
        "bin_aware": True,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
        "lookback_months": 3,
        "weight_surprise": 1,
        "weight_ps": 0,
        "weight_netdebt": 0,
        "weight_revgrowth": 0,
        "weight_ebitdagrowth": 0
    },
    # 4) Top 20% / Bot 20%, No Bin
    {
        "top_quantile_threshold": 0.20,
        "bottom_quantile_threshold": 0.20,
        "bin_aware": False,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
        "lookback_months": 3,
        "weight_surprise": 1,
        "weight_ps": 0,
        "weight_netdebt": 0,
        "weight_revgrowth": 0,
        "weight_ebitdagrowth": 0
    },
    # 5) Top 30% / Bot 30%, Bin-Aware
    {
        "top_quantile_threshold": 0.30,
        "bottom_quantile_threshold": 0.30,
        "bin_aware": True,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
        "lookback_months": 3,
        "weight_surprise": 1,
        "weight_ps": 0,
        "weight_netdebt": 0,
        "weight_revgrowth": 0,
        "weight_ebitdagrowth": 0
    },
    # 6) Top 30% / Bot 30%, No Bin
    {
        "top_quantile_threshold": 0.30,
        "bottom_quantile_threshold": 0.30,
        "bin_aware": False,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
        "lookback_months": 3,
        "weight_surprise": 1,
        "weight_ps": 0,
        "weight_netdebt": 0,
        "weight_revgrowth": 0,
        "weight_ebitdagrowth": 0
    },
    # 7) Top 40% / Bot 40%, Bin-Aware
    {
        "top_quantile_threshold": 0.40,
        "bottom_quantile_threshold": 0.40,
        "bin_aware": True,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
        "lookback_months": 3,
        "weight_surprise": 1,
        "weight_ps": 0,
        "weight_netdebt": 0,
        "weight_revgrowth": 0,
        "weight_ebitdagrowth": 0
    },
    # 8) Top 40% / Bot 40%, No Bin
    {
        "top_quantile_threshold": 0.40,
        "bottom_quantile_threshold": 0.40,
        "bin_aware": False,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
        "lookback_months": 3,
        "weight_surprise": 1,
        "weight_ps": 0,
        "weight_netdebt": 0,
        "weight_revgrowth": 0,
        "weight_ebitdagrowth": 0
    },
    # 9) Top 50% / Bot 50%, Bin-Aware
    {
        "top_quantile_threshold": 0.50,
        "bottom_quantile_threshold": 0.50,
        "bin_aware": True,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
        "lookback_months": 3,
        "weight_surprise": 1,
        "weight_ps": 0,
        "weight_netdebt": 0,
        "weight_revgrowth": 0,
        "weight_ebitdagrowth": 0
    },
    # 10) Top 50% / Bot 50%, No Bin
    {
        "top_quantile_threshold": 0.50,
        "bottom_quantile_threshold": 0.50,
        "bin_aware": False,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
        "lookback_months": 3,
        "weight_surprise": 1,
        "weight_ps": 0,
        "weight_netdebt": 0,
        "weight_revgrowth": 0,
        "weight_ebitdagrowth": 0
    },
    # 11) Top 80%, Bot 0%, Bin-Aware
    {
        "top_quantile_threshold": 0.80,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": True,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
        "lookback_months": 3,
        "weight_surprise": 1,
        "weight_ps": 0,
        "weight_netdebt": 0,
        "weight_revgrowth": 0,
        "weight_ebitdagrowth": 0
    },
    # 12) Top 50%, Bot 0%, Bin-Aware
    {
        "top_quantile_threshold": 0.50,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": True,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
        "lookback_months": 3,
        "weight_surprise": 1,
        "weight_ps": 0,
        "weight_netdebt": 0,
        "weight_revgrowth": 0,
        "weight_ebitdagrowth": 0
    },
    # 13) Top 10%, Bot 0%, Bin-Aware
    {
        "top_quantile_threshold": 0.10,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": True,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
        "lookback_months": 3,
        "weight_surprise": 1,
        "weight_ps": 0,
        "weight_netdebt": 0,
        "weight_revgrowth": 0,
        "weight_ebitdagrowth": 0
    },
    # 14) Top 80%, Bot 0%, No Bin
    {
        "top_quantile_threshold": 0.80,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": False,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
        "lookback_months": 3,
        "weight_surprise": 1,
        "weight_ps": 0,
        "weight_netdebt": 0,
        "weight_revgrowth": 0,
        "weight_ebitdagrowth": 0
    },
    # 15) Top 50%, Bot 0%, No Bin
    {
        "top_quantile_threshold": 0.50,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": False,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
        "lookback_months": 3,
        "weight_surprise": 1,
        "weight_ps": 0,
        "weight_netdebt": 0,
        "weight_revgrowth": 0,
        "weight_ebitdagrowth": 0
    },
    # 16) Top 10%, Bot 0%, No Bin
    {
        "top_quantile_threshold": 0.10,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": False,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
        "lookback_months": 3,
        "weight_surprise": 1,
        "weight_ps": 0,
        "weight_netdebt": 0,
        "weight_revgrowth": 0,
        "weight_ebitdagrowth": 0
    },
    # 17) Top 100%, Bot 0%, Bin
        {
        "top_quantile_threshold": 1.0,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": True,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
        "lookback_months": 3,
        "weight_surprise": 1,
        "weight_ps": 0,
        "weight_netdebt": 0,
        "weight_revgrowth": 0,
        "weight_ebitdagrowth": 0
    }
]

# This list contains all the sanme quantiles with a lookback period of 2 months
lookback_period = 2
w_surprise = 1
w_ps = 0
w_netdebt = 0
w_revgrowth = 0
w_ebitdagrowth = 0


param_list_lookback_2 = [
    # 1) Top 10% / Bot 10%, Bin-Aware
    {
        "top_quantile_threshold": 0.10,
        "bottom_quantile_threshold": 0.10,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 2) Top 10% / Bot 10%, No Bin
    {
        "top_quantile_threshold": 0.10,
        "bottom_quantile_threshold": 0.10,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 3) Top 20% / Bot 20%, Bin-Aware
    {
        "top_quantile_threshold": 0.20,
        "bottom_quantile_threshold": 0.20,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 4) Top 20% / Bot 20%, No Bin
    {
        "top_quantile_threshold": 0.20,
        "bottom_quantile_threshold": 0.20,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 5) Top 30% / Bot 30%, Bin-Aware
    {
        "top_quantile_threshold": 0.30,
        "bottom_quantile_threshold": 0.30,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 6) Top 30% / Bot 30%, No Bin
    {
        "top_quantile_threshold": 0.30,
        "bottom_quantile_threshold": 0.30,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 7) Top 40% / Bot 40%, Bin-Aware
    {
        "top_quantile_threshold": 0.40,
        "bottom_quantile_threshold": 0.40,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 8) Top 40% / Bot 40%, No Bin
    {
        "top_quantile_threshold": 0.40,
        "bottom_quantile_threshold": 0.40,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 9) Top 50% / Bot 50%, Bin-Aware
    {
        "top_quantile_threshold": 0.50,
        "bottom_quantile_threshold": 0.50,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 10) Top 50% / Bot 50%, No Bin
    {
        "top_quantile_threshold": 0.50,
        "bottom_quantile_threshold": 0.50,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 11) Top 80%, Bot 0%, Bin-Aware
    {
        "top_quantile_threshold": 0.80,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 12) Top 50%, Bot 0%, Bin-Aware
    {
        "top_quantile_threshold": 0.50,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 13) Top 10%, Bot 0%, Bin-Aware
    {
        "top_quantile_threshold": 0.10,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 14) Top 80%, Bot 0%, No Bin
    {
        "top_quantile_threshold": 0.80,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 15) Top 50%, Bot 0%, No Bin
    {
        "top_quantile_threshold": 0.50,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 16) Top 10%, Bot 0%, No Bin
    {
        "top_quantile_threshold": 0.10,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 17) Top 100%, Bot 0%, Bin
    {
        "top_quantile_threshold": 1.0,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
]


#This list contains all the same quantiles/bins with a lookback period of 1 month
lookback_period = 1
w_surprise = 1
w_ps = 0
w_netdebt = 0
w_revgrowth = 0
w_ebitdagrowth = 0


param_list_lookback_1 = [
    # 1) Top 10% / Bot 10%, Bin-Aware
    {
        "top_quantile_threshold": 0.10,
        "bottom_quantile_threshold": 0.10,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 2) Top 10% / Bot 10%, No Bin
    {
        "top_quantile_threshold": 0.10,
        "bottom_quantile_threshold": 0.10,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 3) Top 20% / Bot 20%, Bin-Aware
    {
        "top_quantile_threshold": 0.20,
        "bottom_quantile_threshold": 0.20,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 4) Top 20% / Bot 20%, No Bin
    {
        "top_quantile_threshold": 0.20,
        "bottom_quantile_threshold": 0.20,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 5) Top 30% / Bot 30%, Bin-Aware
    {
        "top_quantile_threshold": 0.30,
        "bottom_quantile_threshold": 0.30,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 6) Top 30% / Bot 30%, No Bin
    {
        "top_quantile_threshold": 0.30,
        "bottom_quantile_threshold": 0.30,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 7) Top 40% / Bot 40%, Bin-Aware
    {
        "top_quantile_threshold": 0.40,
        "bottom_quantile_threshold": 0.40,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 8) Top 40% / Bot 40%, No Bin
    {
        "top_quantile_threshold": 0.40,
        "bottom_quantile_threshold": 0.40,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 9) Top 50% / Bot 50%, Bin-Aware
    {
        "top_quantile_threshold": 0.50,
        "bottom_quantile_threshold": 0.50,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 10) Top 50% / Bot 50%, No Bin
    {
        "top_quantile_threshold": 0.50,
        "bottom_quantile_threshold": 0.50,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 11) Top 80%, Bot 0%, Bin-Aware
    {
        "top_quantile_threshold": 0.80,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 12) Top 50%, Bot 0%, Bin-Aware
    {
        "top_quantile_threshold": 0.50,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 13) Top 10%, Bot 0%, Bin-Aware
    {
        "top_quantile_threshold": 0.10,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 14) Top 80%, Bot 0%, No Bin
    {
        "top_quantile_threshold": 0.80,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 15) Top 50%, Bot 0%, No Bin
    {
        "top_quantile_threshold": 0.50,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 16) Top 10%, Bot 0%, No Bin
    {
        "top_quantile_threshold": 0.10,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 17) Top 100%, Bot 0%, Bin
    {
        "top_quantile_threshold": 1.0,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
]

#This list contains all the same quantiles/bins with a lookback period of 3 months and weighting differences
lookback_period = 3
w_surprise = .6
w_ps = .1
w_netdebt = .1
w_revgrowth = .1
w_ebitdagrowth = .1


param_list_weights_1 = [
    # 1) Top 10% / Bot 10%, Bin-Aware
    {
        "top_quantile_threshold": 0.10,
        "bottom_quantile_threshold": 0.10,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 2) Top 10% / Bot 10%, No Bin
    {
        "top_quantile_threshold": 0.10,
        "bottom_quantile_threshold": 0.10,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 3) Top 20% / Bot 20%, Bin-Aware
    {
        "top_quantile_threshold": 0.20,
        "bottom_quantile_threshold": 0.20,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 4) Top 20% / Bot 20%, No Bin
    {
        "top_quantile_threshold": 0.20,
        "bottom_quantile_threshold": 0.20,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 5) Top 30% / Bot 30%, Bin-Aware
    {
        "top_quantile_threshold": 0.30,
        "bottom_quantile_threshold": 0.30,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 6) Top 30% / Bot 30%, No Bin
    {
        "top_quantile_threshold": 0.30,
        "bottom_quantile_threshold": 0.30,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 7) Top 40% / Bot 40%, Bin-Aware
    {
        "top_quantile_threshold": 0.40,
        "bottom_quantile_threshold": 0.40,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 8) Top 40% / Bot 40%, No Bin
    {
        "top_quantile_threshold": 0.40,
        "bottom_quantile_threshold": 0.40,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 9) Top 50% / Bot 50%, Bin-Aware
    {
        "top_quantile_threshold": 0.50,
        "bottom_quantile_threshold": 0.50,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 10) Top 50% / Bot 50%, No Bin
    {
        "top_quantile_threshold": 0.50,
        "bottom_quantile_threshold": 0.50,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 11) Top 80%, Bot 0%, Bin-Aware
    {
        "top_quantile_threshold": 0.80,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 12) Top 50%, Bot 0%, Bin-Aware
    {
        "top_quantile_threshold": 0.50,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 13) Top 10%, Bot 0%, Bin-Aware
    {
        "top_quantile_threshold": 0.10,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 14) Top 80%, Bot 0%, No Bin
    {
        "top_quantile_threshold": 0.80,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 15) Top 50%, Bot 0%, No Bin
    {
        "top_quantile_threshold": 0.50,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 16) Top 10%, Bot 0%, No Bin
    {
        "top_quantile_threshold": 0.10,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 17) Top 100%, Bot 0%, Bin
    {
        "top_quantile_threshold": 1.0,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
]

#This list contains all the same quantiles/bins with a lookback period of 3 months and weighting differences #2
lookback_period = 3
w_surprise = .9
w_ps = .025
w_netdebt = .025
w_revgrowth = .025
w_ebitdagrowth = .025


param_list_weights_3 = [
    # 1) Top 10% / Bot 10%, Bin-Aware
    {
        "top_quantile_threshold": 0.10,
        "bottom_quantile_threshold": 0.10,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 2) Top 10% / Bot 10%, No Bin
    {
        "top_quantile_threshold": 0.10,
        "bottom_quantile_threshold": 0.10,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 3) Top 20% / Bot 20%, Bin-Aware
    {
        "top_quantile_threshold": 0.20,
        "bottom_quantile_threshold": 0.20,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 4) Top 20% / Bot 20%, No Bin
    {
        "top_quantile_threshold": 0.20,
        "bottom_quantile_threshold": 0.20,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 5) Top 30% / Bot 30%, Bin-Aware
    {
        "top_quantile_threshold": 0.30,
        "bottom_quantile_threshold": 0.30,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 6) Top 30% / Bot 30%, No Bin
    {
        "top_quantile_threshold": 0.30,
        "bottom_quantile_threshold": 0.30,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 7) Top 40% / Bot 40%, Bin-Aware
    {
        "top_quantile_threshold": 0.40,
        "bottom_quantile_threshold": 0.40,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 8) Top 40% / Bot 40%, No Bin
    {
        "top_quantile_threshold": 0.40,
        "bottom_quantile_threshold": 0.40,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 9) Top 50% / Bot 50%, Bin-Aware
    {
        "top_quantile_threshold": 0.50,
        "bottom_quantile_threshold": 0.50,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 10) Top 50% / Bot 50%, No Bin
    {
        "top_quantile_threshold": 0.50,
        "bottom_quantile_threshold": 0.50,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 11) Top 80%, Bot 0%, Bin-Aware
    {
        "top_quantile_threshold": 0.80,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 12) Top 50%, Bot 0%, Bin-Aware
    {
        "top_quantile_threshold": 0.50,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 13) Top 10%, Bot 0%, Bin-Aware
    {
        "top_quantile_threshold": 0.10,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 14) Top 80%, Bot 0%, No Bin
    {
        "top_quantile_threshold": 0.80,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 15) Top 50%, Bot 0%, No Bin
    {
        "top_quantile_threshold": 0.50,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 16) Top 10%, Bot 0%, No Bin
    {
        "top_quantile_threshold": 0.10,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 17) Top 100%, Bot 0%, Bin
    {
        "top_quantile_threshold": 1.0,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
]

lookback_period = 3
w_surprise = .2
w_ps = .2
w_netdebt = .2
w_revgrowth = .2
w_ebitdagrowth = .2


param_list_weights_2 = [
    # 1) Top 10% / Bot 10%, Bin-Aware
    {
        "top_quantile_threshold": 0.10,
        "bottom_quantile_threshold": 0.10,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 2) Top 10% / Bot 10%, No Bin
    {
        "top_quantile_threshold": 0.10,
        "bottom_quantile_threshold": 0.10,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 3) Top 20% / Bot 20%, Bin-Aware
    {
        "top_quantile_threshold": 0.20,
        "bottom_quantile_threshold": 0.20,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 4) Top 20% / Bot 20%, No Bin
    {
        "top_quantile_threshold": 0.20,
        "bottom_quantile_threshold": 0.20,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 5) Top 30% / Bot 30%, Bin-Aware
    {
        "top_quantile_threshold": 0.30,
        "bottom_quantile_threshold": 0.30,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 6) Top 30% / Bot 30%, No Bin
    {
        "top_quantile_threshold": 0.30,
        "bottom_quantile_threshold": 0.30,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 7) Top 40% / Bot 40%, Bin-Aware
    {
        "top_quantile_threshold": 0.40,
        "bottom_quantile_threshold": 0.40,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 8) Top 40% / Bot 40%, No Bin
    {
        "top_quantile_threshold": 0.40,
        "bottom_quantile_threshold": 0.40,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 9) Top 50% / Bot 50%, Bin-Aware
    {
        "top_quantile_threshold": 0.50,
        "bottom_quantile_threshold": 0.50,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 10) Top 50% / Bot 50%, No Bin
    {
        "top_quantile_threshold": 0.50,
        "bottom_quantile_threshold": 0.50,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 11) Top 80%, Bot 0%, Bin-Aware
    {
        "top_quantile_threshold": 0.80,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 12) Top 50%, Bot 0%, Bin-Aware
    {
        "top_quantile_threshold": 0.50,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 13) Top 10%, Bot 0%, Bin-Aware
    {
        "top_quantile_threshold": 0.10,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 14) Top 80%, Bot 0%, No Bin
    {
        "top_quantile_threshold": 0.80,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 15) Top 50%, Bot 0%, No Bin
    {
        "top_quantile_threshold": 0.50,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 16) Top 10%, Bot 0%, No Bin
    {
        "top_quantile_threshold": 0.10,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": False,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
    # 17) Top 100%, Bot 0%, Bin
    {
        "top_quantile_threshold": 1.0,
        "bottom_quantile_threshold": 0.00,
        "bin_aware": True,
        "lookback_months": lookback_period,
        "weight_surprise": w_surprise,
        "weight_ps": w_ps,
        "weight_netdebt": w_netdebt,
        "weight_revgrowth": w_revgrowth,
        "weight_ebitdagrowth": w_ebitdagrowth,
        "stop_loss_pct": 0.10,
        "initial_notional_multiplier": 10,
        "max_weight_per_stock": 0.05,
        "funding_rate_annual": 0.043,
        "borrow_rate_annual": 0.033,
        "min_earnings_announcements": 50,
    },
]

import numpy as np
import pandas as pd
import plotly.express as px
from joblib import Parallel, delayed
from IPython.display import HTML, display


def run_quantile_strategy(
    data,
    plot=True,
    lookback_months=3.0,
    top_quantile_threshold=0.30,
    bottom_quantile_threshold=0.30,
    stop_loss_pct=0.10,
    min_earnings_announcements=50,
    initial_notional_multiplier=10,
    max_weight_per_stock=0.05,
    funding_rate_annual=0.043,
    borrow_rate_annual=None,
    weight_surprise=1,
    weight_ps=0,
    weight_netdebt=0,
    weight_revgrowth=0,
    weight_ebitdagrowth=0,
    bin_aware=True,
    spy_df=None,
    trading_costs=0.0005
):
    if borrow_rate_annual is None:
        borrow_rate_annual = funding_rate_annual - 0.01
    daily_fund_rate   = funding_rate_annual / 252
    daily_borrow_rate = borrow_rate_annual / 252

    df = data.copy()
    if "trade_day" in df.columns and "date" not in df.columns:
        df.rename(columns={"trade_day": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df["trade_day"] = df["date"]

    df["factor"] = (
        weight_surprise * df.get("zscore_surprise_vs_move", 0)
        + weight_ps       * df.get("ps_ratio_bin_z", 0)
        + weight_netdebt  * df.get("net_debt_to_ebitda_bin_z", 0)
        + weight_revgrowth* df.get("revenue_growth_bin_z", 0)
        + weight_ebitdagrowth* df.get("ebitda_growth_bin_z", 0)
    )

    df = df.dropna(subset=["factor", "adj_close"])
    if bin_aware:
        df = df.dropna(subset=["avg_eps_bin"])

    start_date = df["date"].min()
    end_date   = df["date"].max()

    meltdown_days = []
    meltdown_i = 0
    d_ = start_date + pd.DateOffset(months=3*meltdown_i)
    while d_ <= end_date:
        meltdown_days.append(d_)
        meltdown_i += 1
        d_ = start_date + pd.DateOffset(months=3*meltdown_i)

    def months_to_days(m_):
        return int(round(m_ * 30.44))
    partial_days = []
    i_ = 0
    d_ = start_date + pd.DateOffset(months=i_ * lookback_months)
    while d_ <= end_date:
        if i_ > 0:
            partial_days.append(d_)
        i_ += 1
        d_ = start_date + pd.DateOffset(months=i_ * lookback_months)

    all_days = pd.date_range(start_date, end_date, freq="D")
    current_positions = {}
    current_capital   = 0.0
    current_capital_no_tc = 0.0
    meltdown_idx      = 0
    partial_idx       = 0
    daily_records     = []
    trade_records     = []
    stop_loss_count   = 0
    have_invested_before = False
    cumulative_trading_costs = 0.0

    meltdown_first_has_invested   = False
    meltdown_first_invest_day     = None
    meltdown_first_invest_i       = 0
    meltdown_first_long_count     = 0
    meltdown_first_short_count    = 0

    def rank_weighting_local(sub_df, rank_col="within_bin_pct"):
        sub_df = sub_df.sort_values(rank_col)
        sub_df["ordinal_rank"] = np.arange(1, len(sub_df)+1)
        return sub_df

    def apply_trading_cost(notional):
        nonlocal current_capital, cumulative_trading_costs
        if notional>0:
            cost_ = trading_costs * notional
            current_capital -= cost_
            cumulative_trading_costs += cost_

    def meltdown_function(day_):
        nonlocal current_positions, current_capital, have_invested_before
        nonlocal meltdown_first_has_invested, meltdown_first_invest_day
        nonlocal meltdown_first_invest_i, meltdown_first_long_count, meltdown_first_short_count
        nonlocal current_capital_no_tc  # ensure we can modify outside
        close_pnl = 0.0
        sum_close_notional = 0.0
        for tkr, pos_ in list(current_positions.items()):
            side = pos_["side"]
            ep   = pos_["entry_price"]
            lp   = pos_.get("last_price", ep)
            shr  = pos_["shares"]
            notional_ = lp * shr
            sum_close_notional += notional_
            if side == "long":
                close_pnl += (lp - ep) * shr
            else:
                close_pnl += (ep - lp) * shr
            del current_positions[tkr]
        current_capital += close_pnl
        current_capital_no_tc += close_pnl

        apply_trading_cost(sum_close_notional)

        max_i = int(np.floor(3.0 / lookback_months))
        meltdown_invested = False
        for i_ in range(1, max_i + 1):
            accum_m = i_ * lookback_months
            start_  = day_ - pd.DateOffset(days=months_to_days(accum_m))
            rebal_slice = df[(df["date"] > start_) & (df["date"] <= day_)]
            if rebal_slice["ticker"].nunique() >= min_earnings_announcements:
                aggregator = {col: "last" for col in rebal_slice.columns if col not in ("ticker")}
                aggregator["ticker"] = "first"
                if "trade_day" in rebal_slice.columns:
                    aggregator["trade_day"] = "last"
                last_info = rebal_slice.groupby("ticker", as_index=False).agg(aggregator)
                if "trade_day" in last_info.columns:
                    last_info = last_info[last_info["trade_day"] <= day_]
                if bin_aware:
                    last_info["within_bin_pct"] = last_info.groupby("avg_eps_bin")["factor"].rank(pct=True)
                else:
                    last_info["within_bin_pct"] = last_info["factor"].rank(pct=True)
                top_cut    = 1.0 - top_quantile_threshold
                bottom_cut = bottom_quantile_threshold
                top_df     = last_info[last_info["within_bin_pct"] >= top_cut].copy()
                bottom_df  = last_info[last_info["within_bin_pct"] <= bottom_cut].copy()
                if bin_aware:
                    top_df    = top_df.groupby("avg_eps_bin", group_keys=False).apply(rank_weighting_local)
                    bottom_df = bottom_df.groupby("avg_eps_bin", group_keys=False).apply(rank_weighting_local)
                else:
                    top_df    = rank_weighting_local(top_df)
                    bottom_df = rank_weighting_local(bottom_df)
                gross_notional = top_df["adj_close"].sum() + bottom_df["adj_close"].sum()
                if gross_notional <= 0:
                    gross_notional = 1_000_000
                if not have_invested_before:
                    base_cap = initial_notional_multiplier * gross_notional
                    if base_cap < 1.0:
                        base_cap = 1_000.0
                    current_capital      = base_cap
                    current_capital_no_tc = base_cap
                    have_invested_before = True
                total_notional = current_capital * 10.0
                half_notional  = total_notional * 0.5
                top_sum = top_df["ordinal_rank"].sum() or 1
                bot_sum = bottom_df["ordinal_rank"].sum() or 1
                picks_count = 0
                new_positions = {}
                sum_open_notional = 0.0
                for _, row_ in top_df.iterrows():
                    px_ = row_["adj_close"]
                    if px_ <= 0:
                        continue
                    share_weight = (row_["ordinal_rank"] / top_sum) * half_notional
                    share_weight = min(share_weight, max_weight_per_stock * total_notional)
                    if share_weight <= 0:
                        continue
                    shr_ = share_weight / px_
                    sum_open_notional += share_weight
                    new_positions[row_["ticker"]] = {
                        "side": "long",
                        "entry_price": px_,
                        "shares": shr_,
                        "last_price": px_,
                    }
                    picks_count += 1
                for _, row_ in bottom_df.iterrows():
                    px_ = row_["adj_close"]
                    if px_ <= 0:
                        continue
                    share_weight = (row_["ordinal_rank"] / bot_sum) * half_notional
                    share_weight = min(share_weight, max_weight_per_stock * total_notional)
                    if share_weight <= 0:
                        continue
                    shr_ = share_weight / px_
                    sum_open_notional += share_weight
                    new_positions[row_["ticker"]] = {
                        "side": "short",
                        "entry_price": px_,
                        "shares": shr_,
                        "last_price": px_,
                    }
                    picks_count += 1
                current_positions = new_positions
                meltdown_invested = True
                if not meltdown_first_has_invested:
                    meltdown_first_has_invested = True
                    meltdown_first_invest_day   = day_
                    meltdown_first_invest_i     = i_
                    n_long  = sum(1 for v in new_positions.values() if v["side"]=="long")
                    n_short = sum(1 for v in new_positions.values() if v["side"]=="short")
                    meltdown_first_long_count  = n_long
                    meltdown_first_short_count = n_short
                trade_records.append({
                    "date": day_,
                    "num_trades": picks_count,
                    "trade_enacted": f"MeltdownReopen(i={i_})"
                })
                apply_trading_cost(sum_open_notional)
                break
        if not meltdown_invested:
            trade_records.append({
                "date": day_,
                "num_trades": 0,
                "trade_enacted": "MeltdownNoReopen"
            })

    def partial_rerank_function(day_):
        nonlocal current_positions, current_capital, current_capital_no_tc
        start_ = day_ - pd.DateOffset(days=months_to_days(lookback_months))
        rebal_slice = df[(df["date"] > start_) & (df["date"] <= day_)]
        if rebal_slice["ticker"].nunique() < min_earnings_announcements:
            return
        aggregator = {col: "last" for col in rebal_slice.columns if col not in ("ticker")}
        aggregator["ticker"] = "first"
        if "trade_day" in rebal_slice.columns:
            aggregator["trade_day"] = "last"
        last_info = rebal_slice.groupby("ticker", as_index=False).agg(aggregator)
        if "trade_day" in last_info.columns:
            last_info = last_info[last_info["trade_day"] <= day_]
        if bin_aware:
            last_info["within_bin_pct"] = last_info.groupby("avg_eps_bin")["factor"].rank(pct=True)
        else:
            last_info["within_bin_pct"] = last_info["factor"].rank(pct=True)
        top_cut    = 1.0 - top_quantile_threshold
        bottom_cut = bottom_quantile_threshold
        top_df     = last_info[last_info["within_bin_pct"] >= top_cut].copy()
        bottom_df  = last_info[last_info["within_bin_pct"] <= bottom_cut].copy()
        if bin_aware:
            top_df    = top_df.groupby("avg_eps_bin", group_keys=False).apply(rank_weighting_local)
            bottom_df = bottom_df.groupby("avg_eps_bin", group_keys=False).apply(rank_weighting_local)
        else:
            top_df    = rank_weighting_local(top_df)
            bottom_df = rank_weighting_local(bottom_df)
        keep_set = set(top_df["ticker"]).union(bottom_df["ticker"])
        partial_close_pnl = 0.0
        sum_close_notional = 0.0
        for tkr in list(current_positions.keys()):
            if tkr not in keep_set:
                pos_  = current_positions[tkr]
                side  = pos_["side"]
                ep    = pos_["entry_price"]
                lp    = pos_.get("last_price", ep)
                shr   = pos_["shares"]
                close_notional = lp * shr
                sum_close_notional += close_notional
                if side == "long":
                    partial_close_pnl += (lp - ep)*shr
                else:
                    partial_close_pnl += (ep - lp)*shr
                del current_positions[tkr]
        current_capital += partial_close_pnl
        current_capital_no_tc += partial_close_pnl
        apply_trading_cost(sum_close_notional)
        total_notional = current_capital * 10.0
        half_notional  = total_notional * 0.5
        top_sum = top_df["ordinal_rank"].sum() or 1
        bot_sum = bottom_df["ordinal_rank"].sum() or 1
        picks_count = 0
        exist_set   = set(current_positions.keys())
        sum_open_notional = 0.0
        for _, row_ in top_df.iterrows():
            tk_ = row_["ticker"]
            if tk_ in exist_set:
                continue
            px_ = row_["adj_close"]
            if px_ <= 0:
                continue
            share_weight = (row_["ordinal_rank"] / top_sum)*half_notional
            share_weight = min(share_weight, max_weight_per_stock * total_notional)
            if share_weight <= 0:
                continue
            shr_ = share_weight / px_
            sum_open_notional += share_weight
            current_positions[tk_] = {
                "side": "long",
                "entry_price": px_,
                "shares": shr_,
                "last_price": px_,
            }
            picks_count += 1
        for _, row_ in bottom_df.iterrows():
            tk_ = row_["ticker"]
            if tk_ in exist_set:
                continue
            px_ = row_["adj_close"]
            if px_ <= 0:
                continue
            share_weight = (row_["ordinal_rank"] / bot_sum)*half_notional
            share_weight = min(share_weight, max_weight_per_stock * total_notional)
            if share_weight <= 0:
                continue
            shr_ = share_weight / px_
            sum_open_notional += share_weight
            current_positions[tk_] = {
                "side": "short",
                "entry_price": px_,
                "shares": shr_,
                "last_price": px_,
            }
            picks_count += 1
        if picks_count>0:
            trade_records.append({
                "date": day_,
                "num_trades": picks_count,
                "trade_enacted": "PartialRebalance"
            })
        apply_trading_cost(sum_open_notional)

    for day_ in all_days:
        day_df = df[df["date"] == day_][["ticker", "adj_close"]]
        daily_pnl = 0.0
        long_notional_today  = 0.0
        short_notional_today = 0.0
        if current_positions and not day_df.empty:
            pos_list = []
            for tkr, pos_ in current_positions.items():
                pos_list.append([
                    tkr,
                    pos_["side"],
                    pos_["entry_price"],
                    pos_.get("last_price", pos_["entry_price"]),
                    pos_["shares"]
                ])
            positions_df = pd.DataFrame(pos_list, columns=["ticker","side","entry_px","last_px","shares"])
            merged_positions = positions_df.merge(day_df, on="ticker", how="left")
            merged_positions["price_col"] = merged_positions["adj_close"].fillna(merged_positions["last_px"])
            side_sign = np.where(merged_positions["side"]=="long", 1, -1)
            px_change = merged_positions["price_col"] - merged_positions["last_px"]
            daily_pnl_sum = (px_change*merged_positions["shares"]*side_sign).sum()
            daily_pnl += daily_pnl_sum
            merged_positions["today_notional"] = merged_positions["price_col"]*merged_positions["shares"]
            long_notional_today  = merged_positions.loc[merged_positions["side"]=="long","today_notional"].sum()
            short_notional_today = merged_positions.loc[merged_positions["side"]=="short","today_notional"].sum()
            rel_move = np.where(
                merged_positions["side"]=="long",
                (merged_positions["price_col"]-merged_positions["entry_px"])/merged_positions["entry_px"],
                (merged_positions["entry_px"]-merged_positions["price_col"])/merged_positions["entry_px"]
            )
            stop_mask = (rel_move <= -stop_loss_pct)
            if stop_mask.any():
                stopped = merged_positions[stop_mask]
                side_s  = np.where(stopped["side"]=="long", 1, -1)
                stop_pnl = ((stopped["price_col"]-stopped["entry_px"])*stopped["shares"]*side_s).sum()
                daily_pnl += stop_pnl
                sum_stop_notional = (stopped["price_col"]*stopped["shares"]).sum()
                for idx_s in stopped.index:
                    st_tkr = stopped.loc[idx_s,"ticker"]
                    if st_tkr in current_positions:
                        del current_positions[st_tkr]
                stop_loss_count += int(len(stopped))
                trade_records.append({
                    "date": day_,
                    "num_trades": int(len(stopped)),
                    "trade_enacted": "StopOut"
                })
                apply_trading_cost(sum_stop_notional)
                merged_positions = merged_positions.loc[~stop_mask]
            merged_positions["last_px"] = merged_positions["price_col"]
            new_dict={}
            for _, row_ in merged_positions.iterrows():
                new_dict[row_["ticker"]] = {
                    "side": row_["side"],
                    "entry_price": row_["entry_px"],
                    "shares": row_["shares"],
                    "last_price": row_["last_px"]
                }
            current_positions = new_dict
        current_capital += daily_pnl
        current_capital_no_tc += daily_pnl
        cost_long  = long_notional_today*daily_fund_rate
        cost_short = short_notional_today*daily_borrow_rate
        current_capital -= (cost_long + cost_short)
        current_capital_no_tc -= (cost_long + cost_short)
        if meltdown_idx<len(meltdown_days) and day_>=meltdown_days[meltdown_idx]:
            meltdown_function(day_)
            meltdown_idx +=1
        if partial_idx<len(partial_days) and day_>=partial_days[partial_idx]:
            partial_rerank_function(day_)
            partial_idx+=1
        daily_records.append({
            "date": day_,
            "portfolio_val": current_capital,
            "portfolio_val_no_tc": current_capital_no_tc
        })

    df_portfolio = pd.DataFrame(daily_records).drop_duplicates("date").sort_values("date")
    idx_port = pd.DatetimeIndex(df_portfolio["date"])
    if spy_df is not None:
        spy_ = spy_df.copy()
        spy_["date"] = pd.to_datetime(spy_["date"])
        spy_.sort_values("date", inplace=True)
        idx_spy = pd.DatetimeIndex(spy_["date"])
        date_union = idx_port.union(idx_spy)
    else:
        date_union = idx_port
    df_portfolio = (
        df_portfolio.set_index("date")
        .reindex(date_union)
        .ffill()
        .reset_index()
        .rename(columns={"index":"date"})
    )
    df_portfolio["date"] = pd.to_datetime(df_portfolio["date"])
    first_invest_ix = df_portfolio.index[df_portfolio["portfolio_val"]>0]
    if len(first_invest_ix)<2:
        df_portfolio["ret"] = 0.0
        final_pnl   = 0.0
        tot_return  = 0.0
        total_days  = (df_portfolio["date"].iloc[-1]-df_portfolio["date"].iloc[0]).days
        if total_days<1:
            total_days=1
        risk_summary = {
            "initial_capital": 0.0,
            "first_trade_date": None,
            "end_date": df_portfolio["date"].iloc[-1].date(),
            "final_portfolio_value": 0.0,
            "cumulative_pnl": 0.0,
            "cumulative_pnl_no_tc": 0.0,
            "total_trading_costs": 0.0,
            "total_return": 0.0,
            "daily_mean_ret": 0.0,
            "annual_mean_ret": 0.0,
            "daily_vol": 0.0,
            "annual_vol": 0.0,
            "daily_sharpe_ratio": 0.0,
            "annual_sharpe_ratio": 0.0,
            "daily_sortino_ratio": 0.0,
            "annual_sortino_ratio": 0.0,
            "downside_beta": 0.0,
            "kurtosis": 0.0,
            "skewness": 0.0,
            "VaR(0.05)": 0.0,
            "cvar": 0.0,
            "total_days": total_days,
            "total_rebalances": 0,
            "total_trades_made": 0,
            "max_drawdown": 0.0,
            "stop_loss_selloff_count": stop_loss_count,
            "did_wait_for_lookback": "No",
            "initial_long_count": 0,
            "initial_short_count": 0,
        }
        df_trades = pd.DataFrame(trade_records).sort_values("date") if trade_records else pd.DataFrame()
        df_risk   = pd.DataFrame({"metric": list(risk_summary.keys()), "value": list(risk_summary.values())})
        if not plot:
            return df_portfolio, df_trades, df_risk
        fig = px.line(df_portfolio, x="date", y="portfolio_val", title="No Trades Occurred")
        fig.show()
        display(df_risk)
        return df_portfolio, df_trades, df_risk
    start_ix = first_invest_ix[0]
    sub_df   = df_portfolio.loc[start_ix:].copy()
    sub_df["ret"] = sub_df["portfolio_val"].pct_change().fillna(0)
    sub_df["ret"].iloc[0] = 0.0
    df_portfolio["ret"] = 0.0
    df_portfolio.loc[start_ix:,"ret"] = sub_df["ret"]
    val_col = sub_df["portfolio_val"].dropna()
    if len(val_col)>0:
        start_val = val_col.iloc[0]
        end_val   = val_col.iloc[-1]
        final_pnl = end_val - start_val
    else:
        start_val = np.nan
        end_val   = np.nan
        final_pnl = np.nan
    total_days = (sub_df["date"].iloc[-1] - sub_df["date"].iloc[0]).days
    if total_days<1:
        total_days=1
    df_portfolio["cum_ret"] = (1+df_portfolio["ret"]).cumprod()-1
    tot_return = df_portfolio["cum_ret"].iloc[-1]
    def compute_risk_metrics(returns):
        if len(returns)<2:
            return {
                "daily_mean_ret":0.0,"daily_vol":0.0,
                "daily_sharpe_ratio":0.0,"daily_sortino_ratio":0.0,
                "max_drawdown":0.0,"skew":0.0,"kurt":0.0,
                "VaR(.05)":0.0,"cvar":0.0
            }
        mean_ret_daily = returns.mean()
        vol_daily      = returns.std()
        skew_ = returns.skew()
        kurt_ = returns.kurt()
        cum  = (1+returns).cumprod()
        peak = cum.cummax()
        dd   = (peak-cum)/peak
        max_dd = dd.max() if len(dd) else 0.0
        var05 = -np.percentile(returns,5)
        sharpe_daily = 0.0
        if vol_daily>1e-12:
            sharpe_daily = mean_ret_daily/vol_daily
        neg_rets = returns[returns<0].sort_values()
        cvar_ = 0.0
        if len(neg_rets)>0:
            cutoff = int(np.ceil(0.95*len(neg_rets)))
            cvar_  = abs(np.mean(neg_rets[:cutoff]))
        sortino_daily=0.0
        if len(neg_rets)>1:
            neg_std = neg_rets.std()
            if neg_std>1e-12:
                sortino_daily = (mean_ret_daily / neg_std)*np.sqrt(252)
        return {
            "daily_mean_ret": mean_ret_daily,
            "daily_vol": vol_daily,
            "daily_sharpe_ratio": sharpe_daily,
            "daily_sortino_ratio": sortino_daily,
            "max_drawdown": max_dd,
            "skew": skew_,
            "kurt": kurt_,
            "VaR(.05)": var05,
            "cvar": cvar_
        }
    risk_vals     = compute_risk_metrics(sub_df["ret"])
    daily_mean_ret= risk_vals["daily_mean_ret"]
    annual_mean_ret= daily_mean_ret*252
    daily_vol     = risk_vals["daily_vol"]
    annual_vol    = daily_vol*(252.0**0.5)
    daily_sharpe  = risk_vals["daily_sharpe_ratio"]
    annual_sharpe = daily_sharpe*(252.0**0.5) if daily_sharpe!=0 else 0.0
    daily_sortino = risk_vals["daily_sortino_ratio"]
    annual_sortino= daily_sortino*(252.0**0.5) if daily_sortino!=0 else 0.0
    max_dd        = risk_vals["max_drawdown"]
    skew          = risk_vals["skew"]
    kurt          = risk_vals["kurt"]
    var_          = risk_vals["VaR(.05)"]
    cvar_         = risk_vals["cvar"]
    ds_beta=0.0
    if spy_df is not None:
        spy_ = spy_df.copy()
        spy_["date"] = pd.to_datetime(spy_["date"])
        spy_.sort_values("date", inplace=True)
        spy_ = (
            spy_.set_index("date")
            .reindex(df_portfolio["date"])
            .ffill()
            .reset_index()
            .rename(columns={"index":"date"})
        )
        spy_["spy_ret"] = spy_.groupby("ticker")["adj_close"].pct_change().fillna(0)
        merged_port = pd.merge(
            sub_df[["date","ret"]],
            spy_[["date","spy_ret"]],
            on="date", how="inner"
        ).dropna()
        if len(merged_port)>1:
            dmask   = merged_port["spy_ret"]<0
            df_down = merged_port[dmask]
            if len(df_down)>1:
                cov_  = np.cov(df_down["ret"],df_down["spy_ret"])[0,1]
                var_2 = np.var(df_down["spy_ret"])
                if var_2>1e-12:
                    ds_beta= cov_/var_2
    df_trades = pd.DataFrame(trade_records).sort_values("date") if trade_records else pd.DataFrame()
    meltdown_events = df_trades.loc[df_trades["trade_enacted"].str.contains("Meltdown"),"num_trades"].count() if not df_trades.empty else 0
    partial_events  = df_trades.loc[df_trades["trade_enacted"].str.contains("PartialRebalance"),"num_trades"].count() if not df_trades.empty else 0
    total_trades    = df_trades["num_trades"].sum() if not df_trades.empty else 0
    if meltdown_first_invest_i>1:
        wait_for_lookback_str = "Yes"
    else:
        wait_for_lookback_str = "No"
    final_capital_value = df_portfolio["portfolio_val"].iloc[-1]
    initial_capital_value = df_portfolio["portfolio_val"].iloc[start_ix]
    final_pnl_net = final_capital_value - initial_capital_value
    final_portfolio_value_no_tc = df_portfolio["portfolio_val_no_tc"].iloc[-1]
    initial_capital_value_no_tc = df_portfolio["portfolio_val_no_tc"].iloc[start_ix]
    final_pnl_no_tc = final_portfolio_value_no_tc - initial_capital_value_no_tc
    risk_summary = {
        "initial_capital": initial_capital_value,
        "first_trade_date": sub_df["date"].iloc[0].date(),
        "end_date": sub_df["date"].iloc[-1].date(),
        "final_portfolio_value": final_capital_value,
        "cumulative_pnl": final_pnl_net,
        "cumulative_pnl_no_tc": final_pnl_no_tc,
        "total_trading_costs": cumulative_trading_costs,
        "total_return": tot_return,
        "daily_mean_ret": daily_mean_ret,
        "annual_mean_ret": annual_mean_ret,
        "daily_vol": daily_vol,
        "annual_vol": annual_vol,
        "daily_sharpe_ratio": daily_sharpe,
        "annual_sharpe_ratio": annual_sharpe,
        "daily_sortino_ratio": daily_sortino,
        "annual_sortino_ratio": annual_sortino,
        "downside_beta": ds_beta,
        "kurtosis": kurt,
        "skewness": skew,
        "VaR(0.05)": var_,
        "cvar": cvar_,
        "total_days": total_days,
        "total_rebalances": meltdown_events+partial_events,
        "total_trades_made": total_trades,
        "max_drawdown": max_dd,
        "stop_loss_selloff_count": stop_loss_count,
        "did_wait_for_lookback": wait_for_lookback_str,
        "initial_long_count": meltdown_first_long_count,
        "initial_short_count": meltdown_first_short_count
    }
    df_risk = pd.DataFrame({"metric": list(risk_summary.keys()), "value": list(risk_summary.values())})
    if not plot:
        return df_portfolio, df_trades, df_risk
    fig = px.line(
        df_portfolio,
        x="date",
        y="portfolio_val",
        title="Bin-Aware L/S Quantile Strategy Performance",
        labels={"date":"Date","portfolio_val":"Portfolio Value (USD)"}
    )
    fig.update_layout(
        template="plotly_white",
        title_font_size=18,
        title_x=0.5,
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),
    )
    fig.show()
    label_map = {
        "initial_capital": "Initial Capital",
        "first_trade_date": "First Trade Date",
        "end_date": "End Date",
        "final_portfolio_value": "Final Portfolio Value",
        "cumulative_pnl": "Cumulative P&L",
        "cumulative_pnl_no_tc": "Cumulative P&L (No Trading Costs)",
        "total_trading_costs": "Total Trading Costs",
        "total_return": "Total Return",
        "daily_mean_ret": "Daily Mean Return",
        "annual_mean_ret": "Annual Mean Return",
        "daily_vol": "Daily Volatility",
        "annual_vol": "Annual Volatility",
        "daily_sharpe_ratio": "Daily Sharpe Ratio",
        "annual_sharpe_ratio": "Annual Sharpe Ratio",
        "daily_sortino_ratio": "Daily Sortino Ratio",
        "annual_sortino_ratio": "Annual Sortino Ratio",
        "downside_beta": "Downside Beta",
        "kurtosis": "Kurtosis",
        "skewness": "Skewness",
        "VaR(0.05)": "VaR(0.05)",
        "cvar": "CVaR",
        "total_days": "Total Trading Days",
        "total_rebalances": "Rebalances",
        "total_trades_made": "Trades Made",
        "max_drawdown": "Max Drawdown",
        "stop_loss_selloff_count": "Stop Loss Selloff Count",
        "did_wait_for_lookback": "Waited for Lookback?",
        "initial_long_count": "Initial Long Count",
        "initial_short_count": "Initial Short Count",
    }
    integer_metrics = {
        "total_days",
        "total_rebalances",
        "total_trades_made",
        "stop_loss_selloff_count",
        "initial_long_count",
        "initial_short_count"
    }
    percent_metrics = {
        "total_return",
        "daily_mean_ret",
        "annual_mean_ret",
        "max_drawdown"
    }
    def _format_value(metric_key, val):
        if isinstance(val, (int,float)) and pd.notna(val):
            if metric_key in integer_metrics:
                return f"{int(round(val))}"
            elif metric_key in percent_metrics:
                return f"{val*100:.2f}%"
            else:
                return f"{val:.2f}"
        elif isinstance(val, pd.Timestamp):
            return val.date().isoformat()
        elif isinstance(val, (pd._libs.tslibs.nattype.NaTType, type(None))):
            return "-"
        else:
            return str(val)
    table_html = """
    <div style="display:inline-block; border:2px solid #2C3E50; padding:15px; border-radius:10px; margin:20px auto;">
      <h3 style='color:#2C3E50; font-family:Arial; text-align:center; margin-top:0;'>Strategy Risk Summary</h3>
      <table style="border-collapse:collapse; font-family:Arial, sans-serif; margin:0 auto; width:auto;">
        <thead style="background-color:#f0f0f0;">
          <tr>
            <th style="padding:6px 10px; border-bottom:2px solid #ccc; text-align:left; white-space:nowrap;">Metric</th>
            <th style="padding:6px 10px; border-bottom:2px solid #ccc; text-align:left; white-space:nowrap;">Value</th>
          </tr>
        </thead>
        <tbody>
    """
    for _, row_ in df_risk.iterrows():
        raw_metric = row_["metric"]
        metric_label = label_map.get(raw_metric, raw_metric)
        val_ = row_["value"]
        val_str = _format_value(raw_metric, val_)
        table_html += f"""
          <tr style="background-color:#FBFCFD;">
            <td style="padding:6px 10px; border-bottom:1px solid #ccc; white-space:nowrap;">{metric_label}</td>
            <td style="padding:6px 10px; border-bottom:1px solid #ccc; white-space:nowrap;">{val_str}</td>
          </tr>
        """
    table_html += """
        </tbody>
      </table>
    </div>
    """
    display(HTML(table_html))
    return df_portfolio, df_trades, df_risk





def _run_one_param_dict(data, spy_data, param_dict):
    """
    Helper to run run_quantile_strategy for one set of parameters,
    build a short label from top_q, bot_q, bin_aware, etc.
    Returns (df_portfolio, risk_dict, label)
    """
    top_q = param_dict.get("top_quantile_threshold", 0)
    bot_q = param_dict.get("bottom_quantile_threshold", 0)
    bin_flag = param_dict.get("bin_aware", False)

    label = f"Top {top_q*100:.0f}%/Bot {bot_q*100:.0f}% "
    label += "(Bin)" if bin_flag else "(NoBin)"

    df_port, df_trades, df_risk = run_quantile_strategy(
        data=data,
        spy_df=spy_data,
        plot=False,
        **param_dict
    )
    df_port["ParamSet"] = label
    risk_dict = dict(zip(df_risk["metric"], df_risk["value"]))

    return df_port, risk_dict, label

def parallel_param_sweep(data, spy_data, param_list):
    """
    param_list = [ {top_quantile_threshold:..., bottom_quantile_threshold:..., bin_aware:..., ...}, ... ]
    
    1. Runs each param dict in parallel.
    2. Combines results in a multi-line Plotly figure.
    3. Builds a Model-7-like HTML table with key metrics in rows and param sets in columns.
    4. Returns df_summary (a DataFrame of metrics).
    """

    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(_run_one_param_dict)(data, spy_data, p_dict) for p_dict in param_list
    )

    all_portfolios = []
    all_risk_dicts = []
    param_labels   = []
    for df_port, risk_dict, label in results:
        all_portfolios.append(df_port)
        all_risk_dicts.append(risk_dict)
        param_labels.append(label)

    df_all_runs = pd.concat(all_portfolios, ignore_index=True)
    fig = px.line(
        df_all_runs,
        x="date",
        y="portfolio_val",
        color="ParamSet",
        title="Multi-Run Quantile & Bin-Aware Sweep",
        labels={"date": "Date", "portfolio_val": "Portfolio Value (USD)"},
    )
    fig.update_layout(
        template="plotly_white",
        title_x=0.5,
        title_font_size=18,
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),
    )
    fig.show()

    ordered_metrics = [
        "initial_capital",
        "first_trade_date",
        "end_date",
        "final_portfolio_value",
        "cumulative_pnl",
        "cumulative_pnl_no_tc",
        "total_trading_costs",
        "total_return",
        "daily_mean_ret",
        "annual_mean_ret",
        "daily_vol",
        "annual_vol",
        "daily_sharpe_ratio",
        "annual_sharpe_ratio",
        "daily_sortino_ratio",
        "annual_sortino_ratio",
        "downside_beta",
        "kurtosis",
        "skewness",
        "VaR(0.05)",
        "cvar",
        "total_days",
        "total_rebalances",
        "total_trades_made",
        "max_drawdown",
        "stop_loss_selloff_count",
        "did_wait_for_lookback",
        "initial_long_count",
        "initial_short_count",
    ]

    df_list_for_concat = []
    for lbl, rd in zip(param_labels, all_risk_dicts):
        rows = []
        for m in ordered_metrics:
            rows.append((m, rd.get(m, None)))
        small_df = pd.DataFrame(rows, columns=["metric", "value"]).set_index("metric")
        small_df.rename(columns={"value": lbl}, inplace=True)
        df_list_for_concat.append(small_df)
    df_summary = pd.concat(df_list_for_concat, axis=1)

    label_map = {
        "initial_capital": "Initial Capital",
        "first_trade_date": "First Trade Date",
        "end_date": "End Date",
        "final_portfolio_value": "Final Portfolio Value",
        "cumulative_pnl": "Cumulative P&L",
        "cumulative_pnl_no_tc": "Cumulative P&L (No Trading Costs)",
        "total_trading_costs": "Total Trading Costs",
        "total_return": "Total Return",
        "daily_mean_ret": "Daily Mean Return",
        "annual_mean_ret": "Annual Mean Return",
        "daily_vol": "Daily Volatility",
        "annual_vol": "Annual Volatility",
        "daily_sharpe_ratio": "Daily Sharpe Ratio",
        "annual_sharpe_ratio": "Annual Sharpe Ratio",
        "daily_sortino_ratio": "Daily Sortino Ratio",
        "annual_sortino_ratio": "Annual Sortino Ratio",
        "downside_beta": "Downside Beta",
        "kurtosis": "Kurtosis",
        "skewness": "Skewness",
        "VaR(0.05)": "VaR(0.05)",
        "cvar": "CVaR",
        "total_days": "Total Trading Days",
        "total_rebalances": "Rebalances",
        "total_trades_made": "Trades Made",
        "max_drawdown": "Max Drawdown",
        "stop_loss_selloff_count": "Stop Loss Selloff Count",
        "did_wait_for_lookback": "Waited for Lookback?",
        "initial_long_count": "Initial Long Count",
        "initial_short_count": "Initial Short Count",
    }
    df_summary.rename(index=label_map, inplace=True)

    # Build a simple formatting approach
    integer_like = {
        "Total Trading Days",
        "Rebalances",
        "Trades Made",
        "Stop Loss Selloff Count",
        "Initial Long Count",
        "Initial Short Count",
    }
    money_like = {
        "Initial Capital",
        "Final Portfolio Value",
        "Cumulative P&L",
        "Cumulative P&L (No Trading Costs)",
        "Total Trading Costs",
    }
    percent_like = {
        "Total Return",
        "Daily Mean Return",
        "Annual Mean Return",
        "Daily Volatility",
        "Annual Volatility",
        "Max Drawdown",
        "VaR(0.05)",
        "CVaR",
    }

    def _fmt_val(row_label, val):
        if val is None or pd.isna(val):
            return ""
        if row_label in {"First Trade Date", "End Date"}:
            if isinstance(val, pd.Timestamp):
                return str(val.date())
            return str(val)
        if row_label == "Waited for Lookback?":
            return str(val)
        if row_label in integer_like:
            return f"{int(round(val))}"
        if row_label in money_like:
            return f"{val:,.2f}"
        if row_label in percent_like:
            return f"{val*100:.2f}%"
        return f"{val:.2f}"

    table_html = """
    <div style="display:inline-block; border:2px solid #2C3E50; padding:15px; border-radius:10px; margin:20px auto;">
      <h3 style='color:#2C3E50; font-family:Arial; text-align:center; margin-top:0;'>Multi-Run Parameter Sweep Summary</h3>
      <table style="border-collapse:collapse; font-family:Arial, sans-serif; margin:0 auto; width:auto;">
        <thead style="background-color:#f0f0f0;">
          <tr>
            <th style="padding:6px 10px; border-bottom:2px solid #ccc; text-align:left; white-space:nowrap;">Metric</th>
    """
    for col_lbl in df_summary.columns:
        table_html += f"""
            <th style="padding:6px 10px; border-bottom:2px solid #ccc; text-align:left; white-space:nowrap;">{col_lbl}</th>
        """
    table_html += """
          </tr>
        </thead>
        <tbody>
    """
    for row_lbl in df_summary.index:
        table_html += f"""
          <tr style="background-color:#FBFCFD;">
            <td style="padding:6px 10px; border-bottom:1px solid #ccc; white-space:nowrap;">{row_lbl}</td>
        """
        for col_lbl in df_summary.columns:
            val_ = df_summary.loc[row_lbl, col_lbl]
            val_str = _fmt_val(row_lbl, val_)
            table_html += f"""
            <td style="padding:6px 10px; border-bottom:1px solid #ccc; white-space:nowrap;">{val_str}</td>
            """
        table_html += "</tr>"
    table_html += """
        </tbody>
      </table>
    </div>
    """
    display(HTML(table_html))

    return df_summary

def combine_sweeps_and_show_topN(
    list_of_dfs,
    chosen_metric="Annual Sharpe Ratio",
    top_n=10,
    plot_metric_as_bar=None,
    approach_names=None,
):
    """
    Merges multiple sweep-summary DataFrames (each with metrics as rows, 
    strategies as columns), ensures unique column names, then picks
    the top N columns by a chosen metric (e.g., "Annual Sharpe Ratio").
    
    Parameters
    ----------
    list_of_dfs : List of pd.DataFrame
        Each df is your sweep summary DataFrame (metrics as rows, strategies as columns).
        
    chosen_metric : str
        Which row label to sort on (e.g. "Annual Sharpe Ratio").
        
    top_n : int
        How many columns to keep after sorting by chosen_metric.
        
    plot_metric_as_bar : str or None
        If provided, we'll create a bar chart of that row for the top N columns.
        Typically you might set it to the same as chosen_metric.
        
    approach_names : List[str] or None
        If provided, must be the same length as list_of_dfs.
        We'll append each approach name to each df's columns to ensure uniqueness 
        across merges. If None, we'll auto-generate names like "Approach #1", etc.
    
    Returns
    -------
    df_top : pd.DataFrame
        A reduced DataFrame of only the top N columns (strategies) by chosen_metric,
        but includes all row metrics for those columns.
    
    Also displays an HTML table.
    """
    
    if approach_names is None:
        approach_names = [f"Approach #{i+1}" for i in range(len(list_of_dfs))]
    else:
        if len(approach_names) != len(list_of_dfs):
            raise ValueError("approach_names must have the same length as list_of_dfs")
    
    renamed_dfs = []
    for i, df in enumerate(list_of_dfs):
        df_renamed = df.copy()
        approach_name = approach_names[i]
        
        new_cols = []
        for c in df_renamed.columns:
            new_cols.append(f"{c} — {approach_name}")
        df_renamed.columns = new_cols
        
        renamed_dfs.append(df_renamed)
    
    df_combined = pd.concat(renamed_dfs, axis=1)

    if chosen_metric not in df_combined.index:
        raise ValueError(f"Chosen metric '{chosen_metric}' not found in the combined index.")
    
    metric_series = df_combined.loc[chosen_metric]
    metric_series = pd.to_numeric(metric_series, errors="coerce")
    
    sorted_cols = metric_series.sort_values(ascending=False).index
    
    top_cols = sorted_cols[:top_n]

    df_top = df_combined[top_cols]
    
    
    integer_like = {
        "Total Trading Days",
        "Rebalances",
        "Trades Made",
        "Stop Loss Selloff Count",
    }
    money_like = {
        "Final Portfolio Value",
        "Cumulative P&L",
        "Initial Capital",
    }
    percent_like = {
        "Total Return",
        "Daily Mean Return",
        "Annual Mean Return",
        "Daily Volatility",
        "Annual Volatility",
        "Max Drawdown",
        "VaR(0.05)",
        "CVaR",
    }

    def _format_val(row_label, val):
        if pd.isna(val):
            return ""
        
        if row_label == "Waited for Lookback?":
            return "Yes" if bool(val) else "No"
        
        if row_label in {"First Trade Date", "End Date"}:
            if isinstance(val, pd.Timestamp):
                return str(val.date())
            return str(val)
        
        if row_label in integer_like:
            return f"{int(round(val))}"
        
        if row_label in money_like:
            return f"{val:,.2f}"
        
        if row_label in percent_like:
            return f"{val*100:.2f}%"
        
        if isinstance(val, float):
            return f"{val:.2f}"
        
        return str(val)
    
    table_html = f"""
    <div style="display:inline-block; border:2px solid #2C3E50; padding:15px; border-radius:10px; margin:20px auto;">
      <h3 style='color:#2C3E50; font-family:Arial; text-align:center; margin-top:0;'>
        Top {top_n} Strategies by {chosen_metric}
      </h3>
      <table style="border-collapse:collapse; font-family:Arial, sans-serif; margin:0 auto; width:auto;">
        <thead style="background-color:#f0f0f0;">
          <tr>
            <th style="padding:6px 10px; border-bottom:2px solid #ccc; text-align:left; white-space:nowrap;">Metric</th>
    """
    
    for col_lbl in df_top.columns:
        table_html += f"""
            <th style="padding:6px 10px; border-bottom:2px solid #ccc; text-align:left; white-space:nowrap;">{col_lbl}</th>
        """
    table_html += """
          </tr>
        </thead>
        <tbody>
    """
    
    for row_lbl in df_top.index:
        table_html += f"""
          <tr style="background-color:#FBFCFD;">
            <td style="padding:6px 10px; border-bottom:1px solid #ccc; white-space:nowrap;">{row_lbl}</td>
        """
        for col_lbl in df_top.columns:
            val_ = df_top.loc[row_lbl, col_lbl]
            val_str = _format_val(row_lbl, val_)
            table_html += f"""
            <td style="padding:6px 10px; border-bottom:1px solid #ccc; white-space:nowrap;">{val_str}</td>
            """
        table_html += "</tr>"
    table_html += """
        </tbody>
      </table>
    </div>
    """
    display(HTML(table_html))
    
    if plot_metric_as_bar is not None:
        if plot_metric_as_bar not in df_top.index:
            print(f"Warning: '{plot_metric_as_bar}' row not found, skipping bar plot.")
        else:
            plot_series = pd.to_numeric(df_top.loc[plot_metric_as_bar], errors="coerce")
            df_plot = pd.DataFrame({"Strategy": plot_series.index, plot_metric_as_bar: plot_series.values})
            fig = px.bar(
                df_plot, 
                x="Strategy", 
                y=plot_metric_as_bar,
                title=f"Top {top_n} by '{chosen_metric}' (Plot of '{plot_metric_as_bar}')",
                text_auto='.2s'
            )
            fig.update_layout(
                xaxis_title="Strategy",
                yaxis_title=plot_metric_as_bar,
                template="plotly_white",
                showlegend=False
            )
            fig.update_traces(textposition='outside')
            fig.show()
    
    return df_top

def plot_risk_return_bubble(
    df_all,
    return_col="Annual Mean Return",
    vol_col="Annual Volatility",
    dd_col="Max Drawdown",
    sharpe_col="Annual Sharpe Ratio",
    approach_col="Approach",
    paramset_col="ParamSet",
    title="Risk–Return Bubble Chart"
):
    df_plot = df_all.copy()

    def safe_str_to_numeric(series):
        if pd.api.types.is_numeric_dtype(series):
            return series
        else:
            series_str = series.astype(str).str.replace("%", "").str.replace(",", "")
            return pd.to_numeric(series_str, errors="coerce")

    for c in [return_col, vol_col, dd_col, sharpe_col]:
        df_plot[c] = safe_str_to_numeric(df_plot[c])

    fig = px.scatter(
        df_plot,
        x=vol_col,
        y=return_col,
        size=dd_col,               
        color=sharpe_col,          
        hover_data=[approach_col, paramset_col],
        template="plotly_white",
        title=title,
        size_max=50
    )

    fig.update_layout(
        xaxis_title=vol_col,
        yaxis_title=return_col,
        coloraxis_colorbar=dict(title=sharpe_col),
        hovermode="closest"
    )

    fig.show()

# Function for combining results of different variations of bin-strategy
def combine_approaches(approach_dfs):
    combined_list = []
    for approach_name, df_ in approach_dfs.items():
        df_t = df_.T.copy()  
        df_t["Approach"] = approach_name
        df_t["ParamSet"] = df_t.index
        combined_list.append(df_t)

    df_combined = pd.concat(combined_list, axis=0, ignore_index=True)

    return df_combined

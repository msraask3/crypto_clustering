import pandas as pd
import numpy as np

def get_cluster_time_series(preds, df_x):
    """
    INPUT
        preds (numpy ndarray): cluster labels of each cryptocurrency
        df_x (pandas DataFrame): row = cryptocurrency, column = day
    """
    # get time series of each labels from df_x using df_pred
    labels = np.unique(preds)
    ts_dict = {}

    for l in labels:
        mask = preds == l
        x_l = df_x.loc[df_x.index[mask]]
        ts_dict[l] = x_l
    return ts_dict

def get_rep_ts(labels):
    # 데이터
    df_lr = pd.read_csv("coin_data/coin_log_return.csv")

    # Market capitalisation
    df_m1 = pd.read_csv("coin_data/coingecko_usd_market_caps_100.csv")
    df_m2 = pd.read_csv("coin_data/coingecko_usd_market_caps_200.csv")
    df_m = pd.merge(df_m1, df_m2, on="uts").drop(columns="uts")

    zero_cap_tokens = ["tokenize-xchange", "klay-token", "trust-wallet-token", "xdce-crowd-sale", "celo", "theta-fuel", "amp-token", "defichain", "constellation-labs", "ecomi", "link", "coinex-token"]

    for zct in zero_cap_tokens:
        df_m[zct] = 0.
        
    df_weighted_lr = df_m * df_lr
    df_weighted_lr = df_weighted_lr.T
    
    # 클러스터별 로그 수익률 뭉치
    cluster_weighted_lr = get_cluster_time_series(labels, df_weighted_lr)
    cluster_market_cap = get_cluster_time_series(labels, df_m.T)
    
    # 군집 대표 시계열 (Market-cap weighted)
    cluster_time_series = {}
    for i, j in zip(cluster_weighted_lr, cluster_market_cap):
        N = cluster_market_cap[j].sum(axis=0)
        Sigma_wx = cluster_weighted_lr[i].sum(axis=0)
        bar_x = Sigma_wx / N
        cluster_time_series[i] = bar_x
        
    # 평균 로그 수익률 시계열 데이터 프레임
    df_ts = pd.DataFrame.from_dict(cluster_time_series)
    
    return df_ts
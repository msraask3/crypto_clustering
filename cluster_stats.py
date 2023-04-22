import pandas as pd
import numpy as np

def rank_matching(df_stat, by="log_return_mean[%]"):
    """ 순위 매칭 """
    
    # 공통 군집 레이블
    common_cluster_label = ["A", "B", "C", "D"]

    df_stat = df_stat.sort_values(by=by, ascending=False)

    df_stat["common_cluster_label"] = common_cluster_label

    return df_stat

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


# 레이블만 제공하면 됨.
# 연율화 수익률, 연율화 로그 수익률
def get_stats(labels, price_data, lr_data):
    from scipy.stats import skew, kurtosis

    # 레이블
    preds = labels

    # 클러스터별 가격 뭉치
    cluster_price = get_cluster_time_series(preds, price_data)

    # 클러스터별 로그 수익률 뭉치
    cluster_lr = get_cluster_time_series(preds, lr_data)

    # 연율화 데이터프레임
    cluster_annualised_R = {}
    cluster_mdd = {}
    cluster = []
    cluster_mean_annualised_R = []
    cluster_std_annualised_R = []
    cluster_median_annualised_R = []
    cluster_mean_mdd = []
    cluster_median_mdd = []
    num_coins = []

    for i in cluster_price.keys():
        cluster.append(i)
        cluster_i_price = cluster_price[i]

        # 연율화 수익률 구하기
        beginning_price = cluster_i_price[price_data.columns[0]]
        ending_price = cluster_i_price[price_data.columns[-1]]

        annualised_R = (ending_price - beginning_price) / beginning_price * 100
        
        mean_annualised_R = np.mean(annualised_R)
        std_annualised_R = np.std(annualised_R)
        median_annualised_R = np.median(annualised_R)

        cluster_mean_annualised_R.append(mean_annualised_R)
        cluster_std_annualised_R.append(std_annualised_R)
        cluster_median_annualised_R.append(median_annualised_R)
        cluster_annualised_R[i] = annualised_R

        # 최대 낙폭 구하기
        high = cluster_i_price.cummax(axis=1) # 전고점
        curr = cluster_i_price # 현재가
        drawdown = (1 - curr/ high) * 100 # 낙폭

        max_drawdown = drawdown.max(axis=1) # 최대 낙폭

        mean_max_drawdown = np.mean(max_drawdown)
        median_max_drawdown = np.median(max_drawdown)

        cluster_mean_mdd.append(mean_max_drawdown)
        cluster_median_mdd.append(median_max_drawdown)
        cluster_mdd[i] = max_drawdown
        
        # 코인 개수
        num_coins.append(len(cluster_i_price))

    df_stat = pd.DataFrame()
    df_stat["cluster"] = cluster
    df_stat["return_mean[%]"] = cluster_mean_annualised_R
    df_stat["return_std[%]"] = cluster_std_annualised_R
    df_stat["return_median[%]"] = cluster_median_annualised_R
    df_stat["mdd_mean[%]"] = cluster_mean_mdd
    df_stat["mdd_median[%]"] = cluster_median_mdd
    df_stat["num_coins"] = num_coins

    # 연율화 로그 수익률 데이터프레임
    cluster_annualised_log_return = {}
    cluster_std_annualised_log_return = {}
    
    cluster = []
    cluster_mean_mean_log_return = []
    cluster_std_mean_log_return = []
    cluster_mean_std_log_return = []
    cluster_std_std_log_return = []
    cluster_median_log_return = []
    cluster_skewness = []
    cluster_kurtosis = []
    num_coins = []

    for i in cluster_lr.keys():
        cluster.append(i)
        cluster_i_lr = cluster_lr[i]

        # 연율화 로그 수익률 평균과 표준편차 구하기
        mean_daily_log_return = cluster_i_lr.mean(axis=1)
        std_daily_log_return = cluster_i_lr.std(axis=1)
        
        mean_annualised_daily_log_return = mean_daily_log_return * 365 * 100
        std_annualised_daily_log_return = std_daily_log_return * np.sqrt(365) * 100
        
        mean_mean_log_return = np.mean(mean_annualised_daily_log_return) # 연율화된 일별 로그 수익률 평균의 평균 => 클러스터 연율화된 일별 로그 수익률 평균
        std_mean_log_return = np.std(mean_annualised_daily_log_return)
        mean_std_log_return = np.mean(std_annualised_daily_log_return)
        std_std_log_return = np.std(std_annualised_daily_log_return)
        median_log_return = np.median(mean_annualised_daily_log_return)

        cluster_mean_mean_log_return.append(mean_mean_log_return)
        cluster_std_mean_log_return.append(std_mean_log_return)
        cluster_mean_std_log_return.append(mean_std_log_return)
        cluster_std_std_log_return.append(std_std_log_return)
        cluster_median_log_return.append(median_log_return)
        cluster_annualised_log_return[i] = mean_annualised_daily_log_return
        cluster_std_annualised_log_return[i] = std_annualised_daily_log_return
        
        # 연율화된 로그 수익률 평균의 왜도
        skewness = skew(mean_annualised_daily_log_return, axis=0)
        
        cluster_skewness.append(skewness)

        # 연율화된 로그 수익률 평균의 첨도
        kurtosis_ = kurtosis(mean_annualised_daily_log_return, axis=0)

        cluster_kurtosis.append(kurtosis_)
        
        # 코인 개수
        num_coins.append(len(cluster_i_lr))

    df_stat_lr = pd.DataFrame()
    df_stat_lr["cluster"] = cluster
    df_stat_lr["log_return_mean[%]"] = cluster_mean_mean_log_return
    df_stat_lr["log_return_mean_std[%]"] = cluster_std_mean_log_return
    df_stat_lr["log_return_median[%]"] = cluster_median_log_return
    df_stat_lr["log_return_std[%]"] = cluster_mean_std_log_return
    df_stat_lr["log_return_std_std[%]"] = cluster_std_std_log_return
    df_stat_lr["skewness"] = cluster_skewness
    df_stat_lr["kurtosis"] = cluster_kurtosis
    
    # 로그 수익률 데이터 프레임 추가정보 (가격으로 계산한 MDD)
    df_stat_lr["mdd_mean[%]"] = cluster_mean_mdd
    df_stat_lr["mdd_median[%]"] = cluster_median_mdd
    df_stat_lr["num_coins"] = num_coins

    return df_stat, df_stat_lr, cluster_annualised_R, cluster_annualised_log_return, cluster_std_annualised_log_return, cluster_mdd
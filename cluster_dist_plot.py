import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_dist_plot(df_stat_lr, annl_lr, annl_lr_std, MDD):
        df_statistics = df_stat_lr

        df_dist_plot = pd.DataFrame()

        ann_log_return_list = []
        ann_std_list = []
        maximum_drawdown_list = []
        common_label_list = []
        model_list = []

        for i in annl_lr.keys():
            # 연율화 로그 수익률
            log_return = annl_lr[i].to_list()
            ann_log_return_list.append(log_return)

            # 연율화 로그 수익률의 표준편차
            standard_deviation = annl_lr_std[i].to_list()
            ann_std_list.append(standard_deviation)

            # 최대낙폭
            mdd = MDD[i].to_list()
            maximum_drawdown_list.append(mdd)

            # 공통 군집 레이블
            mask2 = df_statistics["cluster"] == i
            labels = df_statistics[mask2]["common_cluster_label"].to_list()
            labels = labels * len(log_return)
            common_label_list.append(labels)

        # 이중리스트 풀기
        ann_log_return_list = sum(ann_log_return_list, [])
        ann_std_list = sum(ann_std_list, [])
        maximum_drawdown_list = sum(maximum_drawdown_list, [])
        common_label_list = sum(common_label_list, [])
        model_list = sum(model_list, [])

        df_dist_plot["crypto_annualised_log_return[%]"] = ann_log_return_list
        df_dist_plot["crypto_annualised_std[%]"] = ann_std_list
        df_dist_plot["crypto_mdd[%]"] = maximum_drawdown_list
        df_dist_plot["common_cluster_label"] = common_label_list

        df_dist_plot = df_dist_plot.sort_values(by="common_cluster_label")
        
        return df_dist_plot

def plot_annlr(df_dist_plot):
    sns.set_context("notebook")
    # sns.set_theme(style="whitegrid", palette="husl")
    sns.set_theme(style="darkgrid", palette="husl")

    g = sns.displot(data=df_dist_plot, x="crypto_annualised_log_return[%]", hue="common_cluster_label",
                    kind="kde",
                    height=7, aspect=1.5
                   )

    g.set_xlabels("Annualised log return [%]")
    g.set_titles("{col_name}")
    g._legend.set_title(title="Cluster")
    plt.show()
        
def plot_boxplot(df_dist_plot, lim=None):
    fig, axes = plt.subplots(1, 3, figsize=(22,7))
    # fig.suptitle("Log return")

    sns.set_context("notebook")
    sns.set(rc={'figure.figsize':(11.7,7.27)})
    sns.set_theme(style="whitegrid")

    # 로그 수익률 Boxplot
    g_r = sns.boxplot(
        data=df_dist_plot, x="crypto_annualised_log_return[%]", y="common_cluster_label",
        showcaps=True, width=0.5, palette="husl",
        flierprops={"marker": "x"},
        ax=axes[0]
    )
    g_r.set_title("Log return")
    g_r.set(xlabel="Annualised log return [%]",
          ylabel="cluster")

    # 표준편차 Boxplot
    g_s = sns.boxplot(
        data=df_dist_plot, x="crypto_annualised_std[%]", y="common_cluster_label",
        showcaps=True, width=0.5, palette="husl",
        flierprops={"marker": "x"},
        ax=axes[1]
    )
    g_s.set_title("Standard deviation")
    g_s.set(xlabel="Standard deviation [%]",
            ylabel="cluster")

    # 최대낙폭 Boxplot
    g_m = sns.boxplot(
        data=df_dist_plot, x="crypto_mdd[%]", y="common_cluster_label",
        showcaps=True, width=0.5, palette="husl",
        flierprops={"marker": "x"},
        ax=axes[2]
    )
    g_m.set_title("Maximum drawdown")
    g_m.set(xlabel="Maximum Drawdown [%]",
            ylabel="cluster")
    
    if lim:
        g_r.set(xlim=(lim["r"]))
        g_s.set(xlim=(lim["s"]))
        g_m.set(xlim=(lim["m"]))
        
    plt.show()
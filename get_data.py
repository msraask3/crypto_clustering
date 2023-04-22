import pandas as pd

def get_coin_df(coin_data, coin_id):
    """
    returns coin's price and market cap as Pandas dataframe
    
    Args:
        coin_data: CoinGeckoAPI chart data
        coin_id (str): CoinGecko coin id
    """
    coin_date = []
    coin_prices = []
    coin_market_caps = []
    for i in range(len(coin_data["prices"])):
        coin_date.append(coin_data["prices"][i][0])
        coin_prices.append(coin_data["prices"][i][1])
        coin_market_caps.append(coin_data["market_caps"][i][1])

    df_price = pd.DataFrame(columns=["uts", coin_id])
    df_price["uts"] = coin_date
    df_price[coin_id] = coin_prices

    df_market_caps = pd.DataFrame(columns=["uts", coin_id])
    df_market_caps["uts"] = coin_date
    df_market_caps[coin_id] = coin_market_caps

    return df_price, df_market_caps

def get_coin_df_volume(coin_data, coin_id):
    """
    returns coin's volume as Pandas dataframe

    Args:
        coin_data: CoinGeckoAPI chart data
        coin_id (str): CoinGecko coin id
    """
    coin_date = []
    coin_volume = []
    for i in range(len(coin_data["prices"])):
        coin_date.append(coin_data["prices"][i][0])
        coin_volume.append(coin_data["total_volumes"][i][1])

    df_volume = pd.DataFrame(columns=["uts", coin_id])
    df_volume["uts"] = coin_date
    df_volume[coin_id] = coin_volume

    return df_volume

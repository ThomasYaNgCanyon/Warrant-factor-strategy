from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from typing import Literal, Union, Tuple
import re
import os
import logging
import config
from sortedcontainers import SortedList


def load_databycontract(v: str, v_upper: str, quote_dir: str, logger: logging.Logger):
    dir_path = os.path.join(quote_dir, "DataByContract", v)
    if not os.path.exists(dir_path):
        logger.warning(f"Directory not found: {dir_path}")
        return None

    # match file
    files = os.listdir(dir_path)
    pattern = rf"\w+.{v_upper}\.csv"
    logger.debug(f"Pattern: {pattern}")
    files = [f for f in files if re.match(pattern, f)]
    if len(files) != 1:
        error = f"When reading {v}, expected 1 file, got {len(files)}: {files}"
        raise ValueError(error)
    file = files[0]

    # read file
    logger.info(f"Reading {file}")
    df = pd.read_csv(os.path.join(dir_path, file), index_col=0, parse_dates=True)
    if df.empty:
        raise ValueError(f"Empty dataframe: {file}")

    return df


def scale(
    x: Union[pd.DataFrame, pd.Series],
    how: Literal["minmax", "standard", "1stvalue", "divstd"] = "standard",
) -> pd.DataFrame:
    assert how in [
        "minmax",
        "standard",
        "1stvalue",
        "divstd",
    ], "how must be 'minmax', 'standard', '1stvalue', or 'divstd'"

    if isinstance(x, pd.Series):
        x = x.to_frame()

    if how == "minmax":
        scaler = MinMaxScaler()
        return pd.DataFrame(scaler.fit_transform(x), columns=x.columns, index=x.index)
    if how == "standard":
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(x), columns=x.columns, index=x.index)
    if how == "1stvalue":
        return x.apply(lambda x: x / abs(x.loc[x.first_valid_index()]))
    if how == "divstd":
        return x.apply(lambda x: x / x.std())


def ic(price: pd.Series, factor: pd.Series) -> pd.DataFrame:
    # correlation between x and y with lag 1
    price = price.loc[price.first_valid_index() :]
    factor = factor.loc[factor.first_valid_index() :]
    price, factor = price.align(factor, axis=0, join="inner")
    return price.corrwith(factor.shift(1))


def rule_encode(x: pd.Series) -> pd.Series:
    y = x.copy()
    freq = {
        "日": 0,
        "周": 1,
        "月": 2,
        "年": 3,
    }

    interval = {
        "当": 0,
        "次": 1,
        "隔": 2,
    }

    day_day = {
        "": 0,
    }

    week_day = {
        "周一": 0,
        "周二": 1,
        "周三": 2,
        "周四": 3,
        "周五": 4,
        "周六": 5,
        "周日": 6,
    }

    month_day = {
        "上旬": 0,
        "中旬": 1,
        "下旬": 2,
    }

    year_day = {
        "一月": 0,
        "二月": 1,
        "三月": 2,
        "四月": 3,
        "五月": 4,
        "六月": 5,
        "七月": 6,
        "八月": 7,
        "九月": 8,
        "十月": 9,
        "十一月": 10,
        "十二月": 11,
        "上半年": 20,
        "下半年": 21,
        "第一季度": 30,
        "第二季度": 31,
        "第三季度": 32,
        "第四季度": 33,
    }

    for i, s in enumerate(y):
        s = s.rstrip("发布")

        freq_code = freq[s[1]]
        interval_code = interval[s[0]]
        if freq_code == 0:
            day_code = day_day[s[2:]]
        elif freq_code == 1:
            day_code = week_day[s[2:]]
        elif freq_code == 2:
            day_code = month_day[s[2:]]
        else:
            day_code = year_day[s[2:]]

        code = f"{freq_code}_{interval_code}_{day_code}"

        y[i] = code

    return y


def bt_single(
    price: pd.DataFrame,
    factor: pd.Series,
    trade: Literal["long", "short", "longshort"],
) -> pd.DataFrame:
    long = trade in ["long", "longshort"]
    short = trade in ["short", "longshort"]

    price = price.loc[price.first_valid_index() :]
    factor = factor.loc[factor.first_valid_index() :]
    price, factor = price.align(factor, axis=0, join="inner")

    asset = pd.Series(index=price.index, name="asset")
    asset.iloc[0] = 1
    hold = 0

    for d in price.index[1:]:
        # get last price and factor, update asset
        last_index = price.index.get_loc(d) - 1
        last_factor = factor.iloc[last_index]
        last_price = price.iloc[last_index]
        last_asset = asset.iloc[last_index]

        asset.loc[d] = last_asset + hold * (price.loc[d] - last_price)

        # # update cash and hold
        # if factor.loc[d] > last_factor and hold <= 0:
        #     if long:
        #         hold = asset.loc[d] / price.loc[d]
        #     else:
        #         hold = 0
        # elif factor.loc[d] < last_factor and hold >= 0:
        #     if short:
        #         hold = -asset.loc[d] / price.loc[d]
        #     else:
        #         hold = 0

        # update cash and hold
        if factor.loc[d] > 0 and hold <= 0:
            if long:
                hold = asset.loc[d] / price.loc[d]
            else:
                hold = 0
        elif factor.loc[d] < 0 and hold >= 0:
            if short:
                hold = -asset.loc[d] / price.loc[d]
            else:
                hold = 0

    return asset


def yearly_return(
    asset: pd.Series,
    start: Union[pd.Timestamp, None] = None,
    end: Union[pd.Timestamp, None] = None,
) -> pd.Series:
    if start:
        asset = asset.loc[start:]
    if end:
        asset = asset.loc[:end]

    first = asset.loc[asset.first_valid_index()]
    last = asset.loc[asset.last_valid_index()]

    total_return = last / first - 1
    years = (asset.last_valid_index() - asset.first_valid_index()).days / 365
    return total_return / years


def yearly_vol(
    asset: pd.Series,
    start: Union[pd.Timestamp, None] = None,
    end: Union[pd.Timestamp, None] = None,
) -> pd.Series:
    if start:
        asset = asset.loc[start:]
    if end:
        asset = asset.loc[:end]

    first_index = asset.first_valid_index()
    last_index = asset.last_valid_index()
    asset = asset.loc[first_index:last_index]

    return asset.pct_change().std() * (252**0.5)


def sharpe(asset: pd.Series) -> pd.Series:
    return yearly_return(asset) / yearly_vol(asset)


def max_drawdown(asset: pd.Series) -> pd.Series:
    max_asset = asset.cummax()
    return ((max_asset - asset) / max_asset).max()


def factor_test(
    price: pd.Series,
    factor: pd.Series,
    trade: Literal["long", "short", "longshort"],
    logger: Union[logging.Logger, None] = None,
) -> Tuple[pd.Series, pd.Series]:
    asset = bt_single(price, factor, trade)
    yr = yearly_return(asset)
    yv = yearly_vol(asset)
    sp = sharpe(asset)
    md = max_drawdown(asset)
    close_yr = yearly_return(price, asset.index[0], asset.index[-1])
    exceed_yr = yr - close_yr

    if logger:
        log_text = f"({price.name}, {factor.name}, {trade}, {asset.index[-1].strftime('%Y%m%d')}): yr={yr:.2%},\teyr={exceed_yr:.2%},\tyv={yv:.2%},\tsp={sp:.2f},\tmd={md:.2%}"
        logger.info(log_text)

    row = {
        "yearly_return": yr,
        "exceed_yr": exceed_yr,
        "yearly_vol": yv,
        "sharpe": sp,
        "max_drawdown": md,
    }

    return pd.Series(row), asset


def posi_bt(
    posi: pd.DataFrame,
    price: pd.DataFrame,
) -> pd.Series:
    posi = posi.shift(1)
    price_ret = price.pct_change()

    posi, price_ret = posi.align(price_ret, axis=0, join="inner")

    port_ret = (posi * price_ret).sum(axis=1)
    asset = (1 + port_ret).cumprod()

    return asset


def few2crossing(
    posi: Union[pd.DataFrame, pd.Series],
) -> pd.DataFrame:
    full_v = config.config["full_bt"]
    if isinstance(posi, pd.Series):
        posi = posi.to_frame()

    posi.columns = posi.columns.str.upper()

    posi = posi.reindex(columns=full_v, fill_value=0)
    posi.index = posi.index.strftime("%Y%m%d")
    posi.index.name = "date"

    return posi


def yoy(
    x: pd.Series,
    freq: Literal["m", "d", "w"],
) -> pd.Series:
    if freq == "m":
        x_m = x.resample("M").last()
        x_yoy = x_m.pct_change(12)
    if freq == "d":
        x_d = x.resample("D").last()
        x_yoy = x_d.pct_change(365)
    if freq == "w":
        x_w = x.resample("W").last()
        x_yoy = x_w.pct_change(52)

    x_yoy = x_yoy
    x_yoy = x_yoy.reindex(x.index, method="ffill")

    return x_yoy


def sma(
    x: pd.Series,
    window: int,
) -> pd.Series:
    return x.rolling(window).mean()


def ema(
    x: pd.Series,
    alpha: float,
    adjust: bool = False,
) -> pd.Series:
    return x.ewm(alpha=alpha, adjust=adjust).mean()


def ts_rank(
    s: pd.Series,
    window: int,
    method: Literal["pandas", "numpy", "dynamic"] = "dynamic",
) -> pd.Series:
    s = s.dropna()
    if method == "pandas":
        rankings = s.rolling(window).apply(lambda x: x.rank().iloc[-1])
        return (rankings - 1) / (window - 1)
    if method == "numpy":
        rankings = s.rolling(window).apply(lambda x: np.searchsorted(np.sort(x), x.iloc[-1]) + 1)
        return (rankings - 1) / (window - 1)
        # rank = lambda x: (np.searchsorted(np.sort(x), x.iloc[-1]) + 1) / len(x)
        # return s.rolling(window).apply(rank)
    if method == "dynamic":
        n = len(s)
        rankings = np.full(n, np.nan)
        sorted_list = SortedList()

        # Initialize the sorted list with the first window elements
        for i in range(window):
            sorted_list.add(s.iloc[i])

        # Sliding window
        for i in range(window, n):
            # Remove the element that is sliding out of the window
            sorted_list.remove(s.iloc[i - window])
            # Add the new element that is entering the window
            sorted_list.add(s.iloc[i])

            # Compute the rank of the new element
            rankings[i] = sorted_list.index(s.iloc[i])

        return pd.Series(rankings / (window - 1), index=s.index).dropna()

# -*- encoding: utf-8 -*-
"""
@File    :   Factor_Example.py
@Time    :   2022/04/15 11:23:40
@Author  :   BWQ 
@Version :   1.0.22020509
@Contact :   baowq@spicam.net
@Desc    :   Backtest System for Cross-Sectional
"""

import numpy as np
import os
import pandas as pd

import sys

# 回测框架所在路径
sys.path.append("C:/Users/ifwha/OneDrive - CUHK-Shenzhen/桌面/工作/A04_回测框架/截面项目/dataprepare/code")

from makeFactorPosition import *
from makeNetWorthFuture import MakeNetWorth
from pandas import Series, DataFrame

# 回测所依赖的数据路径
SDATAPATH = "C:/Users/ifwha/OneDrive - CUHK-Shenzhen/桌面/工作/A04_回测框架/截面数据"


if __name__ == "__main__":
    bTrading = None
    sFactorName = sys.argv[1]
    sFactorPath = sys.argv[2]
    # 保存回测结果的路径。请使用绝对路径！
    sRstPath = "C:/Users/ifwha/OneDrive - CUHK-Shenzhen/桌面/工作/A06_策略研究/有色黑色对冲/data/result"

    # 因子制作
    # dfFactor = -pd.read_pickle(sFactorPath).iloc[:-1]
    dfFactor = pd.read_csv(sFactorPath, index_col='date')
    # dfFactor = pd.read_csv("C:/Users/ifwha/OneDrive - CUHK-Shenzhen/桌面/工作/A06_策略研究/黑色板块策略/data/position/position.csv", index_col='date')
    # sFactorName = "黑色板块策略"

    # 如果 `nLongRatio` 和 `nShortRatio` 为 `DataFrame`，那么 `weight` 必须为 `None`
    weight = False  #  等权
    # weight = dfFactor
    nLongRatio = 1 / 3
    # nLongRatio = dfFactor[dfFactor > 0]
    nShortRatio = 1 / 3
    # nShortRatio = dfFactor[dfFactor < 0].abs()
    nStartDate = 20150101
    # nMinOffset = 1
    nFee = 0.0002
    # sTradingTime = 'close'
    sTradingTime = "0950"
    nFactorLag = 0

    sRstPath = os.path.join(sRstPath, sFactorName)
    if not os.path.exists(sRstPath):
        os.makedirs(sRstPath)

    FactorPositionOutput(
        sRstPath,
        sFactorName,
        dfFactor,
        SDATAPATH,
        nStartDate,
        weight,
        nLongRatio,
        nShortRatio,
        nFactorLag=nFactorLag,
        bTrading=bTrading,
    )
    sRstFile = os.path.join(sRstPath, "%s.csv" % sFactorName)
    print("读取持仓文件生成净值: %s" % sRstFile)
    MakeNetWorth(
        sRstFile,
        bTrading=bTrading,
        nFee=nFee,
        sTradingTime=sTradingTime,
        bStatNetByCode=False,
    ).run()

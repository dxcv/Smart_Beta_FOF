import numpy as np
import pandas as pd
from utils import sub_d

trading_days = pd.read_excel('./data/trading_days.xlsx')
trading_days = list(trading_days.iloc[:, 1].values)
# print(trading_days)

trading_days_w = pd.read_excel('./data/trading_days_w.xlsx')
trading_days_w = list(trading_days_w.iloc[:, 1].values)
# print(trading_days_w)

trading_days_m = pd.read_excel('./data/trading_days_m.xlsx')
trading_days_m = list(trading_days_m.iloc[:, 1].values)
# print(trading_days_m)

smart_beta_name = ['300波动', '300相对成长', '300相对价值', '300红利', '300低贝塔', '300等权']
formation_window_t = ['1-' + str(i+1) for i in range(36)]
# print(formation_window_t)
formation_window = ['1-1', '1-3', '1-6', '1-9', '1-12', '1-24', '1-36']

# 收盘价
smart_beta_price = pd.DataFrame()
for index_name in smart_beta_name:
    temp_price = pd.read_excel('./data/' + index_name + '.xlsx')
    temp_price = temp_price.iloc[3:, :]
    temp_price.columns = ['Date', 'close']
    temp_price.loc[:, 'Date'] = temp_price.loc[:, 'Date'].apply(sub_d)
    temp_price.set_index(['Date'], drop=True, inplace=True)
    smart_beta_price = pd.concat([smart_beta_price, temp_price], axis=1, sort=True)
smart_beta_price.columns = smart_beta_name
# print(smart_beta_price)

# 收益率（日度）
smart_beta_return_np = np.log(np.array(smart_beta_price.iloc[1:, :], dtype=float)) - np.log(np.array(smart_beta_price.iloc[:-1, :], dtype=float))
smart_beta_return = pd.DataFrame(smart_beta_return_np)
smart_beta_return.columns = smart_beta_price.columns
smart_beta_return.index = list(smart_beta_price.index)[1:]
# print(smart_beta_return)

# 收益率（月度）
smart_beta_price_m = smart_beta_price.loc[trading_days_m, :]
smart_beta_return_m_np = np.log(np.array(smart_beta_price_m.iloc[1:, :], dtype=float)) - np.log(np.array(smart_beta_price_m.iloc[:-1, :], dtype=float))
smart_beta_return_m = pd.DataFrame(smart_beta_return_m_np)
smart_beta_return_m.columns = smart_beta_price_m.columns
smart_beta_return_m.index = list(smart_beta_price_m.index)[1:]
smart_beta_return_m.reset_index(drop=False, inplace=True)
smart_beta_return_m.loc[:, 'index'] = smart_beta_return_m.loc[:, 'index'].apply(lambda x: x[:7])
smart_beta_return_m.set_index(['index'], drop=True, inplace=True)
# print(smart_beta_return_m)

# # 动量效应检验
# for time_window in formation_window_t:
#     lag_return_volatility = pd.DataFrame()
#     month_return_volatility = pd.DataFrame()
#     temp_lag = np.abs(int(time_window[-2:]))
#     if temp_lag < 12:
#         month_lag = 36
#     else:
#         month_lag = 96
#     for month_index in range(month_lag, len(list(smart_beta_return_m.index))):
#         temp_return = smart_beta_return_m.iloc[month_index - temp_lag:month_index, :].sum().values
#         temp_volatility = smart_beta_return_m.iloc[month_index - month_lag:month_index, :].std().values * np.sqrt(12)
#         temp_divide = temp_return / temp_volatility
#         lag_return_volatility = pd.concat([lag_return_volatility, pd.DataFrame([temp_divide], columns=smart_beta_name)], axis=0)
#
#         temp_return_m = smart_beta_return_m.iloc[month_index, :].values
#         temp_volatility_m = smart_beta_return_m.iloc[month_index - month_lag + 1:month_index+1, :].std().values * np.sqrt(12)
#         temp_divide_m = temp_return_m / temp_volatility_m
#         month_return_volatility = pd.concat([month_return_volatility, pd.DataFrame([temp_divide_m], columns=smart_beta_name)], axis=0)
#     lag_return_volatility.index = list(smart_beta_return_m.index)[month_lag:]
#     month_return_volatility.index = list(smart_beta_return_m.index)[month_lag:]
#     lag_return_volatility_T = lag_return_volatility.T
#     month_return_volatility_T = month_return_volatility.T
#     # print(lag_return_volatility_T)
#     # print(month_return_volatility_T)
#
#     lag_return_volatility_T_d = pd.DataFrame()
#     for index_name in list(lag_return_volatility_T.index):
#         index_name_list_x = [index_name] * lag_return_volatility_T.shape[1]
#         columns_x = lag_return_volatility_T.columns
#         temp_x = list(lag_return_volatility_T.loc[index_name, :].values)
#         temp_df_x = pd.concat([pd.DataFrame(index_name_list_x), pd.DataFrame(columns_x), pd.DataFrame(temp_x)], axis=1, sort=True)
#         lag_return_volatility_T_d = pd.concat([lag_return_volatility_T_d, temp_df_x], axis=0)
#     lag_return_volatility_T_d.columns = ['index', 'month', 'x']
#     # print(lag_return_volatility_T_d)
#
#     month_return_volatility_T_d = pd.DataFrame()
#     for index_name in list(month_return_volatility_T.index):
#         index_name_list_y = [index_name] * month_return_volatility_T.shape[1]
#         columns_y = month_return_volatility_T.columns
#         temp_y = list(month_return_volatility_T.loc[index_name, :].values)
#         temp_df_y = pd.concat([pd.DataFrame(index_name_list_y), pd.DataFrame(columns_y), pd.DataFrame(temp_y)], axis=1, sort=True)
#         month_return_volatility_T_d = pd.concat([month_return_volatility_T_d, temp_df_y], axis=0)
#     month_return_volatility_T_d.columns = ['index', 'month', 'y']
#     # print(month_return_volatility_T_d)
#     lag_return_volatility_T_d['y'] = month_return_volatility_T_d.pop('y')
#     print(lag_return_volatility_T_d)
#     lag_return_volatility_T_d.to_excel('./data/面板数据/' + time_window + '.xlsx', index=False)


init_date = '2014-12-31'
init_month = init_date[:7]
backtest_date = list(smart_beta_return.index)[list(smart_beta_return.index).index(init_date) + 1:]
# print(backtest_date)

smart_beta_momentum_return = pd.DataFrame()
for time_window in formation_window_t:
    weights_CSFM = pd.DataFrame()
    weights_TSFM = pd.DataFrame()
    temp_lag = np.abs(int(time_window[-2:]))
    if temp_lag < 12:
        month_lag = 36
    else:
        month_lag = 96
    for month_index in range(list(smart_beta_return_m.index).index(init_month), len(list(smart_beta_return_m.index))):
        temp_return = smart_beta_return_m.iloc[month_index - temp_lag + 1:month_index + 1, :].sum().values
        temp_volatility = smart_beta_return_m.iloc[month_index - month_lag + 1:month_index + 1, :].std().values * np.sqrt(12)
        temp_divide = temp_return / temp_volatility
        temp_scale = [np.min(np.array([np.max(np.array([d, -2])), 2])) for d in list(temp_divide)]
        temp_scale_array = np.array(temp_scale)

        # 横截面动量权重
        temp_scale_index_CSFM = np.argwhere(temp_return > np.median(temp_return)).flatten()
        temp_weights_CSFM = np.zeros(len(smart_beta_return_m.columns))
        temp_scale_array_CSFM = temp_divide[temp_scale_index_CSFM]
        temp_scale_weights_CSFM_init = (temp_scale_array_CSFM - np.min(temp_scale_array_CSFM)) / (np.max(temp_scale_array_CSFM) - np.min(temp_scale_array_CSFM)) + 1
        temp_scale_weights_CSFM = temp_scale_weights_CSFM_init/np.sum(temp_scale_weights_CSFM_init)
        temp_weights_CSFM[temp_scale_index_CSFM] = temp_scale_weights_CSFM
        # print(temp_weights_CSFM)

        # 时序动量权重
        temp_scale_index_TSFM = np.argwhere(temp_scale_array > 0).flatten()
        temp_weights_TSFM = np.zeros(len(smart_beta_return_m.columns))
        if len(temp_scale_index_TSFM) != 0:
            temp_scale_array_TSFM = temp_scale_array[temp_scale_index_TSFM]
            temp_scale_weights_TSFM = temp_scale_array_TSFM/np.sum(temp_scale_array_TSFM)
            temp_weights_TSFM[temp_scale_index_TSFM] = temp_scale_weights_TSFM
        # print(temp_weights_TSFM)
        weights_CSFM = pd.concat([weights_CSFM, pd.DataFrame([temp_weights_CSFM], columns=smart_beta_return_m.columns)], axis=0)
        weights_TSFM = pd.concat([weights_TSFM, pd.DataFrame([temp_weights_TSFM], columns=smart_beta_return_m.columns)], axis=0)
    weights_CSFM.index = trading_days_m[trading_days_m.index(init_date):]
    weights_TSFM.index = trading_days_m[trading_days_m.index(init_date):]
    # print(weights_TSFM)
    # 换手率
    # TSFM
    if time_window == '1-5':
        turnover_TSFM = np.array([])
        for date_index in range(len(list(weights_TSFM.index))):
            if date_index == 0:
                continue
                # temp_turnover = np.sum(smart_beta_weights.iloc[date_index, :].values)
            else:
                temp_weights_sub = weights_TSFM.iloc[date_index, :].values - weights_TSFM.iloc[date_index - 1, :].values
                temp_turnover = np.sum(np.abs(temp_weights_sub))
            turnover_TSFM = np.append(turnover_TSFM, temp_turnover)
        print(turnover_TSFM)
        print('TSSM_5:', np.mean(turnover_TSFM))
    # CSFM
    if time_window == '1-9':
        turnover_CSFM = np.array([])
        for date_index in range(len(list(weights_CSFM.index))):
            if date_index == 0:
                continue
                # temp_turnover = np.sum(smart_beta_weights.iloc[date_index, :].values)
            else:
                temp_weights_sub = weights_CSFM.iloc[date_index, :].values - weights_CSFM.iloc[date_index - 1, :].values
                temp_turnover = np.sum(np.abs(temp_weights_sub))
            turnover_CSFM = np.append(turnover_CSFM, temp_turnover)
        print(turnover_CSFM)
        print('CSSM_9:', np.mean(turnover_CSFM))

    smart_beta_momentum_return_CSFM = []
    smart_beta_momentum_return_TSFM = []

    for date in backtest_date:
        temp_update_date = np.array(weights_CSFM.index)[np.argwhere(np.array(weights_CSFM.index) < date)[-1][0]]
        temp_return_CSFM = np.dot(smart_beta_return.loc[date, :].values, weights_CSFM.loc[temp_update_date, :].values) * (1-2/1000)
        temp_return_TSFM = np.dot(smart_beta_return.loc[date, :].values, weights_TSFM.loc[temp_update_date, :].values) * (1-2/1000)
        smart_beta_momentum_return_CSFM.append(temp_return_CSFM)
        smart_beta_momentum_return_TSFM.append(temp_return_TSFM)
    smart_beta_momentum_return_CSFM = pd.DataFrame(smart_beta_momentum_return_CSFM, index=backtest_date, columns=['CSSM_' + str(temp_lag)])
    smart_beta_momentum_return_TSFM = pd.DataFrame(smart_beta_momentum_return_TSFM, index=backtest_date, columns=['TSSM_' + str(temp_lag)])
    temp_smart_beta_momentum_return = pd.concat([smart_beta_momentum_return_CSFM, smart_beta_momentum_return_TSFM], axis=1, sort=True)
    smart_beta_momentum_return = pd.concat([smart_beta_momentum_return, temp_smart_beta_momentum_return], axis=1, sort=True)

equal_weights_return = smart_beta_return.loc[backtest_date[0]:, :].mean(axis=1).values
smart_beta_momentum_return = pd.concat([smart_beta_momentum_return, pd.DataFrame(equal_weights_return, index=backtest_date, columns=['等权重'])], axis=1, sort=True)
# smart_beta_momentum_return.to_excel('./data/动量择时_收益.xlsx')
print(smart_beta_momentum_return)
print(smart_beta_momentum_return.sum())

# 计算相关指标（回测净值、总收益率、年化收益率、年化波动率、最大回撤率、日胜率）
# 回测净值
strategy_net_value = pd.DataFrame([np.ones(smart_beta_momentum_return.shape[1])], columns=smart_beta_momentum_return.columns)
for date in list(smart_beta_momentum_return.index):
    temp_return = smart_beta_momentum_return.loc[date, :].values
    temp_net_value = strategy_net_value.iloc[-1, :].values
    strategy_net_value = pd.concat([strategy_net_value, pd.DataFrame([np.multiply(temp_net_value, np.exp(temp_return))], columns=smart_beta_momentum_return.columns)], axis=0)
strategy_net_value.index = list(smart_beta_return.index)[list(smart_beta_return.index).index(init_date):]
# strategy_net_value.to_excel('./data/动量择时_回测净值.xlsx')
# print(strategy_net_value)

# 总收益率
smart_beta_momentum_return_return_to = smart_beta_momentum_return.sum().values
print(smart_beta_momentum_return_return_to)

# 年化收益率
smart_beta_momentum_return_sum = smart_beta_momentum_return.sum().values
trading_num = smart_beta_momentum_return.shape[0]
smart_beta_momentum_return_an = 252 / trading_num * smart_beta_momentum_return_sum
print(smart_beta_momentum_return_an)

# 年化波动率
smart_beta_momentum_volatility_an = np.sqrt(252) * smart_beta_momentum_return.std().values
print(smart_beta_momentum_volatility_an)

# 最大回撤率
smart_beta_momentum_drawdown = np.array([])
for strategy in strategy_net_value.columns:
    temp_strategy_net_value = list(strategy_net_value.loc[:, strategy].values)
    # print(temp_strategy_net_value)
    acc_return_dd = [-(i - np.max(temp_strategy_net_value[:temp_strategy_net_value.index(i) + 1])) / np.max(temp_strategy_net_value[:temp_strategy_net_value.index(i) + 1]) for i in temp_strategy_net_value[1:]]
    smart_beta_momentum_drawdown = np.append(smart_beta_momentum_drawdown, np.max(np.array(acc_return_dd)))
print(smart_beta_momentum_drawdown)

# 日胜率
smart_beta_momentum_rw = np.array([])
for strategy in strategy_net_value.columns:
    smart_beta_momentum_return_sub = list(smart_beta_momentum_return.loc[:, strategy].values - smart_beta_momentum_return.iloc[:, -1].values)
    smart_beta_momentum_return_sub_gr_0 = np.array([1 if i > 0 else 0 for i in smart_beta_momentum_return_sub])
    valuation_equal_rw = np.sum(smart_beta_momentum_return_sub_gr_0)/len(smart_beta_momentum_return_sub_gr_0)
    smart_beta_momentum_rw = np.append(smart_beta_momentum_rw, valuation_equal_rw)
print(smart_beta_momentum_rw)


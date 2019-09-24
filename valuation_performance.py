import numpy as np
import pandas as pd
from utils import sub_d
from scipy.stats import percentileofscore

trading_days = pd.read_excel('./data/trading_days.xlsx')
trading_days = list(trading_days.iloc[:, 1].values)
# print(trading_days)

trading_days_w = pd.read_excel('./data/trading_days_w.xlsx')
trading_days_w = list(trading_days_w.iloc[:, 1].values)
# print(trading_days_w)

trading_days_m = pd.read_excel('./data/trading_days_m.xlsx')
trading_days_m = list(trading_days_m.iloc[:, 1].values)
trading_days_m = [date for date in trading_days_m if date > '2012-12-21']
# print(trading_days_m)

trading_days_y = pd.read_excel('./data/trading_days_y.xlsx')
trading_days_y = list(trading_days_y.iloc[:, 1].values)
# print(trading_days_y)

# Smart Beta策略估值
smart_beta_name = ['300波动', '300相对成长', '300相对价值', '300红利', '300低贝塔', '300等权']
smart_beta_valuation = pd.DataFrame()
for index_name in smart_beta_name:
    temp_valuation = pd.read_excel('./data/' + index_name + '_估值.xlsx', index_col=0)
    temp_valuation = temp_valuation.loc[:, ['市净率', '市销率']]
    smart_beta_valuation = pd.concat([smart_beta_valuation, pd.DataFrame(np.exp(np.log(temp_valuation.prod(axis=1))/temp_valuation.notna().sum(1)).values, index=list(temp_valuation.index))], axis=1, sort=True)
smart_beta_valuation.columns = smart_beta_name
# print(smart_beta_valuation)

# 计算历史分位数（过去多少个月）
smart_beta_valuation_percentile = pd.DataFrame()
smart_beta_valuation_z_score = pd.DataFrame()
lag = 24
for date_index in range(lag, smart_beta_valuation.shape[0] + 1):
    # print(smart_beta_valuation.iloc[date_index-lag:date_index, :])
    temp_valuation = smart_beta_valuation.iloc[date_index-lag:date_index, :]
    temp_valuation_percentile = [percentileofscore(temp_valuation.iloc[:, index_n].values, temp_valuation.iloc[-1, index_n]) / 100 for index_n in range(len(temp_valuation.columns))]
    temp_valuation_z_score = [(temp_valuation.iloc[-1, index_n] - temp_valuation.iloc[:, index_n].mean()) / temp_valuation.iloc[:, index_n].std() for index_n in range(len(temp_valuation.columns))]
    smart_beta_valuation_percentile = pd.concat([smart_beta_valuation_percentile, pd.DataFrame([temp_valuation_percentile], columns=temp_valuation.columns)], axis=0)
    smart_beta_valuation_z_score = pd.concat([smart_beta_valuation_z_score, pd.DataFrame([temp_valuation_z_score], columns=temp_valuation.columns)], axis=0)
smart_beta_valuation_percentile.index = list(smart_beta_valuation.index)[lag-1:]
smart_beta_valuation_z_score.index = list(smart_beta_valuation.index)[lag-1:]
# print(smart_beta_valuation_percentile)
# print(smart_beta_valuation_z_score)

# Smart Beta策略表现
# 收盘价
smart_beta_price = pd.DataFrame()
trading_days_m_lag = trading_days_m[trading_days_m.index(list(smart_beta_valuation_percentile.index)[0]):]
for index_name in smart_beta_name:
    temp_price = pd.read_excel('./data/' + index_name + '.xlsx')
    temp_price = temp_price.iloc[3:, :]
    temp_price.columns = ['Date', 'close']
    temp_price.loc[:, 'Date'] = temp_price.loc[:, 'Date'].apply(sub_d)
    temp_price.set_index(['Date'], drop=True, inplace=True)
    temp_price = temp_price.loc[trading_days_m_lag, :]
    smart_beta_price = pd.concat([smart_beta_price, temp_price], axis=1, sort=True)
smart_beta_price.columns = smart_beta_name
# print(smart_beta_price)

# 对数收益率
smart_beta_return_np = np.log(np.array(smart_beta_price.iloc[1:, :], dtype=float)) - np.log(np.array(smart_beta_price.iloc[:-1, :], dtype=float))
smart_beta_return = pd.DataFrame(smart_beta_return_np)
smart_beta_return.columns = smart_beta_price.columns
smart_beta_return.index = list(smart_beta_price.index)[1:]
# print(smart_beta_return)

# 净值
smart_beta_net_value = pd.DataFrame([np.ones(len(smart_beta_name))], columns=smart_beta_name)
for date in list(smart_beta_return.index):
    temp_return = smart_beta_return.loc[date, :].values
    temp_net_value = smart_beta_net_value.iloc[-1, :].values
    # print(np.multiply(temp_net_value, np.exp(temp_return)))
    smart_beta_net_value = pd.concat([smart_beta_net_value, pd.DataFrame([np.multiply(temp_net_value, np.exp(temp_return))], columns=smart_beta_name)], axis=0)
smart_beta_net_value.index = list(smart_beta_price.index)
# print(smart_beta_net_value)

# 估值及表现保存为文件
# smart_beta_valuation_percentile.to_excel('./data/valuation.xlsx')
# smart_beta_valuation_z_score.to_excel('./data/valuation_z_score.xlsx')
# smart_beta_net_value.to_excel('./data/performance.xlsx')

# Smart Beta策略估值与未来收益相关性
forward_days = [1, 3, 6, 12, 24]
forward_label = ['1个月', '3个月', '6个月', '12个月', '24个月']

# 未来收益率
for lag in forward_days:
    forward_return = pd.DataFrame()
    for index_num in range(len(list(smart_beta_price.index)[:-lag])):
        temp_return = smart_beta_return.iloc[index_num:index_num + lag, :].sum().values
        forward_return = pd.concat([forward_return, pd.DataFrame([temp_return], columns=smart_beta_name)], axis=0)
    forward_return.index = list(smart_beta_price.index)[:-lag]
    # forward_return.to_excel('./data/未来' + forward_label[forward_days.index(lag)] + '收益率.xlsx')

# 计算相关系数
# print(smart_beta_valuation)
corr_df = pd.DataFrame()
for forward_name in forward_label:
    temp_return = pd.read_excel('./data/未来' + forward_name + '收益率.xlsx', index_col=0)
    temp_smart_beta_valuation = smart_beta_valuation_z_score.loc[list(temp_return.index), :]

    corr_list = []
    for index_name in smart_beta_name:
        corr_list.append(np.corrcoef(temp_return.loc[:, index_name], temp_smart_beta_valuation.loc[:, index_name])[0][-1])
    corr_df = pd.concat([corr_df, pd.DataFrame([np.array(corr_list)], columns=smart_beta_name)], axis=0)
corr_df.index = forward_label
print(corr_df)

# 从一年的时间维度上看，发现5个Smart Beta策略与未来收益均为负相关关系，说明在一年的时间维度上，Smart Beta估值是对未来收益有一定的预测能力
# 每年末判断每个Smart Beta策略估值指标在过去一年的分位数
init_re = '2014-12-31'
re_date = trading_days_y[trading_days_y.index(init_re):]
smart_beta_valuation_percentile.loc[:, '300波动'] = 1 - smart_beta_valuation_percentile.loc[:, '300波动']
smart_beta_valuation_percentile.loc[:, '300相对价值'] = 1 - smart_beta_valuation_percentile.loc[:, '300相对价值']
smart_beta_valuation_percentile.loc[:, '300低贝塔'] = 1 - smart_beta_valuation_percentile.loc[:, '300低贝塔']
smart_beta_valuation_percentile.loc[:, '300等权'] = 1 - smart_beta_valuation_percentile.loc[:, '300等权']

# 横截面
# print(smart_beta_valuation_percentile)
for_return_12 = pd.read_excel('./data/未来' + '12个月' + '收益率.xlsx', index_col=0)
smart_beta_valuation_percentile_12 = smart_beta_valuation_percentile.loc[list(for_return_12.index), :]

low_return = []
middle_return = []
high_return = []
init_index = np.arange(smart_beta_valuation_percentile_12.shape[1])
for temp_index_1 in range(smart_beta_valuation_percentile_12.shape[0]):
    temp_valuation_percentile = smart_beta_valuation_percentile_12.iloc[temp_index_1, :].values
    # print(temp_valuation_percentile)
    # print(np.percentile(temp_valuation_percentile, 100 / 3))
    temp_low_index = np.argwhere(temp_valuation_percentile < np.percentile(temp_valuation_percentile, 100 / 3)).flatten()
    temp_high_index = np.argwhere(temp_valuation_percentile >= np.percentile(temp_valuation_percentile, 200 / 3)).flatten()
    temp_middle_index = np.delete(init_index, np.append(temp_low_index, temp_high_index))

    if len(temp_low_index) != 2 or len(temp_middle_index) != 2 or len(temp_high_index) != 2:
        if len(temp_low_index) == 2:
            if len(temp_high_index) == 3:
                if len(np.argwhere(temp_valuation_percentile[temp_high_index] == np.max(temp_valuation_percentile[temp_high_index])).flatten()) > 1:
                    temp_index = np.argwhere(temp_valuation_percentile[temp_high_index] == np.max(temp_valuation_percentile[temp_high_index])).flatten()[-1]
                else:
                    temp_index = np.argwhere(temp_valuation_percentile[temp_high_index] == np.min(temp_valuation_percentile[temp_high_index])).flatten()[-1]
                temp_middle_index = np.append(temp_middle_index, temp_index)
                temp_high_index = np.delete(temp_high_index, temp_index)
            elif len(temp_middle_index) == 3:
                if len(np.argwhere(temp_valuation_percentile[temp_middle_index] == np.max(temp_valuation_percentile[temp_middle_index])).flatten()) > 1:
                    temp_index = np.argwhere(temp_valuation_percentile[temp_middle_index] == np.max(temp_valuation_percentile[temp_middle_index])).flatten()[-1]
                else:
                    temp_index = np.argwhere(temp_valuation_percentile[temp_middle_index] == np.min(temp_valuation_percentile[temp_middle_index])).flatten()[-1]
                temp_high_index = np.append(temp_high_index, temp_index)
                temp_middle_index = np.delete(temp_middle_index, temp_index)
            elif len(temp_middle_index) == 4:
                temp_index = np.argwhere(temp_valuation_percentile[temp_middle_index] == np.max(temp_valuation_percentile[temp_middle_index])).flatten()[-2:]
                temp_high_index = np.append(temp_high_index, temp_index)
                temp_middle_index = np.delete(temp_middle_index, temp_index)
        elif len(temp_low_index) == 1:
            if len(temp_middle_index) == 3:
                if len(np.argwhere(temp_valuation_percentile[temp_middle_index] == np.max(temp_valuation_percentile[temp_middle_index])).flatten()) > 1:
                    temp_index = np.argwhere(temp_valuation_percentile[temp_middle_index] == np.max(temp_valuation_percentile[temp_middle_index])).flatten()[-1]
                else:
                    temp_index = np.argwhere(temp_valuation_percentile[temp_middle_index] == np.min(temp_valuation_percentile[temp_middle_index])).flatten()[-1]
                temp_low_index = np.append(temp_low_index, temp_index)
                temp_middle_index = np.delete(temp_middle_index, temp_index)
    temp_low_return = for_return_12.iloc[temp_index_1, temp_low_index].mean()

    low_return.append(temp_low_return)
    temp_middle_return = for_return_12.iloc[temp_index_1, temp_middle_index].mean()
    middle_return.append(temp_middle_return)
    temp_high_return = for_return_12.iloc[temp_index_1, temp_high_index].mean()
    high_return.append(temp_high_return)
# print(low_return)
# print(middle_return)
# print(high_return)
cross_return_12 = pd.concat([pd.DataFrame(low_return), pd.DataFrame(middle_return), pd.DataFrame(high_return)], axis=1, sort=True)


cross_return_12.columns = ['估值“较低”', '估值“中等”', '估值“较高”']
cross_return_12.index = list(smart_beta_valuation_percentile_12.index)
# cross_return_12.to_excel('./data/cross_return_12.xlsx')
# print(cross_return_12)

# 权重计算
# 做多估值较低的Smart Beta策略
threshold = 0.35
smart_beta_weights = pd.DataFrame()
for date in re_date:
    temp_zeros = np.zeros(len(smart_beta_valuation_percentile.columns))
    temp_valuation_percentile = smart_beta_valuation_percentile.loc[date, :].values
    temp_l_threshold_index = np.argwhere(temp_valuation_percentile <= threshold).flatten()
    temp_zeros[temp_l_threshold_index] = temp_valuation_percentile[temp_l_threshold_index]
    # 相对历史分位数越小 权重越大
    temp_zeros_1 = [1-i if i > 0 else i for i in temp_zeros]
    temp_zeros_1 = np.array(temp_zeros_1)
    temp_zeros_1 = temp_zeros_1/np.sum(temp_zeros_1)
    smart_beta_weights = pd.concat([smart_beta_weights, pd.DataFrame([temp_zeros_1], columns=smart_beta_valuation_percentile.columns)], axis=0)
smart_beta_weights.index = re_date
print(smart_beta_weights)

# 回测
# 日度收益率
smart_beta_price_d = pd.DataFrame()
for index_name in smart_beta_name:
    temp_price = pd.read_excel('./data/' + index_name + '.xlsx')
    temp_price = temp_price.iloc[3:, :]
    temp_price.columns = ['Date', 'close']
    temp_price.loc[:, 'Date'] = temp_price.loc[:, 'Date'].apply(sub_d)
    temp_price.set_index(['Date'], drop=True, inplace=True)
    smart_beta_price_d = pd.concat([smart_beta_price_d, temp_price], axis=1, sort=True)
smart_beta_price_d.columns = smart_beta_name

smart_beta_return_np_d = np.log(np.array(smart_beta_price_d.iloc[1:, :], dtype=float)) - np.log(np.array(smart_beta_price_d.iloc[:-1, :], dtype=float))
smart_beta_return_d = pd.DataFrame(smart_beta_return_np_d)
smart_beta_return_d.columns = smart_beta_price_d.columns
smart_beta_return_d.index = list(smart_beta_price_d.index)[1:]
# print(smart_beta_return_d)

backtest_date = list(smart_beta_return_d.index)[list(smart_beta_return_d.index).index(init_re) + 1:]
# print(backtest_date)

smart_beta_valuation_return = []
for date in backtest_date:
    temp_update_date = np.array(re_date)[np.argwhere(np.array(re_date) < date)[-1][0]]
    temp_return = np.dot(smart_beta_return_d.loc[date, :].values, smart_beta_weights.loc[temp_update_date, :].values) * (1-2/1000)
    smart_beta_valuation_return.append(temp_return)
smart_beta_valuation_return = pd.DataFrame(smart_beta_valuation_return, index=backtest_date, columns=['估值择时'])

smart_beta_return_d_init = smart_beta_return_d.loc[backtest_date[0]:, :]
# print(smart_beta_return_d_init.sum())
equal_weights_return = smart_beta_return_d_init.mean(axis=1).values
valuation_equal_return = pd.concat([smart_beta_valuation_return, pd.DataFrame(equal_weights_return, index=backtest_date, columns=['等权重'])], axis=1, sort=True)
# print(valuation_equal_return)

# 计算相关指标（回测净值、总收益率、年化收益率、年化波动率、最大回撤率、日胜率）
# 回测净值
strategy_net_value = pd.DataFrame([np.ones(valuation_equal_return.shape[1])], columns=valuation_equal_return.columns)
for date in list(valuation_equal_return.index):
    temp_return = valuation_equal_return.loc[date, :].values
    temp_net_value = strategy_net_value.iloc[-1, :].values
    strategy_net_value = pd.concat([strategy_net_value, pd.DataFrame([np.multiply(temp_net_value, np.exp(temp_return))], columns=valuation_equal_return.columns)], axis=0)
strategy_net_value.index = list(smart_beta_return_d.index)[list(smart_beta_return_d.index).index(init_re):]
# strategy_net_value.to_excel('./data/估值择时_回测净值.xlsx')
# print(strategy_net_value)

# 总收益率
valuation_equal_return_to = valuation_equal_return.sum().values
print(valuation_equal_return_to)

# 年化收益率
valuation_equal_return_sum = valuation_equal_return.sum().values
trading_num = valuation_equal_return.shape[0]
valuation_equal_return_an = 252 / trading_num * valuation_equal_return_sum
print(valuation_equal_return_an)

# 年化波动率
valuation_equal_volatility_an = np.sqrt(252) * valuation_equal_return.std().values
print(valuation_equal_volatility_an)

# 最大回撤率
valuation_equal_drawdown = np.array([])
for strategy in strategy_net_value.columns:
    temp_strategy_net_value = list(strategy_net_value.loc[:, strategy].values)
    # print(temp_strategy_net_value)
    acc_return_dd = [-(i - np.max(temp_strategy_net_value[:temp_strategy_net_value.index(i) + 1])) / np.max(temp_strategy_net_value[:temp_strategy_net_value.index(i) + 1]) for i in temp_strategy_net_value[1:]]
    valuation_equal_drawdown = np.append(valuation_equal_drawdown, np.max(np.array(acc_return_dd)))
print(valuation_equal_drawdown)

# 日胜率
valuation_equal_return_sub = list(valuation_equal_return.iloc[:, 0].values - valuation_equal_return.iloc[:, 1].values)
valuation_equal_return_sub_gr_0 = np.array([1 if i > 0 else 0 for i in valuation_equal_return_sub])
valuation_equal_rw = np.sum(valuation_equal_return_sub_gr_0)/len(valuation_equal_return_sub_gr_0)
print(valuation_equal_rw)

# 换手率
turnover = np.array([])
for date_index in range(len(list(smart_beta_weights.index))):
    if date_index == 0:
        continue
        # temp_turnover = np.sum(smart_beta_weights.iloc[date_index, :].values)
    else:
        temp_weights_sub = smart_beta_weights.iloc[date_index, :].values - smart_beta_weights.iloc[date_index - 1, :].values
        temp_turnover = np.sum(np.abs(temp_weights_sub))
    turnover = np.append(turnover, temp_turnover)
print(turnover)
print(np.mean(turnover))

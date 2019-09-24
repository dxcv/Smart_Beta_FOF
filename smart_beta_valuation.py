import numpy as np
import pandas as pd
from utils import sub_d
# from WindPy import *
from iFinDPy import *

# w.start()
# trading_days = w.tdays(beginTime='2005-04-08', endTime='2019-09-16').Data[0]
# trading_days = [date.strftime("%Y-%m-%d") for date in trading_days]
# trading_days = pd.DataFrame(trading_days)
# print(trading_days)
# # trading_days.to_excel('./data/trading_days.xlsx')
#
# trading_days_w = w.tdays(beginTime='2005-04-08', endTime='2019-09-16', Period='W').Data[0]
# trading_days_w = [date.strftime("%Y-%m-%d") for date in trading_days_w]
# trading_days_w = pd.DataFrame(trading_days_w)
# print(trading_days_w)
# # trading_days_w.to_excel('./data/trading_days_w.xlsx')
#
# trading_days_m = w.tdays(beginTime='2005-04-08', endTime='2019-09-16', Period='M').Data[0]
# trading_days_m = [date.strftime("%Y-%m-%d") for date in trading_days_m]
# trading_days_m = pd.DataFrame(trading_days_m)
# print(trading_days_m)
# # trading_days_m.to_excel('./data/trading_days_m.xlsx')
#
# trading_days_y = w.tdays(beginTime='2005-04-08', endTime='2019-09-16', Period='Y').Data[0]
# trading_days_y = [date.strftime("%Y-%m-%d") for date in trading_days_y]
# trading_days_y = pd.DataFrame(trading_days_y)
# print(trading_days_y)
# # trading_days_y.to_excel('./data/trading_days_y.xlsx')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

trading_days = pd.read_excel('./data/trading_days.xlsx')
trading_days = list(trading_days.iloc[:, 1].values)

trading_days_w = pd.read_excel('./data/trading_days_w.xlsx')
trading_days_w = list(trading_days_w.iloc[:, 1].values)

trading_days_m = pd.read_excel('./data/trading_days_m.xlsx')
trading_days_m = list(trading_days_m.iloc[:, 1].values)
# print(trading_days_m)

# 沪深300成份股
CSI300_con = pd.read_excel('./data/300成份进出记录.xlsx')
# print(CSI300_con)
trading_days_CSI300 = trading_days[trading_days.index(CSI300_con.iloc[-1, 0]):]
CSI300_constituents = pd.DataFrame()
exclude = np.array([])
transfer_not_days = []
transfer_all_days = []
for date in trading_days_CSI300:
    print(date)
    if date in list(CSI300_con.loc[CSI300_con['状态'].isin(['纳入'])].loc[:, '日期'].values):
        include = CSI300_con.loc[CSI300_con['状态'].isin(['纳入'])].loc[CSI300_con['日期'].isin([date])].loc[:, '代码'].values
        transfer_all_days.append(date)
        if len(include) > 10:
            transfer_not_days.append(date)
        if exclude.size != 0:
            exclude_index = [list(CSI300_constituents.iloc[-1].values).index(i) for i in list(exclude)]
            temp = np.sort(np.append(np.delete(CSI300_constituents.iloc[-1].values, exclude_index), include))
            CSI300_constituents = pd.concat([CSI300_constituents, pd.DataFrame([temp])], axis=0)
        else:
            if CSI300_constituents.shape[0] != 0:
                temp = np.sort(np.append(CSI300_constituents.iloc[-1].values, include))
                CSI300_constituents = pd.concat([CSI300_constituents, pd.DataFrame([temp])], axis=0)
            else:
                CSI300_constituents = pd.concat([CSI300_constituents, pd.DataFrame([np.sort(CSI300_con.loc[CSI300_con['日期'].isin([date])].loc[:, '代码'].values)])], axis=0)
        exclude = np.array([])
        if date in list(CSI300_con.loc[CSI300_con['状态'].isin(['剔除'])].loc[:, '日期'].values):
            exclude = np.append(exclude, CSI300_con.loc[CSI300_con['状态'].isin(['剔除'])].loc[CSI300_con['日期'].isin([date])].loc[:, '代码'].values)
    elif date in list(CSI300_con.loc[CSI300_con['状态'].isin(['剔除'])].loc[:, '日期'].values):
        exclude = np.append(exclude, CSI300_con.loc[CSI300_con['状态'].isin(['剔除'])].loc[CSI300_con['日期'].isin([date])].loc[:, '代码'].values)

CSI300_constituents.index = transfer_all_days
pd.DataFrame(transfer_not_days).to_excel('./data/transfer_days.xlsx')
CSI300_constituents.to_excel('./data/300成份股.xlsx')
print(transfer_not_days)

transfer_not_days = pd.read_excel('./data/transfer_days.xlsx')
transfer_not_days = list(transfer_not_days.iloc[:, 1].values)
# print(transfer_not_days)

# # 得到各Smart Beta指数的发布时间
# # 300动量（H30260.CSI）、300波动（000803.CSI）、300红利（000821.CSI）、300低贝塔（000829.CSI）、300成长（000918.SH）、300价值（399919.SZ）、300等权（000984.SH）
smart_beta_name = ['沪深300', '300波动', '300相对成长', '300相对价值', '300红利', '300低贝塔', '300等权']
smart_beta_list = ['000300.SH', '000803.CSI', '000920.CSI', '000921.CSI', '000821.CSI', '000829.CSI', '000984.SH']
# smart_beta_date_list = []
# for index in smart_beta_list:
#     temp_smart_beta = w.wset('indexhistory', 'startdate=2005-04-08;enddate=2019-09-04;windcode=' + index, usedf=True)[1]
#     temp_smart_beta.sort_values(by='tradedate', ascending=True, inplace=True)
#     temp_smart_beta_date = temp_smart_beta.iloc[0, 0]
#     smart_beta_date_list.append(temp_smart_beta_date.strftime('%Y-%m-%d'))
# start_date = max(smart_beta_date_list)
# print(smart_beta_date_list)
# print(start_date)

# transfer_not_days = [date for date in transfer_not_days if date > '2013-11-21']
# # print(transfer_not_days)
# trading_days_mm = [date[:7] for date in trading_days]
# trading_days_mm_np = np.array(trading_days_mm)
# weights_update_days = [trading_days[np.argwhere(trading_days_mm_np == date[:7])[-1][0]] for date in transfer_not_days]
# # print(weights_update_days)

trading_days_m = [date for date in trading_days_m if date > '2012-12-21']
print(trading_days_m)

# print(THS_iFinDLogin('#######', '######'))
# for index in smart_beta_list:
#     weights = pd.DataFrame()
#     for date in trading_days_m:
#         temp_date_weights = THS_DataPool('index', date + ';' + index, 'date:Y,thscode:Y,weight:Y')
#         weights = pd.concat([weights, pd.DataFrame.from_dict(temp_date_weights['tables'][0]['table'])], axis=0)
#     index_name = smart_beta_name[smart_beta_list.index(index)]
#     weights.to_excel('./data/' + index_name + '成份权重.xlsx')

# # 沪深300历史所有成分股
# # CSI300_constituents_unique = pd.read_excel('./data/300历史成份股.xlsx', index_col=0)
# # CSI300_constituents_unique_list = list(CSI300_constituents_unique.iloc[:, 0].values)
# # print(CSI300_constituents_unique_list)

market_value = pd.read_excel('./data/总市值.xlsx', index_col=0)
market_value.reset_index(drop=False, inplace=True)
market_value.loc[:, 'index'] = market_value.loc[:, 'index'].apply(sub_d)
market_value.set_index(['index'], drop=True, inplace=True)
# print(market_value)

book = pd.read_excel('./data/归属母公司股东的股东权益.xlsx', index_col=0)
book.reset_index(drop=False, inplace=True)
book.loc[:, 'index'] = book.loc[:, 'index'].apply(sub_d)
book.set_index(['index'], drop=True, inplace=True)

pb = pd.read_excel('./data/300历史成份股市净率.xlsx', index_col=0)
pb.reset_index(drop=False, inplace=True)
pb.loc[:, 'index'] = pb.loc[:, 'index'].apply(sub_d)
pb.set_index(['index'], drop=True, inplace=True)
# print(pb)

earnings = pd.read_excel('./data/归属母公司股东的净利润.xlsx', index_col=0)
earnings.reset_index(drop=False, inplace=True)
earnings.loc[:, 'index'] = earnings.loc[:, 'index'].apply(sub_d)
earnings.set_index(['index'], drop=True, inplace=True)

sales = pd.read_excel('./data/营业收入.xlsx', index_col=0)
sales.reset_index(drop=False, inplace=True)
sales.loc[:, 'index'] = sales.loc[:, 'index'].apply(sub_d)
sales.set_index(['index'], drop=True, inplace=True)

dividends_q = pd.read_excel('./data/区间现金分红总额_季.xlsx', index_col=0)
dividends_q.reset_index(drop=False, inplace=True)
dividends_q.loc[:, 'index'] = dividends_q.loc[:, 'index'].apply(sub_d)
dividends_q.set_index(['index'], drop=True, inplace=True)

dividends_y = pd.read_excel('./data/区间现金分红总额.xlsx', index_col=0)
dividends_y.reset_index(drop=False, inplace=True)
dividends_y.loc[:, 'index'] = dividends_y.loc[:, 'index'].apply(sub_d)
dividends_y.set_index(['index'], drop=True, inplace=True)

issuing_date = pd.read_excel('./data/定期报告实际披露日期.xlsx', index_col=0)
issuing_date.reset_index(drop=False, inplace=True)
issuing_date.loc[:, 'index'] = issuing_date.loc[:, 'index'].apply(sub_d)
issuing_date.set_index(['index'], drop=True, inplace=True)
issuing_date.replace(0, '2022-12-31', inplace=True)


# 沪深300成份权重
CSI_300_weights = pd.read_excel('./data/沪深300成份权重.xlsx')
CSI_300_weights = CSI_300_weights.iloc[:, 1:]
CSI_300_weights.set_index(['DATE'], drop=True, inplace=True)

# Smart Beta估值
# 常用估值指标
# 市净率（PB）、市盈率（PE）、市销率（PS）、本利比（PD）
valuation_indicator = ['市净率', '市盈率', '市销率', '本利比']
for index_name in smart_beta_name[1:]:
    # 估值
    if index_name == '300等风险':
        continue
    # print(index_name)
    # Smart Beta成分权重
    smart_beta_data = pd.read_excel('./data/' + index_name + '成份权重.xlsx')
    smart_beta_data = smart_beta_data.iloc[:, 1:]
    smart_beta_data.set_index(['DATE'], drop=True, inplace=True)
    # print(smart_beta_data)

    relative_valuation_concat = pd.DataFrame()
    valuation_c = ''
    for valuation in valuation_indicator:
        if valuation == '市盈率':
            valuation_c = 'pe'
            valuation_data = earnings
        elif valuation == '市销率':
            valuation_c = 'ps'
            valuation_data = sales
        elif valuation == '市净率':
            valuation_c = 'pb'
            valuation_data = pb
        else:
            valuation_c = 'pd'
            valuation_data = dividends_q
            valuation_data_y = dividends_y

        relative_valuation_list = []
        # w_date = [v for v in trading_days_w if v in list(market_value.index)]
        # m_date = [v for v in trading_days_m if v in list(market_value.index)]
        for date in trading_days_m:
            # 当月smart beta指数成分股及权重
            # temp_weights_update_days = weights_update_days[np.argwhere(np.array(weights_update_days) <= date)[-1][0]]
            temp_constituents = list(smart_beta_data.loc[date, 'THSCODE'].values)
            temp_weights_smart = smart_beta_data.loc[date, 'WEIGHT']
            # print(temp_constituents)
            # 当月对应成分股及市值权重
            temp_300_constituents = CSI_300_weights.loc[date, :]
            temp_300_smart_beta = temp_300_constituents.loc[temp_300_constituents['THSCODE'].isin(temp_constituents)]
            temp_weights_cap = temp_300_smart_beta.loc[:, 'WEIGHT'].values / np.sum(temp_300_smart_beta.loc[:, 'WEIGHT'].values)
            # 当日成份股估值
            # 当日各成份股可获得的已披露的最新报告日期
            temp_constituents_issuing_date = issuing_date.loc[:, temp_constituents]
            # print(temp_constituents)
            # print(temp_constituents_issuing_date)
            # print(list(issuing_date.index)[np.argwhere(temp_constituents_issuing_date.loc[:, '000024.SZ'].values <= date)[-1][0]])
            print(date)
            temp_constituents_latest_date = [list(issuing_date.index)[np.argwhere(temp_constituents_issuing_date.loc[:, code].values <= date)[-1][0]] for code in temp_constituents]
            temp_constituents_quarter = [int(int(temp_constituents_latest_date[temp_constituents.index(code)][5:7]) / 3) for code in temp_constituents]
            if valuation_c == 'pb':
                valuation_value = [valuation_data.loc[date, code] for code in temp_constituents]
                # print(temp_book)
                # valuation_value = market_value.loc[date, temp_constituents].values/np.array(temp_book)
            elif valuation_c == 'pe':
                temp_earnings = [(valuation_data.loc[str(int(temp_constituents_latest_date[temp_constituents.index(code)][:4])-4) + '-12-31', code] + valuation_data.loc[str(int(temp_constituents_latest_date[temp_constituents.index(code)][:4])-3) + '-12-31', code] + valuation_data.loc[str(int(temp_constituents_latest_date[temp_constituents.index(code)][:4])-2) + '-12-31', code] + valuation_data.loc[str(int(temp_constituents_latest_date[temp_constituents.index(code)][:4])-1) + '-12-31', code] + valuation_data.loc[temp_constituents_latest_date[temp_constituents.index(code)], code]*(4/temp_constituents_quarter[temp_constituents.index(code)]))/5 for code in temp_constituents]
                valuation_value = market_value.loc[date, temp_constituents].values/np.array(temp_earnings)
            elif valuation_c == 'ps':
                temp_sales = [(valuation_data.loc[str(int(temp_constituents_latest_date[temp_constituents.index(code)][:4])-4) + '-12-31', code] + valuation_data.loc[str(int(temp_constituents_latest_date[temp_constituents.index(code)][:4])-3) + '-12-31', code] + valuation_data.loc[str(int(temp_constituents_latest_date[temp_constituents.index(code)][:4])-2) + '-12-31', code] + valuation_data.loc[str(int(temp_constituents_latest_date[temp_constituents.index(code)][:4])-1) + '-12-31', code] + valuation_data.loc[temp_constituents_latest_date[temp_constituents.index(code)], code]*(4/temp_constituents_quarter[temp_constituents.index(code)]))/5 for code in temp_constituents]
                valuation_value = market_value.loc[date, temp_constituents].values/np.array(temp_sales)
            else:
                # print(valuation_data[temp_constituents_latest_date[temp_constituents.index('000001.SZ')], '000001.SZ'] == 0)
                temp_dividends = [valuation_data_y.loc[str(int(temp_constituents_latest_date[temp_constituents.index(code)][:4])-5) + '-12-31':str(int(temp_constituents_latest_date[temp_constituents.index(code)][:4])-1) + '-12-31', code].sum()/5 if valuation_data.loc[temp_constituents_latest_date[temp_constituents.index(code)], code] == 0 else (valuation_data_y.loc[str(int(temp_constituents_latest_date[temp_constituents.index(code)][:4])-4) + '-12-31':str(int(temp_constituents_latest_date[temp_constituents.index(code)][:4])-1) + '-12-31', code].sum() + valuation_data.loc[temp_constituents_latest_date[temp_constituents.index(code)], code]) / 5 for code in temp_constituents]
                valuation_value = market_value.loc[date, temp_constituents].values/np.array(temp_dividends)
            # Smart beta加权估值
            smart_val = np.dot(valuation_value, temp_weights_smart.values) / 100
            # print(smart_val)
            # 市值加权估值
            cap_val = np.dot(valuation_value, temp_weights_cap)
            # print(cap_val)
            # 相对估值
            relative_valuation = smart_val / cap_val
            relative_valuation_list.append(relative_valuation)
        relative_valuation_df = pd.DataFrame(relative_valuation_list)
        relative_valuation_df.index = trading_days_m

        relative_valuation_concat = pd.concat([relative_valuation_concat, relative_valuation_df], axis=1, sort=True)
    relative_valuation_concat.columns = valuation_indicator
    relative_valuation_concat.to_excel('./data/' + index_name + '_估值.xlsx')


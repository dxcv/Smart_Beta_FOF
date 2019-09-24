from jqdatasdk import *
import pandas as pd

auth('###########', '##########')
pd.set_option('display.max_rows', None)
df = get_index_weights('000984.XSHG', date='2019-04-30')
print(df)

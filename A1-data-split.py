"""
Created on Sun Feb  7 23:52:26 2021

@author: x
"""
import pandas as pd

df = pd.read_excel('/Users/vanessazhu/Desktop/selectdata.xlsx', sheet_name='Sheet1')

df.columns

df['Issue Date'] = pd.to_datetime(df['Issue Date'], format='%Y/%m/%d').dt.date
# df['Issue Date'] = df['Issue Date'].apply(lambda x: x.strftime('%Y/%m/%d'))

df['Maturity Date'] = pd.to_datetime(df['Maturity Date'], format='%Y/%m/%d').dt.date
# df['Maturity Date'] = df['Maturity Date'].apply(lambda x: x.strftime('%Y/%m/%d'))


df.drop(['< 0-3 years bonds', 'Issue Price', 'No. of Payments per Year',
         'Last Coupon Date', 'Months since last coupon date',
         'months until maturity date', 'Coupon Payment Date',
         'Coupon Start Date', 'Final Coupon Date', 'Close Price'], axis=1, inplace=True)

df = df.dropna(how='all')

# df['Close Price']

selected_bonds = ['CA135087A610', 'CA135087F254', 'CA135087F585', 'CA135087G328', 'CA135087H490',
                  'CA135087ZU15', 'CA135087J546', 'CA135087J967', 'CA135087K528', 'CA135087K940', 'CA135087L518']

df = df[df['ISIN'].isin(selected_bonds)]

comm_cols = ['Name', 'ISIN', 'Coupon', 'Issue Date', 'Maturity Date']
# isin_cols = ['2021-01-18', '2021-01-19', '2021-01-20', '2021-01-21', '2021-01-22',
#              '2021-01-25', '2021-01-26', '2021-01-27', '2021-01-28', '2021-01-29']
# isin_cols = ['2021/1/18', '2021/1/19', '2021/1/20', '2021/1/21', '2021/1/22',
#              '2021/1/25', '2021/1/26', '2021/1/27', '2021/1/28', '2021/1/29']

isin_cols = list(pd.bdate_range('2021-01-18', '2021-01-31'))
date_cols = df.columns.difference(comm_cols)

with pd.ExcelWriter('./data/data6.xlsx') as writer:
    for col, isin_col in zip(date_cols, isin_cols):
        # sub_df.columns = ['name', 'ISIN', 'coupon', 'issue date', 'maturity date', 'close price']
        # sub_df  = sub_df[['name', 'close price', 'coupon', 'ISIN', 'issue date', 'maturity date']]
        # sub_df = df[comm_cols + [col]].rename(columns={'Name': isin_col})
        sub_df = df[comm_cols + [col]].rename(
            columns={'Name': isin_col, 'Coupon': 'coupon', 'Issue Date': 'issue date', 'Maturity Date': 'maturity date',
                     col: 'close price'})
        print(sub_df.columns)
        # sub_df.rename(columns = {sub_df.columns[-1]: "close price", 'Coupon': 'coupon', 'Issue Date': 'issue date', 'Maturity Date': 'maturity date'}, inplace = True)
        # sub_df = df[comm_cols + [col]].rename(columns={'Name': isin_col})
        sub_df.to_excel(writer, sheet_name=col, index=False)
    writer.save()

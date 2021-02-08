"""
Created on Sun Feb  7 23:53:24 2021

@author: x
"""


import scipy.interpolate
import pandas as pd
from pandas import *
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from numpy import linalg as LA

# read the data file in the repository
xls = pd.ExcelFile('/Users/vanessazhu/Desktop/originaldata.xlsx')
df1 = pd.read_excel(xls, 'Jan 18th')
df2 = pd.read_excel(xls, 'Jan 19th')
df3 = pd.read_excel(xls, 'Jan 20th')
df4 = pd.read_excel(xls, 'Jan 21th')
df5 = pd.read_excel(xls, 'Jan 22th')
df6 = pd.read_excel(xls, 'Jan 25th')
df7 = pd.read_excel(xls, 'Jan 26th')
df8 = pd.read_excel(xls, 'Jan 27th')
df9 = pd.read_excel(xls, 'Jan 28th')
df10 = pd.read_excel(xls, 'Jan 29th')
df = [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10]

# =============================================================================
# a)
# =============================================================================
# calculation of half year time to maturity
def ttm(x):
    current_date = list(x.columns.values)[0]
    x['time to maturity'] = [(maturity - current_date).days for maturity in x['maturity date']]
    
# calculation of yield to maturity
def ytm(x):
    tr, yr = [], []
    current_date = list(x.columns.values)[0]
    for i, bond in x.iterrows():
        ttm = bond['time to maturity']
        tr.append(ttm/365)
        
        # separate the time to maturity to small time interval
        y = int(ttm/182)
        init = (ttm%182)/365
        time = np.asarray([2 * init + n for n in range(0,y+1)])

        # convert the clean close price to dirty price
        coupon = bond['coupon']*100
        accrued_interest = coupon * ((182-ttm%182)/365)
        dirty_price = bond['close price'] + accrued_interest
        
        # make each payments in different time period as an array
        pmt = np.asarray([coupon/2] * y + [coupon/2 + 100])
        
        # use optimization to solve the yield to maturity
        ytm_func = lambda y: np.dot(pmt, (1+y/2) ** (-time)) - dirty_price
        ytm = optimize.fsolve(ytm_func, .05)
        yr.append(ytm)
    return tr, yr

# plot the yield curve:
labels = ['Jan 18th', 'Jan 19th', 'Jan 20th', 'Jan 21th', 'Jan 22th', 'Jan 25th',
       'Jan 26th', 'Jan 27th', 'Jan 28th', 'Jan 29th']
plt.xlabel('time to maturity')
plt.ylabel('yield to maturity')
plt.title('original 5-year yield curve')
i = 0
for d in df:
    ttm(d)
    plt.plot(ytm(d)[0], ytm(d)[1], label = labels[i])
    i = i+1
plt.legend(loc = 'upper right', prop={"size":8})


# interpolation
def ip(tr, yr):
    t = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    y = []
    interp = scipy.interpolate.interp1d(tr, yr, bounds_error=False)
    for i in t:
        value = float(interp(i))
        if not scipy.isnan(value):
            y.append(value)
    return t,y

# plot the interpolated yield curve:
labels = ['Jan 18th', 'Jan 19th', 'Jan 20th', 'Jan 21th', 'Jan 22th', 'Jan 25th',
       'Jan 26th', 'Jan 27th', 'Jan 28th', 'Jan 29th']
plt.xlabel('time to maturity')
plt.ylabel('yield to maturity')
plt.title('interpolated 5-year yield curve')
i = 0
for d in df:
    ttm(d)
    result = ytm(d)
    a = np.asarray(result[0])
    b = np.asarray(result[1]).squeeze()
    y = ip(a,b)
    plt.plot(y[0], y[1], label = labels[i])
    i = i+1
plt.legend(loc = 'upper right', prop={"size":8})

# =============================================================================
# b)
# =============================================================================
# bootstrapping and calculate the spot rate:
def spot(x):
    s = np.empty([1, 11])
    tr = []
    coupons = []
    dirty_price = []
    for i, bond in x.iterrows():
        ttm = bond['time to maturity']
        tr.append(ttm/365)
        coupon = bond['coupon']*100
        coupons.append(coupon)
        accrued_interest = coupon * (0.5 - (ttm % 182)/365)
        dirty_price.append(bond['close price'] + accrued_interest)
        
    for i in range(0, 11):
        if i == 0:
            # 0 <= T <= 0.5:
            s[0, i] = -np.log(dirty_price[i]/(coupons[i]/2+100))/tr[i]
        else:
            # 0.5 <= T <= 1:
            pmt = np.asarray([coupons[i]/2] * i + [coupons[i]/2 + 100])
            spot_func = lambda y: np.dot(pmt[:-1], 
                        np.exp(-(np.multiply(s[0,:i], tr[:i])))) + pmt[i] * np.exp(-y * tr[i]) - dirty_price[i]      
            s[0, i] = optimize.fsolve(spot_func, .05)
    s[0, 5] = (s[0, 4] + s[0, 6])/2
    s[0, 7] = (s[0, 5] + s[0, 8])/2
    return tr, s

# plot the spot curve:
labels = ['Jan 18th', 'Jan 19th', 'Jan 20th', 'Jan 21th', 'Jan 22th', 'Jan 25th',
       'Jan 26th', 'Jan 27th', 'Jan 28th', 'Jan 29th']
plt.xlabel('time to maturity')
plt.ylabel('spot rate')
plt.title('5-year spot curve')
i = 0
for d in df:
    ttm(d)
    plt.plot(spot(d)[0], spot(d)[1].squeeze(), label = labels[i])
    i = i+1
plt.legend(loc = 'upper right', prop={"size":8})


# spot rate after interpolation:
labels = ['Jan 18th', 'Jan 19th', 'Jan 20th', 'Jan 21th', 'Jan 22th', 'Jan 25th',
       'Jan 26th', 'Jan 27th', 'Jan 28th', 'Jan 29th']
plt.xlabel('time to maturity')
plt.ylabel('yield to maturity')
plt.title('interpolated 5-year spot curve')
i = 0
for d in df:
    ttm(d)
    result = spot(d)
    a = np.asarray(result[0])
    b = np.asarray(result[1]).squeeze()
    y = ip(a,b)
    plt.plot(y[0], y[1], label = labels[i])
    i = i+1
plt.legend(loc = 'upper right', prop={"size":8})

# =============================================================================
# c)
# =============================================================================
# the forward curve:
def forward(x):
    ttm(d)
    result = spot(d)
    a = np.asarray(result[0])
    b = np.asarray(result[1]).squeeze()
    y = ip(a,b)
    f1 = (y[1][3] * 2 - y[1][1] * 1)/(2-1)
    f2 = (y[1][5] * 3 - y[1][1] * 1)/(3-1)
    f3 = (y[1][7] * 4 - y[1][1] * 1)/(4-1)
    f4 = (y[1][9] * 5 - y[1][1] * 1)/(5-1)
    f = [f1,f2,f3,f4]
    return f

# plot the forward curve:
labels = ['Jan 18th', 'Jan 19th', 'Jan 20th', 'Jan 21th', 'Jan 22th', 'Jan 25th',
       'Jan 26th', 'Jan 27th', 'Jan 28th', 'Jan 29th']
plt.xlabel('year to year')
plt.ylabel('forward rate')
plt.title('1-year forward curve')
i = 0
f_m = np.empty([4, 10])
for d in df:
    ttm(d)
    plt.plot(['1yr-1yr','1yr-2yr','1yr-3yr','1yr-4yr'], forward(d), label = labels[i])
    f_m[:,i] = forward(d)
    i = i+1
plt.legend(loc = 'upper right', prop={"size":8})



# the calculation of log returns matrix of yield
log = np.empty([5, 9])
yi = np.empty([5, 10])
for i in range(len(df)):
    ttm(df[i])
    result = ytm(df[i])
    a = np.asarray(result[0])
    b = np.asarray(result[1]).squeeze()
    y = ip(a,b) 
    yi[0,i] = y[1][1]
    yi[1,i] = y[1][3]
    yi[2,i] = y[1][5]
    yi[3,i] = y[1][7]
    yi[4,i] = y[1][9]
    
for i in range(0, 4):
    log[0, i] = np.log(yi[0,i+1]/yi[0,i])
    log[1, i] = np.log(yi[1,i+1]/yi[1,i])
    log[2, i] = np.log(yi[2,i+1]/yi[2,i])
    log[3, i] = np.log(yi[3,i+1]/yi[3,i])
    log[4, i] = np.log(yi[4,i+1]/yi[4,i])
    
       
# calculation of the covariance matrix
np.cov(log)
# np.cov(log.T, rowvar=0)

# calculation of the covariance matrix
np.cov(f_m)

# eigenvalues and eigenvectors of covariance matrix of log returns of yield:
# w, v = LA.eig(np.cov(log))
# w

# eigenvalues and eigenvectors of covariance matrix of forward rates:
a, b = LA.eig(np.cov(f_m))
a

b









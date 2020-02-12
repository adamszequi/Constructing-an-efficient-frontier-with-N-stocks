# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 23:42:39 2020

@author: Dell
"""

import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import fmin
import os
import datetime
import matplotlib.pyplot as plt
from numpy.linalg import inv,pinv

dataLocations=[r'C:\Users\Dell\Desktop\data\2010-2019\GOIL VWAP closing prces (2).xlsx',\
               r'C:\Users\Dell\Desktop\data\2010-2019\SCB VWAP closing prces.xlsx',\
               r'C:\Users\Dell\Desktop\data\2010-2019\SIC VWAP closing prces.xlsx',\
               r'C:\Users\Dell\Desktop\data\2010-2019\TOTAL VWAP closing prces (2).xlsx',\
               r'C:\Users\Dell\Desktop\data\2010-2019\TBL VWAP closing prces (2).xlsx']

uploadedData=[]
def retriveData(ticker):
        for tick in ticker:
            data=pd.read_excel(tick,parse_dates=[1]) 
            uploadedData.append(data)
        return uploadedData

#code for defining two functions(return function and objective function)
def monthlyReturn(ticker):
    data=retriveData(ticker)
    returns=[]
    for _ in data:
        logReturns=np.log(_['Closing Price VWAP (GHS)'][1:].values/\
                          _['Closing Price VWAP (GHS)'][:-1].values)
        returns.append(logReturns)
        date=[]
        for dated in range(0,np.size(logReturns)):
             #getting the dates from the first item on list
             dates=data[0]['Date']
             #appending onlu 4 characters of date whixch are the year
             date.append(dates[dated][:4])
        annuallyCummulatedReturnsList=pd.DataFrame(returns).T
        annuallyCummulatedReturnsList.index=pd.Series(date)
        #annuallyCummulatedReturnsList.columns=['GOIL','SCB','SIC','TOTAL']
    return annuallyCummulatedReturnsList.groupby\
        (annuallyCummulatedReturnsList.index).sum()
   # return annuallyCummulatedReturnsList.T


def objectiveFunction(w,r,targetReturns):
    stockMean=np.mean(r,axis=0)
    portfolioMean=np.dot(w,stockMean)
    #variance-covariance matrix
    varianceCovarianceMatrix=np.cov(r.T)
    portfolioVariance=np.dot(np.dot(w,varianceCovarianceMatrix),w.T)
    penalty=2000*abs(portfolioMean-targetReturns)
    return np.sqrt(portfolioVariance)+penalty


outMean,outStandardDeviation,outWeight=[],[],[]
finalReturns=np.array(monthlyReturn(dataLocations))
finalReturnsMean=np.mean(finalReturns,axis=0)
numberStocksInList=len(dataLocations)

#returns numbers with equal intervals in samples(num) wher min and max
for _ in np.linspace(np.min(finalReturnsMean),\
                     np.max(finalReturnsMean),num=100):
    #starting from equal weights where np.ones creates lists of 
    #ones and divides by len(list)
    weights=np.ones([numberStocksInList])/numberStocksInList
    #bounds of minimization function where create 0,1 in len(number of stocks)
    b_=[(0,1) for i in range(numberStocksInList) ] 
    #constraints of minimization function
    c_=({'type':'eq', 'fun': lambda weights: sum(weights)-1})
    #optimization function
    results=sp.optimize.minimize(objectiveFunction,weights,(finalReturns,_),\
                                 method='SLSQP',constraints=c_,bounds=b_)
   
    if not results.success:
        BaseException(results.message)
    outMean.append(round(_,4))
    standardDeviation=round(np.std(np.sum(finalReturns*results.x,axis=1)),6)
    outStandardDeviation.append(standardDeviation)
    outWeight.append(results.x)

print(results)
#plotting graph for efficient frontier
plt.title('EFFICIENT FRONTIER')
plt.xlabel('STANDARD DEVIATION')
plt.ylabel('RETURN OF PORTFOLIO')
plt.figtext(0.5,0.75,str(numberStocksInList)+' stock are used: ')
plt.figtext(0.5,0.7,' '+str(['GOIL','SCB','SIC','TOTAL','TBL']))
plt.figtext(0.5,0.65,'Time period: '+str(2010)+' ------'+str(2019))
plt.plot(outStandardDeviation,outMean,'--')
plt.show()

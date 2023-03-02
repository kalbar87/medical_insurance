# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 11:30:25 2023

@author: michalk
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import  cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error



def distribution_plot(feature):
    sk = feature.skew()
    kurt = feature.kurtosis()
    fig, axd = plt.subplot_mosaic([['upper left', 'right'],
                                   ['lower left', 'right']], figsize=(14,12))

    #Histogram
    sns.histplot(feature, ax=axd['upper left'], stat='density', kde='False', color='skyblue')
    sns.kdeplot(feature, color='blue', ax=axd['upper left'], label='KDE')
    axd['upper left'].axvline(x = feature.mean(), c='red', linewidth=1.5, label = 'mean')
    axd['upper left'].axvline(x = feature.median(), c='green', linewidth=1.5, label = 'median')
    axd['upper left'].legend()
    
    #Pobability plot
    stats.probplot(feature, plot=axd['lower left'])
    value = [sk, kurt]
    lbl = ['Skew', 'Kurtosis']
    
    [axd['lower left'].text(0.03,0.9-0.1*x,'%s = %.2f' %(lbl[x],y) ,
                           fontsize=18, transform=axd['lower left'].transAxes, 
                           bbox=dict(facecolor='skyblue', alpha=.3)) for x,y in enumerate(value)]

    #Box plot
    medianprops = dict(linewidth=3, linestyle='-', color='crimson')
    boxprops = dict(linestyle='-', linewidth=1.5, color='darkblue')
    capprops = dict(color='darkblue')
    whiskerprops = dict(color='darkblue')

    xs = np.random.normal(1,0.04, feature.shape[0])
    axd['right'].scatter(xs, feature, c='skyblue', alpha=.5)
    axd['right'].boxplot(feature, medianprops=medianprops, boxprops=boxprops, capprops=capprops, whiskerprops=whiskerprops)
    axd['right'].set_xlabel(feature.name)

def residual_plot(y_test, y_pred, fig, ax0, ax1, color, lbl,i):
    res = y_pred-y_test
    max_res = max(abs(res))
    ind = np.where(abs(res) == max_res)
 

    ax0.plot(y_pred, res,'o', markeredgecolor=color, label=lbl, zorder=i)
    ax0.axhline(y=0, xmin=0, xmax=40000, c='black', zorder=1)
    ax0.vlines(x = y_pred[ind], ymin = 0, ymax = res.iloc[ind].values, color='dark%s' %color)
    ax0.set_ylim(-30000,30000)
    #ax0.set_xlim(-1,8)
    ax0.set_title('Target Residual plot', fontsize=22)
    ax0.set_ylabel('Residuals')
    ax0.set_xlabel('Predictied values')
    ax0.legend()
    sns.histplot(y=res, bins=40,ax=ax1, color='light%s'%color, stat='density')
    sns.kdeplot(y=res, color=color, ax=ax1)
    ax1.axhline(y=0, xmin=0, xmax=5, c='black', zorder=1)
    ax1.set_ylim(-30000, 30000)
    ax1.text(0.3,1-0.05*i, '%s_mean = %.2f' %(lbl,np.mean(res)), transform=ax1.transAxes, fontsize=13)
    ax1.axis('off')

def qq_plot(real, pred, color, order, lbl, ax0, ax1):
    plt.rc('lines', markersize='8', markeredgecolor=color, markerfacecolor='white', markeredgewidth=1, linewidth=2)
    fig = stats.probplot(real-pred, plot=ax0) 
    ax1.plot(real, pred, 'o', alpha=0.7, zorder=order, label=lbl)
    sns.regplot(x=real, y=pred, scatter=False, ax=ax1,color='blue', ci=0)
    ax1.set_xlabel('Charge')
    ax1.set_ylabel('Predicted Charge')
    ax1.set_title('Test vs predicted values')
    ax1.legend()

def model_score(model, X,y, kf):
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=0.2)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    RMSE = np.sqrt(-cross_val_score(model, X_train,y_train, cv=kf, scoring = 'neg_mean_squared_error')).mean()
    MAE = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    _, p_value = stats.pearsonr(y_test, y_pred)
    print('%s model\n\n' %str(model).split('(')[0],
          'ROOT MEAN SQUARED ERROR:\t%.2f\n' %RMSE,
          'MEAN ABSOLUTE ERROR:\t\t%.2f\n' %MAE,
          'P-VALUE:\t\t\t%.1e\n\n' %p_value,
          'R2 SCORE:\t\t\t%.2f %%' %(r2*100)
          )

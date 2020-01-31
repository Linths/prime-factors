# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:43:43 2020

@author: super
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

mods = [3,5,7,11,13,17,19,23,29]
#mods = [3,5]
for i,mod in enumerate(mods):
    res = np.arange(1,mod)
    repl = np.arange(0,mod-1)
    
    df_mod = pd.read_excel('ConfMatrices.xlsx', sheet_name='mod%d'%mod, header=None)
    
    df_mod.columns = res
    df_mod = df_mod.rename(index=dict(zip(repl,res)))
    y_min = mod-1
    
    s = int(mod/2)
    plt.figure(i, figsize=(s,s))
    
    plt.figure(i)
    hm = sns.heatmap(df_mod, annot=True, cmap='Blues', fmt='g', vmin=0)
    axes = hm.axes
    axes.set_ylim(y_min, 0)
    hm.xaxis.set_ticks_position('top')
    hm.xaxis.set_label_position('top')
    hm.set(xlabel='predicted', ylabel='actual')
    
    heatplot = hm.get_figure()
    heatplot.savefig("heatmap_mod%d.png"%mod, bbox_inches='tight')
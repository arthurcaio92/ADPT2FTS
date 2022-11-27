# -*- coding: utf-8 -*-
from Gridsearch import run_Gridsearch
from datasets import get_TAIEX,get_NASDAQ,get_Brent_Oil,get_SP500
import numpy as np
import pandas as pd

'------------------------------------------------ Data set import -------------------------------------------------'

taiex_df = get_TAIEX()
taiex = taiex_df.avg               
taiex = taiex.to_numpy()  

df_brent_oil = get_Brent_Oil()
brent_oil = df_brent_oil.Price  
brent_oil = brent_oil.to_numpy()


'------------------------------------------------ Gridsearch Parameters -------------------------------------------------'

datasets = [taiex,brent_oil]
dataset_names = ['taiex','ebop']
diff = 1                                       #If diff = 1, data is differentiated
partition_parameters = np.arange(1,5)         #partiions must be a list
orders = [1,2,3]
partitioners = ['ADP']                        #partitioners: 'chen' 'SODA' 'ADP' 'DBSCAN' 'CMEANS' 'entropy' 'FCM'  
mfs = ['triangular','trapezoidal','gaussian']                           #mfs: 'triangular' ou 'trapezoidal' ou 'gaussian'


'------------------------------------------------ Running the model -------------------------------------------------'


'Builds and runs the model'
run_Gridsearch(datasets,dataset_names,diff,partition_parameters,orders,partitioners,mfs,training = 0.7)



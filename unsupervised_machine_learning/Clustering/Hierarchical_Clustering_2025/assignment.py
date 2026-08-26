# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:47:39 2026

@author: Ayush
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from AutoClean import AutoClean
print("AutoClean imported successfully!")

from sklearn.preprocssing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from clusteval import clusteval

from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
from urllib.parse import quote


uni = pd.read_csv(r"C:\Users\admin\Downloads\customer_segmentation_clustering_dataset.csv")

user = '********'
pw = quote("***********")  
db = 'coustomer_segmentation'  
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

uni.to_sql('coustomer_tbl', con = engine, if_exists = 'replace', chunksize = 1000, index = False)



sql = 'select * from coustomer_tbl;'
df = pd.read_sql_query(text(sql), engine.connect())


df.info()


df.describe() 


import dtale


d = dtale.show(df, host = 'localhost', port = 8000)


d.open_browser()


input("Press Enter to continue...")



df.drop(['Customer_ID'], axis = 1, inplace = True)  
df.info()  

print(df.dtypes)
print(df.iloc[:, 1:].dtypes)

from AutoClean import AutoClean

clean_pipeline = AutoClean(
    df.iloc[:, 1:].astype(float, errors='ignore'),        
    mode = 'manual',        
    missing_num = 'auto',    
    outliers = 'winz',      
    encode_categ = 'auto' 
)

help(AutoClean)  

df_clean = clean_pipeline.output  


df_clean.head()  





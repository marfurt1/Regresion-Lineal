from utils import db_connect
# your code here
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
#!pip install seaborn
import seaborn as sns
import seaborn as sb
#!pip install plotly
import plotly.graph_objects as go
#!pip install folium
#!pip install statsmodels
import folium
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from folium.plugins import MarkerCluster
from folium import plugins
from folium.plugins import FastMarkerCluster
from folium.plugins import HeatMap
import plotly.express as px
import pickle
import statsmodels.api as sm

#load data
url = 'https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv'
df_raw=pd.read_csv(url)


pd.get_dummies(df_raw,drop_first=True)

# drop duplicates
df_raw=df_raw.drop_duplicates().reset_index(drop= True)


#loaded_model = pickle.load(open(filename, 'rb'))

modelo = pickle.load(open("../models/finalized_model.sav", 'rb'))

#Predict using the model whith new data

print('Predicted prima : \n', modelo.predict([[40,1,22,1,1,1,0,0]]))
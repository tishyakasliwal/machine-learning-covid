import numpy as np 
import pandas as pd 
from pandas import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose 



def parser(x):
    return datetime.strptime(x, "%Y-%m-%d")
    
covid = pd.read_csv('time_series_covid_19_confirmed india.csv', 
                       index_col ='Date', 
                       parse_dates = [0], date_parser = parser)



result = seasonal_decompose(covid['Cases'],  
                            model ='multiplicative')
result.plot()



from pmdarima import auto_arima 
  
# Ignore harmless warnings 
import warnings 
warnings.filterwarnings("ignore") 

train = covid.iloc[:len(covid)-30] 
test = covid.iloc[len(covid)-30:]

model = auto_arima(train['Cases'], trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
result = model.fit(train['Cases']) 
result.summary()







from statsmodels.tsa.statespace.sarimax import SARIMAX
  
model = SARIMAX(train['Cases'],  
                order = (2, 2, 3),  
                seasonal_order =(0, 0, 0, 0)) 
  
result = model.fit() 
result.summary() 





start = len(train) 
end = len(train) + len(test) - 1
  

predictions = result.predict(start, end, 
                             typ = 'levels').rename("Predictions") 
  
# plot predictions and actual values 
predictions.plot(legend = True) 
test['Cases'].plot(legend = True)

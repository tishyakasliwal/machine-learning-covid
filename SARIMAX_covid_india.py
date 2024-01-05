import pandas as pd
from pandas import datetime
def parser(x):
    return datetime.strptime(x, "%Y-%m-%d")
 

data = pd.read_csv('time_series_covid_19_confirmed india.csv', 
                       index_col ='Date', 
                       parse_dates = [0], date_parser = parser)



data.columns = ['Cases', 'Date', 'time_status',"testing_capacity"]


import pmdarima as pm

train = data.iloc[:len(data)-30] 
test = data.iloc[len(data)-30:]

# SARIMAX Model
sxmodel = pm.auto_arima(train[['Cases']], exogenous=train[['time_status']],
                           start_p=1, start_q=1,
                           test='adf',
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=None, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

sxmodel.summary()

from statsmodels.tsa.statespace.sarimax import SARIMAX 
  
model = SARIMAX(train['Cases'], order = (3,1,0),seasonal_order =(0,1,1,12),exogenous=train['time_status']) 
  
result = model.fit() 
result.summary()



start = len(train) 
end = len(train) + len(test) - 1
  

predictions = result.predict(start, end, 
                             typ = 'levels').rename("Predictions") 
  
# plot predictions and actual values 
predictions.plot(legend = True) 
test['Cases'].plot(legend = True)



import numpy as np 

test['Cases'], predictions = np.array(test['Cases']), np.array(predictions)
x= np.mean(np.abs((test['Cases'] - predictions) /test['Cases'])) * 100
print(x)

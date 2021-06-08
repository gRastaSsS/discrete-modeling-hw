import pandas as pd

pd.set_option('display.max_columns', 9)

df = pd.read_csv('data.csv')
df_survival_rate = pd.read_csv('survival_rate.csv')
df_m = df[df['Sex'] == 'Male'].drop(labels=["Sex"], axis=1).set_index('Date')
df_f = df[df['Sex'] == 'Female'].drop(labels=["Sex"], axis=1).set_index('Date')
df_joined = df_m.join(df_f, on="Date", how='inner',
                      lsuffix='_m', rsuffix='_f')

df_prediction = pd.read_csv('prediction.csv')
df_prediction_m = df_prediction[df_prediction['Sex'] == 'Male'].drop(labels=["Sex", "Year"], axis=1)
df_prediction_f = df_prediction[df_prediction['Sex'] == 'Female'].drop(labels=["Sex", "Year"],
                                                                       axis=1)




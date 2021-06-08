import numpy as np
import matplotlib.pyplot as plt
from BaseModel import BaseModel
from load_data import df_survival_rate, df_joined, df_f, df_m, df_prediction_m, df_prediction_f
from utils import calculate_fertility_rate, calculate_gender_rate


def estimate_survival_rate(sex):
    d = df_survival_rate[df_survival_rate["Sex"] == sex]
    d = d.drop(labels=["Sex", "Date"], axis=1)
    result = []

    for col_name in d:
        col = d[col_name]
        col = col[col != "#VALUE!"]
        col = col.astype(float)
        result.append(col[-9:].mean())

    return result


def estimate_total_fertility_rate(last_n_periods, use_median=False):
    if use_median:
        return calculate_fertility_rate(df_joined)[-last_n_periods:].median()
    else:
        return calculate_fertility_rate(df_joined)[-last_n_periods:].mean()


def estimate_gender_rate():
    return calculate_gender_rate(df_joined)[-9:].mean()


survival_rate = np.array([
    estimate_survival_rate('Male'),
    estimate_survival_rate('Female')
])

fertility_rate = estimate_total_fertility_rate(4)
gender_rate = estimate_gender_rate()
initial_state = np.array([
    df_m.to_numpy()[-1:][0].astype(dtype=float),
    df_f.to_numpy()[-1:][0].astype(dtype=float)
])

base_model = BaseModel(survival_rate, fertility_rate, gender_rate)

prediction = base_model.eval(initial_state, 9)

age_ranges = [
    "0 - 4", "5 - 9", "10 - 14", "15 - 19",
    "20 - 24", "25 - 29", "30 - 34", "35 - 39",
    "40 - 44", "45 - 49", "50 - 54", "55 - 59",
    "60 - 64", "65 - 69", "70 - 74", "75 - 79",
    "80 - 84"
]

data_years = [1950, 1955, 1960, 1965, 1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005]
pred_years = [2010, 2015, 2020, 2025, 2030, 2035, 2040, 2045, 2050]

range_r_pred_m, range_r_pred_f = dict(), dict()
range_prediction_m, range_prediction_f = dict(), dict()

for column in df_m:
    range_r_pred_m[column] = df_prediction_m[column].to_numpy()
    range_r_pred_f[column] = df_prediction_f[column].to_numpy()

for pred in prediction:
    pred_m, pred_f = pred

    for i, age_range in enumerate(age_ranges):
        if age_range not in range_prediction_m:
            range_prediction_m[age_range] = []
            range_prediction_f[age_range] = []

        range_prediction_m[age_range].append(pred_m[i])
        range_prediction_f[age_range].append(pred_f[i])

max_ranges_per_figure = 2
counter = 0

fig = plt.figure()
fig.set_tight_layout(True)
i = 1
j = 0

for sex in ['Male', 'Female']:
    if sex == 'Male':
        est_pred = range_prediction_m
        un_pred = range_r_pred_m
    else:
        est_pred = range_prediction_f
        un_pred = range_r_pred_f

    for age_range in age_ranges:
        ax = plt.subplot(3, 2, i)
        ax.set_title(f'{sex}: {age_range}')
        plt.plot(pred_years, est_pred[age_range], color='r')
        plt.plot(pred_years, un_pred[age_range], color='g')
        if i % 6 == 0:
            i = 1
            j += 1
            plt.autoscale()
            plt.savefig(f'./task-1/pl-{j}.png')
            fig = plt.figure()
            fig.set_tight_layout(True)
        else:
            i += 1

j += 1
plt.autoscale()
plt.savefig(f'./task-1/pl-{j}.png')

flg = plt.figure()
fig.set_tight_layout(True)

for i, sex in enumerate(['Male', 'Female']):
    if sex == 'Male':
        est_pred = range_prediction_m
        un_pred = range_r_pred_m
    else:
        est_pred = range_prediction_f
        un_pred = range_r_pred_f

    reduced_est = np.zeros(len(pred_years))
    for per_age_group in est_pred.values():
        for year in range(len(per_age_group)):
            reduced_est[year] += per_age_group[year]

    reduced_un = np.zeros(len(pred_years))
    for per_age_group in un_pred.values():
        for year in range(len(per_age_group)):
            reduced_un[year] += per_age_group[year]

    ax = plt.subplot(2, 1, i + 1)
    ax.set_title(f'{sex}: All')
    plt.plot(pred_years, reduced_est, color='r')
    plt.plot(pred_years, reduced_un, color='g')

plt.autoscale()
plt.savefig('./task-1/pl-all.png')
plt.close()

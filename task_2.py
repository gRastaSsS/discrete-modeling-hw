from collections import OrderedDict

import numpy as np
from SALib.analyze import sobol
from SALib.sample import saltelli
from load_data import df_survival_rate, df_joined
from BaseModel import BaseModel
from utils import calculate_fertility_rate, calculate_gender_rate
import seaborn as sns
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, threshold=np.inf)

age_ranges = [
    "0 - 4", "5 - 9", "10 - 14", "15 - 19",
    "20 - 24", "25 - 29", "30 - 34", "35 - 39",
    "40 - 44", "45 - 49", "50 - 54", "55 - 59",
    "60 - 64", "65 - 69", "70 - 74", "75 - 79",
    "80 - 84", "85 - 89", "90 - 94", "95 - 99",
    "100+"
]

initial_state = [
    np.array([3.708674e+03, 3.366827e+03, 4.129467e+03, 6.102203e+03, 6.204210e+03,
              5.448342e+03, 5.057844e+03, 4.607838e+03, 5.418435e+03, 5.653337e+03,
              4.847216e+03, 3.605566e+03, 1.929557e+03, 2.830726e+03, 1.617531e+03,
              1.287721e+03, 4.094630e+02, 1.420950e+02, 6.528400e+01, 1.325300e+01,
              1.784000e+00]),
    np.array([3.516752e+03, 3.212522e+03, 3.951828e+03, 5.901660e+03, 6.095060e+03,
              5.488430e+03, 5.133842e+03, 4.775562e+03, 5.812684e+03, 6.329395e+03,
              5.766926e+03, 4.592200e+03, 2.828547e+03, 4.671970e+03, 3.082444e+03,
              3.060767e+03, 1.552632e+03, 6.073560e+02, 3.064170e+02, 6.079000e+01,
              6.415000e+00])
]


def calculate_boundaries():
    bounds = dict()

    df_surv = df_survival_rate.replace({
        'FROM 75 - 79 TO 80 - 84': {'#VALUE!': 0},
        'FROM 80 - 84 TO 85 - 89': {'#VALUE!': 0},
        'FROM 85 - 89 TO 90 - 94': {'#VALUE!': 0},
        'FROM 90 - 94 TO 95 - 99': {'#VALUE!': 0},
        'FROM 95 - 99 TO 100+': {'#VALUE!': 0}
    })

    for group in range(len(age_ranges) - 1):
        df_loc = df_surv[df_surv['Sex'] == 'Male']
        survival_rate = df_loc[f'FROM {age_ranges[group]} TO {age_ranges[group + 1]}'].astype(
            float)
        min_rate = survival_rate.min()
        max_rate = survival_rate.max()
        bounds[f'FROM {age_ranges[group]} TO {age_ranges[group + 1]} M'] = [min_rate, max_rate]

    for group in range(len(age_ranges) - 1):
        df_loc = df_surv[df_surv['Sex'] == 'Female']
        survival_rate = df_loc[f'FROM {age_ranges[group]} TO {age_ranges[group + 1]}'].astype(
            float)
        min_rate = survival_rate.min()
        max_rate = survival_rate.max()
        bounds[f'FROM {age_ranges[group]} TO {age_ranges[group + 1]} F'] = [min_rate, max_rate]

    fertility_rate = calculate_fertility_rate(df_joined)
    bounds['FERTILITY_RATE'] = [fertility_rate.min(), fertility_rate.max()]
    gender_rate = calculate_gender_rate(df_joined)
    bounds['GENDER_RATE'] = [gender_rate.min(), gender_rate.max()]
    return bounds


def prepare_problem(param_boundaries, exclude_params=None):
    if exclude_params is None:
        exclude_params = {}

    names = []
    bounds = []
    groups = []

    for group in range(len(age_ranges) - 1):
        name = f'FROM {age_ranges[group]} TO {age_ranges[group + 1]} M'

        if name in exclude_params:
            continue

        names.append(name)
        bounds.append(param_boundaries[name])
        groups.append('Survival_rate_m')

    for group in range(len(age_ranges) - 1):
        name = f'FROM {age_ranges[group]} TO {age_ranges[group + 1]} F'

        if name in exclude_params:
            continue

        names.append(name)
        bounds.append(param_boundaries[name])

        if group in [4, 5, 6, 7]:
            groups.append('Survival_rate_fertile_f')
        elif group in [0]:
            groups.append('Survival_rate_newborn_f')
        else:
            groups.append('Survival_rate_other_f')

    if 'FERTILITY_RATE' not in exclude_params:
        names.append('FERTILITY_RATE')
        bounds.append(param_boundaries['FERTILITY_RATE'])
        groups.append('Fertility_rate')

    if 'GENDER_RATE' not in exclude_params:
        names.append('GENDER_RATE')
        bounds.append(param_boundaries['GENDER_RATE'])
        groups.append('Gender_rate')

    return {
        'num_vars': len(names),
        'names': names,
        'groups': groups,
        'bounds': bounds
    }


def evaluate(params_mtx, year):
    period = int((year - 2010) / 5) + 1

    Y = []
    for params in params_mtx:
        survival_rate = [
            params[0:20],
            params[20:40]
        ]
        fertility_rate = params[40]
        gender_rate = params[41]
        model = BaseModel(survival_rate, fertility_rate, gender_rate)
        result = np.array(model.eval(initial_state, period)[-1])
        result = result.sum()
        Y.append(result)

    return np.array(Y)


boundaries = calculate_boundaries()
problem = prepare_problem(boundaries)
param_values = saltelli.sample(problem, 2000)

Y = evaluate(param_values, 2030)

output_graphs = True

if output_graphs:
    for param_index, name in enumerate(problem['names']):
        plt.figure()

        bins = 30
        X = param_values[:, param_index]

        left_b, right_b = boundaries[name]
        hist = np.digitize(X, np.linspace(left_b, right_b, bins))

        true_X = []
        true_Y = []

        for index, bucket in enumerate(hist):
            true_X.append((right_b - left_b) / bins * bucket + left_b)
            true_Y.append(Y[index])

        ax = sns.lineplot(x=true_X, y=true_Y)
        plt.title(name)
        plt.savefig(f'./task-2/confidence_graph_{name}.png')


head_printed = False
for year in [2020, 2030, 2060, 2110]:
    s = sobol.analyze(problem, evaluate(param_values, year))

    if not head_printed:
        unique_group_names = list(OrderedDict.fromkeys(s.problem['groups']))
        print(unique_group_names)
        head_printed = True

    print(s['S1'])

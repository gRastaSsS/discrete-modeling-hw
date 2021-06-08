import numpy as np


class BaseModel:
    def __init__(self, survival_rate, fertility_rate, gender_rate):
        self.survival_rate = survival_rate
        self.fertility_rate = fertility_rate
        self.gender_rate = gender_rate

    def eval(self, initial_state, n):
        result = []
        current_population_m = initial_state[0]
        current_population_f = initial_state[1]

        for year in range(n):
            survived_population_m = current_population_m[:-1] * self.survival_rate[0]
            next_population_m = np.concatenate(([0], survived_population_m))

            survived_population_f = current_population_f[:-1] * self.survival_rate[1]
            next_population_f = np.concatenate(([0], survived_population_f))

            newborn_total = current_population_f[4:8].sum() * self.fertility_rate

            estimated_gender_rate = self.gender_rate
            newborn_m = newborn_total / (estimated_gender_rate + 1)
            newborn_f = newborn_total - newborn_m

            next_population_m[0] = newborn_m
            next_population_f[0] = newborn_f

            current_population_m = next_population_m
            current_population_f = next_population_f

            result.append((current_population_m, current_population_f))

        return result

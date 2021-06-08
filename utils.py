def calculate_fertility_rate(df):
    babies = df['0 - 4_m'] + df['0 - 4_f']
    women = df['20 - 24_f'] + df['25 - 29_f'] + df['30 - 34_f'] + df[
        '35 - 39_f']
    return babies / women


def calculate_gender_rate(df):
    return df.apply(lambda row: row['0 - 4_f'] / row['0 - 4_m'], axis=1)

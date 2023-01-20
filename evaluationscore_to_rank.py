import pandas as pd
# converting evaluation scores to rank for each evaluation metric and for each task
df = pd.read_csv('Results.csv')
i = 0
while i < 480:
    df_rows = df.iloc[i:i+12]
    i = i + 12
    print(i)
    df_rows['rank'] = df_rows['score'].rank(ascending=False)
    df_rows.sort_values(["score", "model"], axis=0, ascending=False, inplace=True, na_position='first')
    df_rows.to_csv('Ranking_Results.csv', sep=',', float_format='%.6f', mode = 'a', header=True,
          columns=['task_id', 'evaluation_metric', 'model', 'rank'], index=False)


import pandas as pd

df = pd.read_json('data.json')
print(df.to_string())



new_df = df.dropna()
print(new_df.to_string())


corr = df.corr()
print("This is the correlation table\n", corr)

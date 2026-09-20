# this is me trying to rewrite the code that is going to be used to generate the
# candlestick graph from the pandas file
# if it is possible generate a csv also for the token graphs paired to the token generated


# these are the first modules that are going to be used


# basically this is the candlestick graph

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pathlib 


btc_file_path = r"C:\Users\ebnyk\OneDrive\Desktop\Machine_learning_for_big_data_Analytics_assignments\machine learning for big data analytics final exam papers\BTCUSDT_1h_2020_to_today.csv"
df = pd.read_csv(btc_file_path)
window_info=[]
for n in range(300):
    row = df.iloc[n]
    row_stats = {
        'q1': min((row['open'],row['close'])),
        'q3': max((row['open'],row['close'])),
        'whislo': row['low'],
        'whishi': row['high'],
        'med' : (row['open']+row['close']) / 2
        }
    
    window_info.append(row_stats)

# currently just function to generate the graph of a particular window say first 30 days

fig, ax = plt.subplots()
boxplot = ax.bxp(
    patch_artist=True,
    showfliers=False,
    bxpstats=window_info,
    showcaps=False
    )

for i, box in enumerate(boxplot['boxes']):
    row = df.iloc[i]
    if row['open'] > row['close']:
        box.set_facecolor('#EF5350')
        box.set_edgecolor('#EF5350')
    else:
        box.set_facecolor('#26A69A')
        box.set_edgecolor('#26A69A')

for med in boxplot['medians']:
    med.set_visible(False)


for i, whisker in enumerate(boxplot['whiskers']):
    candle_index = i // 2
    row = df.iloc[candle_index]
    if row['open'] > row['close']:
        whisker.set_color('#EF5350')
    else:
        whisker.set_color('#26A69A')
        
plt.axis("off")
plt.savefig(str(pathlib.Path.home()/"OneDrive/Desktop/candle1.png"))
plt.show()




































































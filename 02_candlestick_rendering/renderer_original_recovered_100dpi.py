# this is the reconstructed candlestick graph generation source code
# together with the generation of the full dataset

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pathlib 


csv_path = r'C:/Users/ebnyk/OneDrive/Desktop/Machine_learning_for_big_data_Analytics_assignments/machine learning for big data analytics final exam papers/BTCUSDT_1h_2020_to_today.csv'
df = pd.read_csv(csv_path)

# this part is me trying with 4 candles



def generate_kline(dataframe,
                   start_index,
                   stop_index=29,
                   show_image=True,
                   save_image=None,
                   ):

    IMG_SIZE_PX = 360
    IMG_DPI = 100

    df = dataframe
    fig, axs = plt.subplots(figsize=(IMG_SIZE_PX / IMG_DPI, IMG_SIZE_PX / IMG_DPI), dpi=IMG_DPI)
    
    info_list = []
    colour_reader = []
    for n in range(start_index, stop_index+1):
        row = df.iloc[n]
        row_info = {
            'q1':row['open'] if row['open'] < row['close'] else row['close'],
            'q3':row['close'] if row["close"] >= row['open'] else row['open'],
            'whishi':max((row['high'], row['low'])),
            'whislo':min((row['high'], row['low'])),
            'med': (row['open'] + row['close']) / 2
            }
        if row['open'] <= row['close']:
            colour_reader.append('green')
        else:
            colour_reader.append('red')
        info_list.append(row_info)

    boxplots = axs.bxp(
        bxpstats=info_list,
        showcaps=False,
        patch_artist=True,
        showfliers=False
        )

    i = 0
    for box in boxplots['boxes']:
        box.set_facecolor(colour_reader[i])
        box.set_edgecolor(colour_reader[i])
        i += 1
        
    for n, whisker in enumerate(boxplots["whiskers"]):
        candle_index = n // 2
        whisker.set_color(colour_reader[candle_index])

    for median in boxplots["medians"]:
        median.set_visible(False)


    plt.axis('off')
    
    if save_image is not None:
        plt.savefig(save_image, dpi=IMG_DPI, pad_inches=0)

    if show_image:
        plt.show()
        
    plt.close(fig)

if __name__ == '__main__':
    generate_kline(dataframe=df,
        start_index=100,
        stop_index=129,
        show_image=False,
        save_image=pathlib.Path.home()/"OneDrive/Desktop/sample_image.jpg",
        )











































































































































































































# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import drive
import csv
import os

drive.mount('/content/drive')

def run_analysis(title, col_labels, row_labels, col_val, row_val, sim_count, start_bal, analysis_func):
  temp = np.zeros((len(row_labels), len(col_labels)))
  df_out = pd.DataFrame(temp, index=row_labels, columns=col_labels)

  for col_label in col_val:
    df = pd.read_csv("/content/drive/MyDrive/Roulette/data/data_" + title + str(col_label) + "_percent.csv")
    for row_label in row_val:
      data = np.asarray([row['end balance'] for _, row in df.iterrows() if row['starting bet count'] == row_label])

      df_out.at[str(row_label) + " bets", str(col_label) + " percent"] = analysis_func(data, sim_count, start_bal)
  return df_out

def construct_df(data, title, col_labels, row_labels, col_val, row_val, sim_count, start_bal, analysis_func):
  temp = np.zeros((len(row_labels), len(col_labels)))
  df_out = pd.DataFrame(temp, index=row_labels, columns=col_labels)

  for col_label in col_val:
    df = pd.read_csv("/content/drive/MyDrive/Roulette/data/data_" + title + str(col_label) + "_percent.csv")
    for row_label in row_val:
      data = np.asarray([row['end balance'] for _, row in df.iterrows() if row['starting bet count'] == row_label])

      df_out.at[str(row_label) + " bets", str(col_label) + " percent"] = analysis_func(data, sim_count, start_bal)
  return df_out

def load_data_row(file, row_val):
  data = []
  with open(file, newline='') as csvfile:
    data = list(csv.reader(csvfile))
  # remove column labels
  data = data[1:]
  # sort and convert to numpy arrays
  data = np.asarray([np.asarray([float(row[1]) for row in data if row[0] == str(bet_count)], dtype=object) for bet_count in row_val], dtype=object)
  return data

def load_data(title, col_val, row_val):
  data = np.asarray([load_data_row("/content/drive/MyDrive/Roulette/data/data_" + title + str(col) + "_percent.csv", row_val) for col in col_val], dtype=object)
  return data

# modify these
col_val = [num for num in range(5,55,5)]
row_val = [num for num in range(2,11)]
col_labels = [str(percent) + " percent" for percent in col_val]
row_labels = [str(bets) + " bets" for bets in row_val]
norm_row_val = [6,8,14]
norm_row_labels = [str(bets) + " bets" for bets in norm_row_val]
sim_count = 1000
bal = 1000

roi_percent_func = lambda data, sim_count, start_bal: sum(data) / (sim_count * start_bal)
win_percent_func = lambda data, sim_count, start_bal: len([item for item in data if item > start_bal]) / sim_count

def calc_func_data(data, col_val, row_val, analysis_func, sim_count, start_bal):
  data_out = np.zeros((len(row_val), len(col_val)))
  for j, percent_block in enumerate(data):
    for i, seq in enumerate(percent_block):
      data_out[i, j] = analysis_func(seq, sim_count, start_bal)
  return data_out

gen_data = load_data("", col_val, row_val)
norm_data = load_data("norm_", col_val, norm_row_val)
pascal_data = load_data("pascal_", col_val, row_val)

norm_data_win = calc_func_data(norm_data, col_val, norm_row_val, win_percent_func, sim_count, bal)
gen_data_win = calc_func_data(gen_data, col_val, row_val, win_percent_func, sim_count, bal)
pascal_data_win = calc_func_data(pascal_data, col_val, row_val, win_percent_func, sim_count, bal)

norm_data_roi = calc_func_data(norm_data, col_val, norm_row_val, roi_percent_func, sim_count, bal)
gen_data_roi = calc_func_data(gen_data, col_val, row_val, roi_percent_func, sim_count, bal)
pascal_data_roi = calc_func_data(pascal_data, col_val, row_val, roi_percent_func, sim_count, bal)

fig = plt.figure()
ax = plt.subplot(111)
# let the y axis be the metric (profitability/win percent) let x be the number of bets made
# let data sets be the percentage of desired winnings
for i in range(0, len(norm_data_roi)):
  plt.plot(col_val, norm_data_roi[i], marker='o',label=str(norm_row_labels[i]))
ax.set_xlabel("Percentage to Earn")
ax.set_ylabel("Overall Earnings")
ax.set_title("Norm Data ROI")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

fig = plt.figure()
ax = plt.subplot(111)
# let the y axis be the metric (profitability/win percent) let x be the number of bets made
# let data sets be the percentage of desired winnings
for i in range(0, len(pascal_data_roi)):
  ax.plot(col_val, pascal_data_roi[i], marker='o',label=str(row_labels[i]))
ax.set_xlabel("Percentage to Earn")
ax.set_ylabel("Overall Earnings")
ax.set_title("Pascal Data ROI")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

fig = plt.figure()
ax = plt.subplot(111)
# let the y axis be the metric (profitability/win percent) let x be the number of bets made
# let data sets be the percentage of desired winnings
for i in range(0, len(gen_data_roi)):
  ax.plot(col_val, gen_data_roi[i], marker='o',label=str(row_labels[i]))
ax.set_xlabel("Percentage to Earn")
ax.set_ylabel("Overall Earnings")
ax.set_title("Gen. Data ROI")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

fig = plt.figure()
ax = plt.subplot(111)
# let the y axis be the metric (profitability/win percent) let x be the number of bets made
# let data sets be the percentage of desired winnings
for i in range(0, len(norm_data_win)):
  plt.plot(col_val, norm_data_win[i], marker='o',label=str(norm_row_labels[i]))
ax.set_xlabel("Percentage to Earn")
ax.set_ylabel("Overall Earnings")
ax.set_title("Norm Data Win Rate")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

fig = plt.figure()
ax = plt.subplot(111)
# let the y axis be the metric (profitability/win percent) let x be the number of bets made
# let data sets be the percentage of desired winnings
for i in range(0, len(pascal_data_win)):
  ax.plot(col_val, pascal_data_win[i], marker='o',label=str(row_labels[i]))
ax.set_xlabel("Percentage to Earn")
ax.set_ylabel("Overall Earnings")
ax.set_title("Pascal Data Win Rate")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

fig = plt.figure()
ax = plt.subplot(111)
# let the y axis be the metric (profitability/win percent) let x be the number of bets made
# let data sets be the percentage of desired winnings
for i in range(0, len(gen_data_win)):
  ax.plot(col_val, gen_data_win[i], marker='o',label=str(row_labels[i]))
ax.set_xlabel("Percentage to Earn")
ax.set_ylabel("Overall Earnings")
ax.set_title("Gen. Data Win Rate")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid()

"""# Running Profit Pecentage"""

norm_roi_df = run_analysis("norm_", col_labels, norm_row_labels, col_val, norm_row_val, sim_count, bal, roi_percent_func)
norm_roi_df

pascal_roi_df = run_analysis("pascal_", col_labels, row_labels, col_val, row_val, sim_count, bal, roi_percent_func)
pascal_roi_df

gen_roi_df = run_analysis("", col_labels, row_labels, col_val, row_val, sim_count, bal, roi_percent_func)
gen_roi_df

custom_roi_df = run_analysis("custom_", ["10 percent"], ["5 bets", "7 bets"], [10], [5,7], sim_count, bal, roi_percent_func)
custom_roi_df

"""# Running Win Percentage"""

norm_win_df = run_analysis("norm_", col_labels, norm_row_labels, col_val, norm_row_val, sim_count, bal, win_percent_func)
norm_win_df

pascal_win_df = run_analysis("pascal_", col_labels, row_labels, col_val, row_val, sim_count, bal, win_percent_func)
pascal_win_df

gen_win_df = run_analysis("", col_labels, row_labels, col_val, row_val, sim_count, bal, win_percent_func)
gen_win_df

custom_win_df = run_analysis("custom_", ["10 percent"], ["5 bets", "7 bets"], [10], [5,7], sim_count, bal, win_percent_func)
custom_win_df

"""The expected value of a game of roulette can be found by taking the total amount to win and multiplying it by the chance of winning and subtracting the total amount to lose multiplied by the chance of losing.

Here X is a random variable taking the values (1000 * (1 + amount to win)) for a win and -1000 for a loss

E[X] = *total to win* * (*Probability Win*) - *total to lose* * (1 - *Probability Win*)

Therefore a bet is profitable if E[X] > 0
We can then find the values that required win rate for each percentage to win value (5%-50%)

"""

roi = [round(x / 100, 2) for x in range(5,55,5)]
win_rate = [round(1.0  - (x / 100),2) for x in range(0,100)]

min_val_dict = {}
for r in roi:
    min_rate = 1 # initially set to 100%
    for win in win_rate:
        if 1000 * (1 + r) * (win) - 1000 * (1 - win) > 1000:
            if win < min_rate:
                min_rate = win
    key = str(int(r * 100)) + " percent"
    min_val_dict[key] = min_rate
print(min_val_dict)

"""Look at the values that are close to, at, or greater than our necessary minimum win rate for each desired profitability"""

x_mod = 0.93 # finds values that are no less than 7% of the desired amount (trying to find the closest values in each column, could be any percent)

for col in col_labels:
  print(col)
  print(pascal_win_df[(pascal_win_df[col] > (min_val_dict[col] * x_mod))][col])

for col in col_labels:
  print(col)
  print(gen_win_df[(gen_win_df[col] > (min_val_dict[col] * x_mod))][col])

for col in col_labels:
  print(col)
  print(norm_win_df[(norm_win_df[col] > (min_val_dict[col] * x_mod))][col])



"""Given that betting sequences starting with 4 bets are the closest in the gen_win_df and pascal_win_df, we will try and raise the win percentage for this sequence (when trying to earn 5%)

Additionally, we will also try bet sequences of 5 since this is also close in sequence length and in win percentage for both general and pascal style bet sequences.

For a desired profit of 5%, we need a win rate of 0.98.
Both 4 and 5 bet sequences are 0.054-0.07 off of this value
"""
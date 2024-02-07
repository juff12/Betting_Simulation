import numpy as np
import pandas as pd
import csv
import random

# create a csv file for a given arrray of data
def create_csv(title, data):
  file_name = "data/data_" + title + ".csv"
  with open(file_name, 'w', newline='') as file:
    writer = csv.writer(file)
    field = ['starting bet count', 'end balance']
    writer.writerow(field)
    for row in data:
      writer.writerow(row)

def gamble_sim(sequence, balance):
    """Labouchère betting."""
    # Won
    if len(sequence) < 1:
        return balance

    # If the sequence is of length 1, the bet is the number in the sequence.
    # Otherwise, it is the first number added to the last number.
    if len(sequence) == 1:
        bet = sequence[0]
    else:
        bet = sequence[0] + sequence[-1]

    # Lost the entire round
    if bet > balance:
        return balance

    won = random.randint(0,37)


    # let 0-17 be red, 18-35 be black and 36 and 37 be green
    # value is 1-18 then win
    if won < 18:
      return gamble_sim(sequence[1:-1], balance + bet)
    else:
        # Lost bet
        return gamble_sim(sequence + [bet], balance - bet)

def func_pascal(n):
    build_row = lambda line_num: np.asarray([binomialCoeff(line_num, i) for i in range(0, line_num + 1)])
    #arr = np.asarray([build_row(line_num) for line_num in range(0, n)], dtype=object)
    if n <= 0:
        return np.array([])
    return build_row(n-1)
def binomialCoeff(n, k):
    res = 1
    if (k > n - k):
        k = n - k
    for i in range(0 , k):
        res = res * (n - i)
        res = res // (i + 1)
    return res

def gamble_sim_run(bal, sim_count, row_labels, col_labels, title, pascal):
    # function to find size of each bet
    bet_size = lambda x, y: (bal * x) * y
    
    # function to create array of each bet repeated in sequence
    # ex: want to win 50 dollars, 5 inital bets would be [10,10,10,10,10]
    bet_array = lambda col_label, row_label: np.asarray([bet_size(col_label, (1 / row_label))
                                                         for _ in range(0,row_label)], dtype=object)
    # builds an array for functions with given data
    data_bet_array = lambda col_label, pascal_arr: np.asarray([bet_size(col_label, data_elem / sum(pascal_arr))
                                                               for data_elem in pascal_arr], dtype=object)
    # decides which which way to build the array
    func_decider = lambda col_label, row_label, pascal: data_bet_array(col_label, 
                                                                       func_pascal(row_label)) if pascal else bet_array(col_label, row_label)
    # function to create a row of a TOTAL bet amount split into intervals of 2-10 inclusive
    row_array = lambda row_label, pascal: np.asarray([func_decider(col_label, row_label, pascal)
                                                      for col_label in col_labels], dtype=object)
    # function to pad the rows with zeros so that each bet array has the same number of zeros as the max number of bets (10)
    pad_row = lambda row_label, pascal: np.asarray([np.pad(arr, (0, max(row_labels)-row_label))
                                                    for arr in row_array(row_label, pascal)], dtype=object)
    # sequence array seq_array[i] returns a (9x10) array of bets with bet counts sized from 2-10 (inclusive) for the i'th percentage of balance bet
    # ex seq_array[0] contains the bets that sum to a win of $50 (ie. 1000 * 0.05) spread over 2,3,4,5,6,7,8,9,10 bets. (there is an seperate arraay for each number of bets)
    seq_array = np.stack(np.asarray([pad_row(row_label, pascal)
                                     for row_label in row_labels], dtype=object), axis=1)

    create_data(bal, sim_count, row_labels, col_labels, seq_array, title)



def create_data(bal, sim_count, row_labels, col_labels, seq_array, title):
    # runs simulation for percent data
    for index, percent_block in enumerate(seq_array):
        # runs the gamble_sim function 1000 times for each bet sequence saves the results to an array of 9000x2
        # 9000x2 array has a column for the number of bets to start and the ending balance
        gamble_func = lambda p_block: [np.asarray([row_labels[i], gamble_sim(list(p_block[i]), bal)], dtype=object)
                                       for _ in range(0,sim_count) for i in range(0,len(row_labels))]
        # stores and converts returned value from the gambling helper function
        data_percent = np.asarray(gamble_func(percent_block), dtype=object)

        create_csv(title + str(int(col_labels[index] * 100)) + "_percent", data_percent)

# normalizes a custom set of data for simulation
def normalize_data(data, bal, percent_array):
    max_len_row = max([len(row) for row in data])
    bet_size = lambda x, y: (bal * x) * y
    row_func = lambda row, percent: np.asarray([bet_size(percent, item) for item in row], dtype=object)
    pad_func = lambda data, percent: np.asarray([np.pad(row_func(row, percent),
                                                        (0, max_len_row-len(row))) for row in data], dtype=object)    
    return np.asarray([pad_func(data, percent) for percent in percent_array])



# percentage to win, what percent of current balance the player wishes to win
# 5% to 50% in increments of 5%
# initial testing showed that a higher percent desired to win, relative to the current balance had
# a lower chance of being profitable (decided to cap at 50% and a 0% return is not valid or of interest)
percent_to_win = np.linspace(0.05,0.5,10)


# bets from 2-10(inclusive)
# list must have 2 elements to function at start
num_bets = np.arange(2,11,1)

#    pascal_bet = lambda x, y: (bal * x) * y
#    build_func = lambda percent: np.asarray([np.asarray([pascal_bet(percent, item) for item in pascal_arr[i]], dtype=object) for i in range(0, len(pascal_arr))], dtype=object)
#    pascal_arr = np.array([build_func(percent) for percent in col_labels])

norm_data = [[0.022, 0.136, 0.341, 0.341, 0.136, 0.022], [0.001, 0.021, 0.136, 0.341, 0.341, 0.136, 0.021, 0.001],
             [0.001, 0.005, 0.017, 0.044, 0.092, 0.15, 0.191, 0.191, 0.15, 0.092, 0.044, 0.017, 0.005, 0.001]]
norm_row_labels = [len(row) for row in norm_data]
norm_data = normalize_data(norm_data, 1000, percent_to_win)


custom_data = [[0.10,0.20,0.40,0.20,0.10], [0.05,0.10,0.15,0.40,0.15,0.10,0.05]]
custom_row_labels = [len(row) for row in custom_data]
custom_data = normalize_data(custom_data, 1000, np.array([0.1]))
gamble_sim_run(1000,1000,num_bets, percent_to_win, "", pascal=False)
gamble_sim_run(1000,1000,num_bets,percent_to_win,"pascal_",pascal=True)
create_data(1000,1000,norm_row_labels,percent_to_win,norm_data,"norm_")
create_data(1000,1000,custom_row_labels,np.array([0.10]),custom_data,"custom_")
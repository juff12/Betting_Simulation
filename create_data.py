import numpy as np
import pandas as pd
import csv
import random

def create_csv(title, data):
    """Creates a csv file in the data folder of the current directory of the
    completed simulation data

    Args:
        title (string): the name of the csv file
        data (ndarray): the completed simulation data
    """    
    file_name = "data/data_" + title + ".csv"
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        field = ['starting bet count', 'end balance']
        writer.writerow(field)
        for row in data:
            writer.writerow(row)

def gamble_sim(sequence, balance):
    """Labouchère betting simulation for a given sequence of starting bets and a starting
    balance. Runs until there are no values left in the sequence, balance hits zero, or no
    more bets can be made

    Args:
        sequence (list): a python list of the starting bets
        balance (int): the starting balance in dollars

    Returns:
        float: The ending balance after simulation finishes
    """        
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
    """Returns the desired row from pascals triangle as an array
    ex: n=5, -> [1,4,6,4,1]

    Args:
        n (int): the row number to return from pascals triangle

    Returns:
        ndarray: the desired row in pascals triangle as an array
    """    
    build_row = lambda line_num: np.asarray([binomialCoeff(line_num, i)
                                             for i in range(0, line_num + 1)])
    if n <= 0:
        return np.array([])
    return build_row(n-1)

def binomialCoeff(n, k):
    """Binomial Coefficial for pascal trianlge (helper function)

    Args:
        n (int): total number of choices
        k (int): how many to choose

    Returns:
        int: The binomial coefficient for the current term in a binomial expansion
    """    
    res = 1
    if (k > n - k):
        k = n - k
    for i in range(0 , k):
        res = res * (n - i)
        res = res // (i + 1)
    return res

# runs that gambling simulation for a given number of simulations with a given
# starting balance
def gamble_sim_run(bal, sim_count, row_labels, col_labels, title, pascal):
    """Runs a roulette simulation a given number of times, with a given starting balance,
    for a given set of bet patterns. Creates a sequence of arrays to be used in the simulation.

    Args:
        bal (int): The dollar value of the starting balance
        sim_count (int): The number of times to run a simulation
        row_labels (list): A list of strings of row labels, amount of starting bets
        col_labels (list): A list of strings of col labels, percentage of balance to win
        title (string): The title of the simulation (pascal, general)
        pascal (bool): If the simulation should be run as pascal (true), general(false)
    """    
    # function to find size of each bet
    bet_size = lambda x, y: (bal * x) * y
    
    # function to create array of each bet repeated in sequence
    # ex: want to win 50 dollars, 5 inital bets would be [10,10,10,10,10]
    bet_array = lambda c_labl,r_labl: np.asarray([bet_size(c_labl,(1/r_labl))
                                                  for _ in range(0,r_labl)],dtype=object)
    # builds an array for functions with given data
    data_arr = lambda c_labl,pascal_arr: np.asarray([bet_size(c_labl,elem/sum(pascal_arr))
                                                     for elem in pascal_arr],dtype=object)
    # decides which which way to build the array
    fn_dec = lambda c_labl,r_labl,pascal: data_arr(c_labl,
                                                   func_pascal(r_labl)) if pascal else bet_array(c_labl,r_labl)
    # function to create a row of a TOTAL bet amount split into intervals of 2-10 inclusive
    row_array = lambda r_labl,pascal: np.asarray([fn_dec(c_labl,r_labl,pascal)
                                                  for c_labl in col_labels],dtype=object)
    # function to pad the rows with zeros so that each bet array
    # has the same number of zeros as the max number of bets (10)
    pad_row = lambda r_labl,pascal: np.asarray([np.pad(arr,(0,max(row_labels)-r_labl))
                                                 for arr in row_array(r_labl,pascal)],dtype=object)
    # sequence array seq_array[i] returns a (9x10) array of bets with bet counts sized
    # from 2-10 (inclusive) for the i'th percentage of balance bet
    # ex seq_array[0] contains the bets that sum to a win of $50 (ie. 1000 * 0.05)
    # spread over 2,3,4,5,6,7,8,9,10 bets. (there is an seperate arraay for each number of bets)
    seq_array = np.stack(np.asarray([pad_row(r_labl,pascal)
                                     for r_labl in row_labels],dtype=object), axis=1)

    create_data(bal,sim_count,row_labels,col_labels,seq_array,title)

def create_data(bal, sim_count, row_labels, col_labels, seq_array, title):
    """Creates the data of the simulation for the the given parameters. Runs the simulation
    and stores the data.

    Args:
        bal (int): The dollar value of the starting balance
        sim_count (int): The number of times to run a simulation
        row_labels (list): A list of strings of row labels, amount of starting bets
        col_labels (list): A list of strings of col labels, percentage of balance to win
        seq_array (ndarray(ndarray)): a sequence of ndarrays to be used in the simulations
        title (string): The title of the simulation
    """    
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
def normalize_seq(seq, bal, percent_array):
    """Normalizes the seq array so that for each element (a percentage of the entire sequence),
    is convereted to a dollar value of the desired profit for each profit level in the percent_array.

    Args:
        seq (ndarray(ndarray)): a list of sequences, each element in the sequences is a percentage, all add to 1
        bal (int): the starting balance in dollars
        percent_array (ndarray): an array of the desired profitability percentages

    Returns:
        ndarray(ndarray): An array of len(seq)xlen(percent_array) where there is one of each sequence arrays
        for the profit levels in the percent array
    """    
    max_len_row = max([len(row) for row in seq])
    bet_size = lambda x, y: (bal * x) * y
    row_func = lambda row, percent: np.asarray([bet_size(percent, item) for item in row], dtype=object)
    pad_func = lambda data, percent: np.asarray([np.pad(row_func(row, percent),
                                                        (0, max_len_row-len(row))) for row in data], dtype=object)    
    return np.asarray([pad_func(seq, percent) for percent in percent_array])



# percentage to win, what percent of current balance the player wishes to win
# 5% to 50% in increments of 5%
# initial testing showed that a higher percent desired to win, relative to the current balance had
# a lower chance of being profitable (decided to cap at 50% and a 0% return is not valid or of interest)
percent_to_win = np.linspace(0.05,0.5,10)


# bets from 2-10(inclusive)
# list must have 2 elements to function at start
num_bets = np.arange(2,11,1)

# create data from normal distribution
norm_seq = [[0.022, 0.136, 0.341, 0.341, 0.136, 0.022], [0.001, 0.021, 0.136, 0.341, 0.341, 0.136, 0.021, 0.001],
             [0.001, 0.005, 0.017, 0.044, 0.092, 0.15, 0.191, 0.191, 0.15, 0.092, 0.044, 0.017, 0.005, 0.001]]
norm_row_labels = [len(row) for row in norm_seq]
norm_seq = normalize_seq(norm_seq, 1000, percent_to_win)


# create custom data set 
custom_seq = [[0.10,0.20,0.40,0.20,0.10], [0.05,0.10,0.15,0.40,0.15,0.10,0.05]]
custom_row_labels = [len(row) for row in custom_seq]
custom_seq = normalize_seq(custom_seq, 1000, np.array([0.1]))
gamble_sim_run(1000,1000,num_bets, percent_to_win, "", pascal=False)
gamble_sim_run(1000,1000,num_bets,percent_to_win,"pascal_",pascal=True)
create_data(1000,1000,norm_row_labels,percent_to_win,norm_seq,"norm_")
create_data(1000,1000,custom_row_labels,np.array([0.10]),custom_seq,"custom_")
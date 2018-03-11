# Author: Changyu Liu
# Last update: 11/06/2017

import sys
import pandas as pd

def parse_file(input_file):
    price_table = pd.read_csv(input_file,
                              parse_dates=[0],
                              index_col=0,)
    open_table = price_table.resample('1T').first()
    close_table = price_table.resample('1T').last()
    high_table = price_table.resample('1T').max()
    low_table = price_table.resample('1T').min()

    return open_table, close_table, high_table, low_table

if __name__ == "__main__":
    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv))
    open_table, close_table, high_table, low_table = parse_file(sys.argv[1])
    print (open_table)
    print (close_table)
    print (high_table)
    print (low_table)
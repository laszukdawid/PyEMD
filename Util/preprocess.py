# Author: Changyu Liu
# Last update: 11/06/2017

import sys
import pandas as pd

def reformat(input_file, output_file):
    try:
        price_table = pd.read_csv(input_file,
                                  names=['Date', 'Time', 'Bid', 'Ask', 'ex1', 'ex2', 'ex3'],
                                  header=None,
                                  parse_dates=[[0, 1]],
                                  index_col=0)
        price_table.to_csv(output_file)
    except:
        print("Unexpected error:", sys.exc_info()[0])

if __name__ == "__main__":
    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv))
    reformat(sys.argv[1], sys.argv[2])

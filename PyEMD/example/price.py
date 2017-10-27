# Author: Changyu Liu 
# Last update: 10/24/2017
from __future__ import division, print_function
import sys
sys.path.append('/Users/cliu270/Projects/PyEMD')

import numpy  as np
import pylab as plt
from PyEMD import EMD
import pandas as pd

# Read price data from file
price_table = pd.read_csv("~/Documents/Data/AAPL.csv")
time_column = price_table.Time
close_column = price_table.Close
#time_column = time_column.values
#close_column = close_column.values
time_column = time_column.as_matrix(columns=None)
close_column = close_column.as_matrix(columns=None)
time_column = time_column[:200]
close_column = close_column[:200]
#print (time_column)
#print (close_column)

# Define signal
#t = np.linspace(0, 1, 200)
#s = np.cos(11*2*np.pi*t*t) + 6*t*t
t = time_column
s = close_column

# Execute EMD on signal
IMF = EMD().emd(s,t)
N = IMF.shape[0]+1

# Plot results
plt.subplot(N,1,1)
plt.plot(t, s, 'r')
plt.title("Input signal: AAPL close price")
plt.xlabel("Time [s]")

for n, imf in enumerate(IMF):
    plt.subplot(N,1,n+2)
    plt.plot(t, imf, 'g')
    plt.title("IMF "+str(n+1))
    plt.xlabel("Time [s]")

plt.tight_layout()
plt.savefig('simple_example')
plt.show()

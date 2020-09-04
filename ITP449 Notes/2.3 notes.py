#--- split a (8,4) shaped matrix of random integers ranging from 10 to 20 into 4 matrices with shape (4,2)
import numpy as np

x=np.random.randint(low=10, high=20, size=(8,4))

[x1,x2] = np.hsplit(x,2)
[x3,x4] = np.vsplit(x1,2)
[x5,x6] = np.vsplit(x2,2)
print(x3, "\n")
print(x4, "\n")
print(x5, "\n")
print(x6)


#---create a series of length 20 that contains random integers between 1 and 10...---
import pandas as pd
import numpy as np

srs = pd.Series(np.random.randint(low=1,high=11, size=20)

print(len(srs[srs.value > 4]))

#{} means you are creating a dictionary

#----create the DataFrame shown below and filter to show only the rows..

import pandas as pd

data = {'dessert': ['cupcake', 'donut', 'cake', 'ice cream'],
        'worthIt': [False, True, True, True]}
frame = pd.DataFrame(data)

print(frame[frame['worthIt'].values])


#---combine two series to form a dataframe
import pandas as pd
import numpy as np

ser1 = pd.Series(list('abcdefghijklmnopqrstuvwxyz'))
ser2 = pd.Series(np.arange(26))

data = {'col1': ser1, 'col2': ser2}
frame = pd.DataFrame(data)

print(frame.head())
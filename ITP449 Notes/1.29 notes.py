---- one dimension----

import numpy as np

x=np.arange(2, 11)
print(x, "\n")
print("1:", x[2])
print("2", x[3:])
print("3", x[:4])
print("4", x[2:8])
print("5", x[2:8:2])
print("6", x[3:])

--- two dimension ---

import numpy as np

x=np.arange(2, 11).reshape(3,3)
print(x, "\n")
print("1:", x[1])
print("2", x[:2])
print("3", x[1:2])
print("4 \n", x[1:3, 1:3])
print("5", x[2:8:2])
print("6", x[3:])

--- more ---
import numpy as np

array1=np.arange(10)
print(x, "\n")
print("1:", array1)
print("2", array1[5])
print("3", array1[5:8])

array1[5:8] = 12
print("\n Update arry1[5:8} slice \n")
print("5", array1)

--- mean ---
import numpy as np

np_2d = np.array([[1,3,5], [2,4,6]])

print(np.mean(np_2d))
print(np.mean(np_2d[0]))
print(np.mean(np_2d[:,1:3]))

#--- 4x4 matrix with values ranging from 100 to 115; compute mean----
import numpy as np

x = np.arange(100,116).reshape(4,4)
print(x, "\n")
evens = x[::,::2]
print(evens, "\n")
print(evens.mean())

--- more example--
import numpy as np

x = np.arange(20).reshape(4,5)
print(x, "\n")
print(x >= 5, "\n")
print(x[x >= 5])

---comparison ----
#didn't take the notes

--
import numpy as np

x = np.arange(['Bob','Joe', 'Will', 'Bob'])
data = np.array([[1,2,3],
                 [4,5,6],
                 [7,8,9],
                 [10,11,12]])

print(names == 'Bob')
print("Boolean Indexing: \n", data[names == 'Bob'])

--index for column--
    import numpy as np

    x = np.arange(['Bob', 'Joe', 'Will', 'Bob'])
    data = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9],
                     [10, 11, 12]])

    print(names == 'Bob')
    print("Boolean and Column Indexing: \n", data[names == 'Bob', :2])

--- exercise ---

import numpy as np

a = np.array([1,2,3,np.nan, 5, 6,7,np.nan])
b = a[np.isnan(a) == False].astype(np.int64)

print(b)

--- exercise ---
import numpy as np

a = np.ones((3,3))
b = np.zeros((2,3))
n = np.full(3,2)

print(np.vstack((a,b))) #vstack is to stack array together
print(np.hstack((a,b))) #hstack is stack array horizontally
print(np.column_stack((a,b,c)))
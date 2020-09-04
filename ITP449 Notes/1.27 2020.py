x= int(input("first no:"))
y= int(input("second no:"))
z= int(input("third no:"))

list = [x, y, z]

list.sort()

if x%2 = 0 and y%2 = 0 and z%2 = 0:
    print("All numbers are even")

elif list[2]%2 != 0:
    print(list[2], "is the highest odd number")

elif list[1]%2 != 0:
    print(list[1], "is the highest odd number")

else list[0]%2 != 0:
    print(list[0], "is the highest odd number")



-----
import numpy as np

odds = [1, 3, 5, 7]
evens = [2, 4, 6, 8]

np_odds = np.array(odds)
np_evens = np.array(evens)

print(type(odds))
print(type(np_odds))
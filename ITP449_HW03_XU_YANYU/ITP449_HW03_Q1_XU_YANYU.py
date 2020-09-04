#Yanyu Xu
#ITP_449, Spring 2020
#HW03
#Question 1


import matplotlib.pyplot as plt
import numpy as np

x = np.random.randint(0, 201,300)
y = np.random.randint(0, 301,300)

#plt.scatter(x,y,'c*')
fig = plt.figure()


plt.plot(x,y, 'g.')



#axis = np.random.random(200)

#print(axis)

plt.xticks([0,50,100,150,200,250,300])
plt.yticks([0,50,100,150,200,250,300])
plt.xlabel('Random integers 1 to 200', )
plt.ylabel('Random integers 1 to 300')
plt.title('Scatter plot of random integers', color ='g')
plt.grid(which = 'major')

plt.show()
#Yanyu Xu
#ITP_449, Spring 2020
#HW03
#Question 2

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()

axes = fig.subplots(2, 2, sharex = True, sharey = True)
x = np.random.randn(400)


plot1 = fig.add_subplot(2, 2, 1)
plot1.hist(x, bins = 20, color = 'm')
plot1.set_xticks([-3, -2, -1, 0, 1, 2])
plot1.set_yticks ([0, 10, 20, 30, 40, 50])
plot1.set_xticklabels([]) #turns off xticks
plot1.grid(which = 'major')


plot2 = fig.add_subplot(2, 2, 2)
plot2.hist(x, bins = 40, color = 'm')
plot2.set_xticks([-3, -2, -1, 0, 1, 2])
plot2.set_yticks ([0, 10, 20, 30, 40, 50])
plot2.set_xticklabels([]) #turns off xticks
plot2.set_yticklabels([]) #turns off xticks
plot2.grid(which = 'major')


plot3 = fig.add_subplot(2, 2, 3)
plot3.hist(x, bins = 60, color = 'm')
plot3.set_xticks([-3, -2, -1, 0, 1, 2])
plot3.set_yticks ([0, 10, 20, 30, 40, 50])
plot3.grid(which = 'major')


plot4 = fig.add_subplot(2, 2, 4)
plot4.hist(x, bins = 80, color = 'm')
plot4.set_xticks([-3, -2, -1, 0, 1, 2])
plot4.set_yticks ([0, 10, 20, 30, 40, 50])
plot4.set_yticklabels([]) #turns off xticks
plot4.grid(which = 'major')



plt.show()

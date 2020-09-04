#Yanyu Xu
#ITP_449, Spring 2020
#HW02
#Question 2

import random
def main():
    tries = 10000
    head = 0
    tail = 0
    max1 = 0
    streak = 0
    lastnum = -1
    for i in range(tries):
        coin = random.randint(0, 1)
        if coin == 0:
            head += 1
        elif coin == 1:
            tail += 1
        if coin == lastnum:
            streak += 1
        else:
            streak = 0
            lastnum = coin
        max1 = max(max1, streak)
        headcount = (head/total)*100
        tailcount = (tail / total) * 100
    print("Heads:", headcount,"%")
    print("Tails:",tailcount, "%")
    print("Longest streak:", max1)

main()
#Yanyu Xu
#ITP_449, Spring 2020
#HW02
#Question 3

import re
def main():
    ask = input("Please enter your password:")
    while True:
        if (len(ask) < 8):
            print(":( Try Again")
            ask = input("Please enter your password:")
        elif not re.search("[a-z]", ask):
            print(":( Try Again")
            ask = input("Please enter your password:")
        elif not re.search("[A-Z]", ask):
            print(":( Try Again")
            ask = input("Please enter your password:")
        elif not re.search("[0-9]", ask):
            print(":( Try Again")
            ask = input("Please enter your password:")
        elif not re.search("[-!@#$]", ask):
            print(":( Try Again")
            ask = input("Please enter your password:")
        elif re.search("\s", ask):
            print(":( Try Again")
            ask = input("Please enter your password:")
        else:
            print("Access Granted")
            break





main()
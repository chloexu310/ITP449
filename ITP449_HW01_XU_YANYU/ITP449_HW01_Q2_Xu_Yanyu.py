#Yanyu Xu
#ITP_449, Spring 2020
#HW01
#Question 2



def main():
    userInput = input("Enter your name:")
    ltrList = list(userInput)
    ltrList.remove(" ")
    print(userInput, ",your name is", len(ltrList), "characters long")


main()
#Yanyu Xu
#ITP_449, Spring 2020
#HW01
#Question 3

def main():
    userInput = int(input("Enter the month number:"))
    while userInput > 12 or userInput < 0:
        userInput = int(input("Enter a valid month number between 1 and 12:"))

    if userInput == 1:
        print("January has 31 days")
    elif userInput == 2:
        print("February has 28 days")
    elif userInput == 3:
        print("March has 31 days")
    elif userInput == 4:
        print("April has 30 days")
    elif userInput == 5:
        print("May has 31 days")
    elif userInput == 6:
        print("June has 30 days")
    elif userInput == 7:
        print("July has 31 days")
    elif userInput == 8:
        print("August has 31 days")
    elif userInput == 9:
        print("September has 30 days")
    elif userInput == 10:
        print("October has 31 days")
    elif userInput == 11:
        print("November has 30 days")
    elif userInput == 12:
        print("December has 31 days")


main()
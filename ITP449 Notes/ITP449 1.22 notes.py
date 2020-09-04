print("Hello World", end="***")
print("Where does this line go?")

print("Hello World!", end=" ")
print("Where does this line go?")

print("Hello World!", end ="")

print("I love" + "naps.")
print("I love", "naps")

print("\"Python\" comes from a 1970s comedy series")



age = 10
name = "Sophie"
chanceOfRain = 0.8
isItRaining = True

language = "Python"
message = "I love programming in " + language
print(message)

lastName = "Steinbeck"
print(lastName)
print(lastName.upper())


-----Input-------
name = input("What's your name?")
print("hi", name)

#sum two number
num1 = int(input("Enter a number: "))
num2 = int(input("Enter another number"))
print(num1+num2)

#password
password = input("Enter your password:")
if password == "secret":
    print("Access Granted")
else:
    print("Access Denied")

#vote
age = int(input("Enter your age: "))
if age >= 18:
    print("You can vote")

else:
    print("You can't vote yet")

#vote and disaply the years
age = int(input("Enter your age: "))
if age >= 18:
    print("You can vote")

else:
    years = 18 - age
    print("You can vote in" + str(years) + " years!")

#fav topping
favFood = input("Enter favorite food: ")
if favFood == "pizza":
    print("Your favorite food is pizza!")
    favTopping = input("Enter favorite topping: ")
    if favTopping == "sausage":
        print("Your favorite topping is sausage")
        print("Me too")
else:
    print("something")
print("Have a great day")

#score for the midterm
scoreOnMidtermExam = int(input("What is the score on the midterm exam?"))
if scoreOnMidtermExam >= 90:
    scoreOnMidtermExam = "A"
elif scoreOnMidtermExam >= 80:
    scoreOnMidtermExam = "B"
print(scoreOnMidtermExam)

#pick a number 1-10, display even or odd and error checking
input = int(input("Pick a number: "))
if input < 1 or input > 10:
    print("You must enter a number between 1 and 10!")
elif input % 2 != 0:
    print(str(number) + " is an odd number.")
else:
    print(str(number) + " is an even number.")

#bottle-loops
numBottles = 99
while numBottles > 0:
    print(numBottles, "bottles of cold brew on the wall")
    print("Take one down and pass it around")
    numBottles = numBottles - 1

print("Cold brew is all gone")

#error checking loop
answer = input("Do you want cream(y/n)?")
while answer != "y" or answer != "n":
    answer = input("Do you want cream(y/n)?")
if answer == "y":
    print("You added cream")
elif answer == "n":
    print("You did not add the cream")

#another example
sum = 0
areMore = True
print("Enter positive numbers or -1 to quit")
while areMore == True:
    nextNum = int(input("Enter a number: "))
    if nextNum < 0:
        areMOre = False
    else:
        sum = sum + nextNum
print("The sum is " + str(sum))

#for loop
for num in range(3):
    print(num + 2)

print("done")

#mix
mix = [1, "a", 2, "b"]
print(mix[1])
print(mix[-3])
print(mix[:2])
print(mix[2:])
print(mix[1:3])
print(mix[::-1])

#num list
num = [1, 2]

print(num)

num.append(3)
print(num)

letter = ["a", "b", "c"]
num.extend(letter)
print(num)

num.insert(1, "x")
print(num)

num.remove("x")
print(num)

#
num = [1, 2, 3]
letter = ["a", "b", "c"]

num.sort()
print(num)

letter.sort()
print(letter)

mix.sort()
print(mix) #this is an error

#some functions
x = [2, 40, 21, 5, 103]
#mean
sum(x)/len(x)

---module---
import random
number = random.randrange(6) + 1
print(number)





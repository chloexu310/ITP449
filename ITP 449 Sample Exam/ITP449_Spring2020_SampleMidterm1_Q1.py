print("Multiplication Table\n")
num = int(input("Please enter a whole number: "))
symbol = "X"
for index in range(13):
    product = index * num
    msg = ""
    if index < 10:
        msg = " "
    msg = msg + str(index)
    print(msg, symbol, num, "=", product)
print("Math is fun!")

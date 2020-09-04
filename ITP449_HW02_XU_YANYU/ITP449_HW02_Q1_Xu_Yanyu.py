#Yanyu Xu
#ITP_449, Spring 2020
#HW02
#Question 1

def main():
    print("Change for $1.00:")
    for q in range(0,5,1):
        for d in range(0,11,1):
            for n in range(0,21,1):
                for p in range(0,101,5):
                    if (25*q + 10*d + 5*n +1*p ) == 100:

                        print(q,"quarter", d,"dime", n,"nickel", p,"penny")


main()

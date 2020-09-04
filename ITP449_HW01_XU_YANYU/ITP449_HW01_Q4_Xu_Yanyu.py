#Yanyu Xu
#ITP_115, FALL 2019
#LAB 6
#yanyuxu@usc.edu

def main():
    firstWordList = []
    firstWord = input("Enter a word: ")
    firstWord = firstWord.lower()
    firstWordList.append(firstWord)

    str1 = ""

    firstWord = "".join(firstWord.split())
    for i in firstWord:
        str1 = i+str1
    if str1 == firstWord:
        print("\"", firstWord, "\"","is a palindrome!")
    else:
        print("Sorry", firstWord, "is not a palindrome")


main()
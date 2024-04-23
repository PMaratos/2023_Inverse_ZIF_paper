def splitInputByCommaOrSpace():
    user_input = input().replace(',', ' ').split(' ')
    result_input = []
    for input_item in user_input:
        if input_item != '':
            result_input.append(input_item)

    return result_input

def printOptions(available):
    for optionIndex in range(len(available)):
        print(str(optionIndex + 1) + ". " + available[optionIndex])

def validateInput(available):
    inputIndex = input()
    cleanInputIndex = ''.join(char for char in inputIndex if char.isnumeric())

    while (cleanInputIndex) == "" or int(cleanInputIndex) - 1 not in range(len(available)):
        print("Please select the number of one of the possible choices.")
        inputIndex = input()
        cleanInputIndex = ''.join(char for char in inputIndex if char.isnumeric())

    return int(cleanInputIndex)

def printEmptyLine():
    print("\n")
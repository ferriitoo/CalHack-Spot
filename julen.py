import sys

mapping = {
    'w': 'forward',
    's': 'backwards',
    'a': 'left',
    'd': 'right',
    'z': 'cleaning'
}

def main():
    return_message = ''

    if len(sys.argv) != 2:
        return_message += "\nUsage: python char_echo.py <character>"
        return

    char = sys.argv[1]

    if len(char) != 1:
        return_message += "\nPlease enter only one character."
        return

    instruction_file = '/home/spot/julen_yash/instructions_buffer.txt'

    try:
        if char == 'z':
            with open(instruction_file, 'w') as file:
                file.write('')  # This will empty the file
                print('Cleaning the .txt file')
                
        else:
            with open(instruction_file, 'a') as file:
                file.write(char)
                file.write('\n')
                print('')


    except Exception as e:
        return_message += f'\n{e}'

    # Process the character (you can add more logic here if needed)
    result = process_char(char)

    # Print the processed character
    return_message += f"\nProcessed character: {char}.\nProcessed instruction: {result}.\nWritten {char} to {instruction_file}\n"
    print(return_message)
    sys.stdout.flush()

    # Return the processed character as an exit code
    sys.exit(result)

def process_char(char):
    # You can add your processing logic here
    # For now, it just returns the character as is
    global mapping
    return mapping.get(char, "Unknown")

if __name__ == "__main__":
    main()

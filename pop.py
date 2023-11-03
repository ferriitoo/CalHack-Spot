# import time

# instruction_file = '/home/spot/julen_yash/instructions_buffer.txt'

# while True:
#     time.sleep(5)  # Wait for 5 seconds

#     with open(instruction_file, 'r') as file:
#         lines = file.readlines()

#     if lines:
#         # Read the first character
#         char = lines[0].strip()

#         # Update the file with the remaining characters
#         with open(instruction_file, 'w') as file:
#             file.writelines(lines[1:])

#         # Print the message
#         print(f"I popped this char: {char}")
#     else:
#         print("No characters in the file")



def getting_char():

    instruction_file = '/home/spot/julen_yash/instructions_buffer.txt'
    instruction_file = 'instructions_buffer.txt'

    with open(instruction_file, 'r') as file:
        lines = file.readlines()

    if lines:
        # Read the first character
        char = lines[0].strip()

        # Update the file with the remaining characters
        with open(instruction_file, 'w') as file:
            file.writelines(lines[1:])

        # Print the message
        print(f"I popped this char: {char}")
        return char

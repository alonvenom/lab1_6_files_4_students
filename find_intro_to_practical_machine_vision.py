# Define the path to the README.md file
file_path = 'README.md'

# Define the string to check for
search_string = "hello intro to practical machine vision"

# Function to check if the string exists in the file
def check_readme_for_string(file_path, search_string):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            if search_string in content:
                print(f"The string '{search_string}' was found in {file_path}.")
            else:
                print(f"The string '{search_string}' was NOT found in {file_path}.")
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")

# Call the function to check the README.md file
check_readme_for_string(file_path, search_string)
import re

file_path='descriptions.txt'
out_path='textDescr'

def logFile(fpath,style,content):
    with open(fpath,style) as f:
        f.write(content)

def process_prompts(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            # Split the content by one or more newlines using regular expressions
            prompts = re.split(r'\n+', content)
            # Strip extra whitespace and filter out any empty prompts
            prompts = [prompt.strip() for prompt in prompts if prompt.strip()]

        for i, prompt in enumerate(prompts, start=1):
            logFile(f'{out_path}/p_{i:04d}.txt','w',prompt)
            print(prompt)

    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
# file_path = 'prompts.txt'  # Replace with your actual file path
process_prompts(file_path)

import argparse

from utils import generate_response

# Set up command line argument parsing
parser = argparse.ArgumentParser(description='Generate a response using a GPT model and write to a file or display it.')
parser.add_argument('--model', type=str, required=True, help='The GPT model to use (e.g., gpt-3.5-turbo-1106)')
parser.add_argument('--prompt_filepath', type=str, help='Path to a file containing the prompt. If not provided, prompt will be taken from user input.')
parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for response generation.')
parser.add_argument('--top_p', type=float, default=1.0, help='Top-p value for response generation.')
parser.add_argument('--frequency_penalty', type=float, default=0.5, help='Frequency penalty for response generation.')
parser.add_argument('--presence_penalty', type=float, default=0.5, help='Presence penalty for response generation.')
parser.add_argument('--stream', action='store_true', help='If set, partial message deltas will be sent, like in ChatGPT.')
parser.add_argument('--output_file', type=str, help='Path to the output file. If not provided, response will be printed to the console.')

# Parse arguments
args = parser.parse_args()

terminate = False 
conversation_history = []

# Read the prompt from a file or take user input
if args.prompt_filepath:
    with open(args.prompt_filepath, 'r') as file:
        prompt = file.read()
    print(prompt)
else:
    prompt = input("Enter your prompt: ")

while True:
    # Generate the responsemessage to the conversation history
    conversation_history.append({"role": "user", "content": prompt})

    # Generate the response
    response = generate_response(args.model, conversation_history, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
    print("Response:\n", response)

    # Append model's response to the conversation history
    conversation_history.append({"role": "assistant", "content": response})

    # Write the response to a file if given
    if args.output_file:
        with open(args.output_file, 'a+') as file:
            file.write(response)
        print(f"Response written to {args.output_file}")
    
    prompt = input("Enter your prompt: ")
    if prompt.lower() == "quit" or prompt.lower() == "exit":
        break
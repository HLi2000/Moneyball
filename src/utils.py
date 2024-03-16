import json
import time
import openai 
import os
import random

import numpy as np 

from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()

# # Define your OpenAI API key
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI client
client = openai.Client()

# Declare a global dictionary to store execution times
execution_times = {}


def timeit(func):
    """
    A decorator that stores the execution time of the decorated function in a global dictionary.
    """
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        # Store the execution time with the function's name as the key
        execution_times[func.__name__] = execution_time
        return result
    return wrapper


def load_and_stack_embeddings(file_names, embeddings_dir):
    """
    Load embeddings from successful and unsuccessful datasets and stack them.
    
    Args:
        file_names (list): List of embedding file names.
        embeddings_dir (str): Directory where embedding files are stored.
        
    Returns:
        dict: A dictionary where keys are the types of embeddings and values are the stacked embeddings.
    """
    # Initialize a dictionary to hold the embeddings, categorized by type
    embeddings_by_type = {}
    
    # Organize file names by type
    for file_name in file_names:
        # Determine the type of embeddings and whether it's successful or unsuccessful
        parts = file_name.replace('.npy', '').split('_')
        embedding_type = '_'.join(parts[1:])  # Exclude the first part (successful/unsuccessful)
        category = parts[0]  # 'successful' or 'unsuccessful'
        
        # Add to the embeddings_by_type dictionary
        if embedding_type not in embeddings_by_type:
            embeddings_by_type[embedding_type] = {'successful': None, 'unsuccessful': None}
        embeddings_by_type[embedding_type][category] = os.path.join(embeddings_dir, file_name)
    
    # Load and stack embeddings for each type
    stacked_embeddings = {}
    for embedding_type, paths in embeddings_by_type.items():
        successful_path = paths['successful']
        unsuccessful_path = paths['unsuccessful']
        
        if successful_path and unsuccessful_path:
            successful_embeddings = np.load(successful_path)[:200]
            unsuccessful_embeddings = np.load(unsuccessful_path)[:200]
            stacked = np.vstack((successful_embeddings, unsuccessful_embeddings))
            stacked_embeddings[embedding_type] = stacked
        else:
            # Handle cases where either successful or unsuccessful embeddings are missing
            print(f"Missing embeddings for {embedding_type}. Only found {'successful' if successful_path else 'unsuccessful'} category.")
    
    return stacked_embeddings


def generate_response(model, prompt, temperature=0.9, top_p=0.1, frequency_penalty=0.5, presence_penalty=0.5, stream_flag=False):
    """
    Generate a response from the GPT model.

    Args:
        model (str): The GPT model to use (e.g., 'gpt-3.5-turbo-1106').
        prompt (str or list): The user prompt or a list of message dictionaries for maintaining conversation context.
                              Each message in the list should be a dictionary with 'role' (either 'user' or 'assistant') and 'content'.
        top_p (float): Top-p value for response generation, controlling diversity of the response.
        temperature (float): Temperature for response generation.
        top_p (float): Top-p value for response generation.
        frequency_penalty (float): Frequency penalty for response generation.
        presence_penalty (float): Presence penalty for response generation.
        stream_flag (bool): If set, partial message deltas will be sent, like in ChatGPT. 
    Returns:
        str: The generated response from the model.
    """
    # Prepare the messages
    if isinstance(prompt, list):
        messages = prompt
    else:
        messages = [{"role": "user", "content": prompt}]

    # Get a response from the model
    if stream_flag:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stream=stream_flag
        )

        # Extract the model's response
        response = ""
        for chunk in stream:    
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content
    else:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stream=stream_flag
        )
        response = completion.choices[0].message.content

    return response


def select_random_profile(preprocessed_df, extracted_info_df, embeddings):
    """
    Extracts a random porfile and its corresponding data from provided DataFrames and embeddings,
    then removes these entries from each structure.

    Args:
        preprocessed_df (pd.DataFrame): The DataFrame containing preprocessed data.
        embeddings (dict): A dictionary where keys are types of embeddings and values are numpy arrays.
        extracted_info_df (pd.DataFrame): The DataFrame containing extracted information.

    Returns:
        tuple: Containing the extracted row from preprocessed_df, corresponding embeddings, 
               and the extracted row from extracted_info_df. The original structures are modified in-place.
    """
    # Select a random row index
    random_index = random.choice(preprocessed_df.index.tolist())
    # random_index = 1139
    # Extract corresponding data
    preprocessed_row = preprocessed_df.loc[random_index]
    extracted_info_row = extracted_info_df.loc[random_index]
    
    # Extract corresponding embeddings, assuming embeddings are ordered
    corresponding_embeddings = {key: value[random_index] for key, value in embeddings.items()}

    # Remove the entries from the DataFrames and embeddings
    preprocessed_df.drop(index=random_index, inplace=True)
    extracted_info_df.drop(index=random_index, inplace=True)
    for key in embeddings.keys():
        embeddings[key] = np.delete(embeddings[key], random_index, axis=0)

    # Adjust the index of the DataFrames after deletion
    preprocessed_df.reset_index(drop=True, inplace=True)
    extracted_info_df.reset_index(drop=True, inplace=True)

    return preprocessed_row, extracted_info_row, corresponding_embeddings


def select_top_and_random(data: List[List], k: int = 5) -> Tuple[List[List], List[List]]:
    """
    Selects the top k elements based on the second element of each sublist in descending order,
    and also selects k random elements from the remaining elements, ensuring no overlap with the top k.
    
    Arguments:
        data (List[List]): The input data, a list of lists where each sublist contains at least two elements.
        k (int): The number of elements to select for both the top k and random k. Default is 5
    
    Returns:
        Tuple[List[List], List[List]]: A tuple containing two lists:
            - The first list contains the top k elements based on the second element of the sublists.
            - The second list contains k random elements from the remaining data, excluding the top k elements.
    """
    # Sort the data based on the second element of each sublist in descending order
    sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
    # Select the top k elements
    top_k = sorted_data[:k]
    # Remove the top k elements from the pool for random selection
    remaining_data = [item for item in data if item not in top_k]
    # Select k random elements from the remaining data
    random_k = random.sample(remaining_data, k)
    
    return top_k, random_k


def read_prompts_to_dict(directory_path):
    """
    Reads all .txt files in the specified directory and stores their content in a dictionary.

    Args:
        directory_path (str): the path to the directory containing .txt files.

    Returns:
        prompts_dict (dict): a dictionary with filenames as keys and file contents as values.
    """
    prompts_dict = {}
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                # The key is the filename without the .txt extension
                prompts_dict[os.path.splitext(filename)[0]] = file.read()
    return prompts_dict


def load_json_data(filepath):
    try:
        with open(filepath, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {filepath}")
        return None
    
    
def normalize_evaluation_score(score):
    # Check if the score is already on a 0-1 scale
    if 0 <= score <= 1:
        return score
    # Check if the score is on a 0-10 scale
    elif 0 <= score <= 10:
        return score / 10
    # Check if the score is on a 0-100 scale
    elif 0 <= score <= 100:
        return score / 100
    # If the score does not fit in any of these categories, return an error or handle accordingly
    else:
        raise ValueError("Score must be between 0-1, 0-10, or 0-100")
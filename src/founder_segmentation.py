import argparse
import ast
import json
import logging
import os

import concurrent.futures
import pandas as pd
import requests

import torch
from dotenv import load_dotenv
from torch.utils.data import Dataset, DataLoader

from utils import (
    select_random_profile,
    select_top_and_random,
    load_and_stack_embeddings,
    load_json_data,
    read_prompts_to_dict,
    timeit,
    execution_times,
    generate_response
)

# Ensure the logs/ directory exists
logs_dir = 'logs'
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Suppress verbose HTTP messages from specified libraries
# logging.getLogger('urllib3').setLevel(logging.WARNING)
# logging.getLogger("requests").setLevel(logging.WARNING)
# logging.getLogger("openai").setLevel(logging.WARNING)

# Create a custom logger
logger = logging.getLogger("Founder Segmentation")
logger.setLevel(logging.INFO)  # Set the log level to INFO

# Create handlers for logging to file and console
file_handler = logging.FileHandler(os.path.join(logs_dir, 'founder_segmentation_output.log'), mode='w+')
console_handler = logging.StreamHandler()

# Optionally, set a formatter if you want a specific log message format
formatter = logging.Formatter('%(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


load_dotenv()


@timeit
def load_data_and_embeddings(args):
    # Load extracted information DataFrames and label sources
    extracted_info_successful_df = pd.read_csv(args.extracted_info_successful_path)
    extracted_info_unsuccessful_df = pd.read_csv(args.extracted_info_unsuccessful_path)
    extracted_info_successful_df['source'] = 'Successful'
    extracted_info_unsuccessful_df['source'] = 'Unsuccessful'
    extracted_info_combined_df = pd.concat([extracted_info_successful_df.iloc[:200], extracted_info_unsuccessful_df.iloc[:200]], ignore_index=True)

    # Load preprocessed DataFrames and label sources
    preprocessed_successful_df = pd.read_csv(args.preprocessed_successful_path)
    preprocessed_unsuccessful_df = pd.read_csv(args.preprocessed_unsuccessful_path)
    preprocessed_successful_df['source'] = 'Successful'
    preprocessed_unsuccessful_df['source'] = 'Unsuccessful'
    preprocessed_combined_df = pd.concat([preprocessed_successful_df.iloc[:200], preprocessed_unsuccessful_df.iloc[:200]], ignore_index=True)

    return extracted_info_combined_df, preprocessed_combined_df

class FounderDataset(Dataset):
    # def __init__(self, preprocessed_df, extracted_info_df, embeddings_dict, prompts_dict):
    def __init__(self, extracted_info_df, preprocessed_df, prompts_dict):
        self.extracted_info_df = extracted_info_df
        self.preprocessed_df = preprocessed_df
        self.prompts_dict = prompts_dict

    def __len__(self):
        return len(self.extracted_info_df)

    def __getitem__(self, idx):
        profile_data = self.preprocessed_df.iloc[idx]
        extracted_data = self.extracted_info_df.iloc[idx]

        return idx, profile_data['founder_linkedin_url'], extracted_data['summary'], profile_data['source']

def collate_fn(batch):
    # summaries, profiles, extracted_data, embeddings = zip(*batch)
    indices, urls, summaries, sources = zip(*batch)
    profile_batch = {
        'indices': indices,
        'urls': urls,
        'summaries': summaries,
        'sources': sources,
    }
    return profile_batch


def main(args):
    #
    logger.info("Founder Segmentation")
    logger.info("\nargs: " + json.dumps(vars(args), indent=4))

    # Ensure the output directories exist
    os.makedirs(args.datasets_output_folder, exist_ok=True)

    # Load prompts
    prompts_dict = read_prompts_to_dict(args.prompts_dir)

    # Load data
    logger.info("\nLoading data ...")
    # extracted_info_combined_df, preprocessed_combined_df, embeddings = load_data_and_embeddings(args)
    extracted_info_combined_df, preprocessed_combined_df = load_data_and_embeddings(args)
    logger.info("Loading data completed.\n")

    logger.info(f"\nSeed set to {args.seed} \n")
    torch.manual_seed(args.seed)

    # dataset = FounderDataset(preprocessed_combined_df.iloc[:150], extracted_info_combined_df.iloc[:150], embeddings, prompts_dict)
    dataset = FounderDataset(extracted_info_combined_df, preprocessed_combined_df, prompts_dict)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    url_level_source = []  # List to store tuples of (url, level, source)
    original_indices = []  # List to store original indices

    for batch_idx, batch in enumerate(dataloader):
        indices = batch['indices']
        urls = batch['urls']
        summaries = batch['summaries']
        sources = batch['sources']

        concatenated_string = ""
        for idx, summary in enumerate(summaries):
            summary = summary.replace('\n\n', '\n')
            concatenated_string += f"Profile {idx}:\n{summary}\n\n"
        concatenated_string = concatenated_string.rstrip()

        prompt = prompts_dict['Founders_Segmentation'] + '\n\n' + concatenated_string
        logger.info(f"\n\n----------------------- Batch {batch_idx} -----------------------\n")
        logger.info("\nPrompt: \n\n" + prompt)

        # Attempt to generate response
        retry_attempts = 3
        for attempt in range(retry_attempts):
            response = generate_response(
                # model='gpt-3.5-turbo',
                model='gpt-4-1106-preview',
                prompt=prompt,
                temperature=0.9,
                top_p=0.5,
                frequency_penalty=0.5,
                presence_penalty=0.5,
                stream_flag=True
            )

            try:
                # Try to evaluate the last line of the response
                levels = ast.literal_eval(response.split("\n")[-1])
                if all(level in ["L1", "L2", "L3", "L4", "L5"] for level in levels) and len(levels) == len(urls):
                    logger.info("\nResponse: \n\n" + response)
                    break
            except Exception as e:
                logger.warning(f"Failed on attempt {attempt + 1}. Retrying...")
        else:
            raise RuntimeError(f"Failed after {retry_attempts} attempts.")

        for index, url, level, source in zip(indices, urls, levels, sources):
            original_indices.append(index)
            url_level_source.append((url, level, source))


    # Convert the list of tuples into a DataFrame
    segmentation_df = pd.DataFrame(url_level_source, columns=['url', 'level', 'source'])
    segmentation_df.index = original_indices
    segmentation_df = segmentation_df.reindex([*range(len(original_indices))])

    # Save the DataFrame as a CSV file
    segmentation_filename = 'founder_segmentation_results.csv'
    segmentation_df.to_csv(os.path.join(args.datasets_output_folder, segmentation_filename), index=True)
    print(f"\n\nFounder segmentation results exported to {os.path.join(args.datasets_output_folder, segmentation_filename)}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FounderSegmentation")
    parser.add_argument('--extracted_info_successful_path', type=str, required=True, help="Path to the CSV file containing extracted info for successful profiles.")
    parser.add_argument('--extracted_info_unsuccessful_path', type=str, required=True, help="Path to the CSV file containing extracted info for unsuccessful profiles.")
    parser.add_argument('--preprocessed_successful_path', type=str, required=True, help="Path to the CSV file containing preprocessed data for successful profiles.")
    parser.add_argument('--preprocessed_unsuccessful_path', type=str, required=True, help="Path to the CSV file containing preprocessed data for unsuccessful profiles.")
    parser.add_argument('--prompts_dir', type=str, required=True, help="Directory containing the prompts files.")
    parser.add_argument('--datasets_output_folder', type=str, default='data', help='Folder to save the extracted info datasets.')
    parser.add_argument('--batch_size', type=int, default=5, help="Batch size for founder segmentation.")
    parser.add_argument('--seed', type=int, default=2024, help="Seed for random process in PyTorch.")

    # Parse the arguments
    args = parser.parse_args()

    # # Provide arguments directly
    # args = argparse.Namespace(
    #     extracted_info_successful_path='../data/summarised_extracted_info_successful_company_and_founder_profiles_dataset.csv',
    #     extracted_info_unsuccessful_path='../data/summarised_extracted_info_unsuccessful_company_and_founder_profiles_dataset.csv',
    #     preprocessed_successful_path='../data/preprocessed/successful_company_and_founder_profiles.csv',
    #     preprocessed_unsuccessful_path='../data/preprocessed/unsuccessful_company_and_founder_profiles.csv',
    #     prompts_dir='../prompts',
    #     datasets_output_folder='../data',
    #     batch_size=4,
    #     seed=2024,
    # )

    main(args)

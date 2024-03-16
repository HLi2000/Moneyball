import argparse
import ast
import json
import logging
import os

import concurrent.futures
import pandas as pd
import requests

from dotenv import load_dotenv

from utils import (
    read_prompts_to_dict,
    timeit,
    execution_times,
    generate_response
)


load_dotenv()


def generate_summary_from_row(row: pd.Series, extracted_info_df, prompts_dict) -> str:
    # idx = 1
    # idx = 3
    # idx = 145
    idx = row.name
    profile_data = row
    extracted_data = extracted_info_df.iloc[idx]

    prompt = (prompts_dict['Generate_Detailed_Founder_Profile_Summary'] +
              "\n\n## Employers' Information:\n" + extracted_data['employer_info'] +
              "\n\n## The Latest Founded Company:" + profile_data['org_name'] +
              "\n\n## JSON STRING:\n" + profile_data['json_string'])
    # print(prompt)
    profile_summary = generate_response(
        # model='gpt-3.5-turbo',
        model='gpt-4-1106-preview',
        prompt=prompt,
        temperature=0.05,
        top_p=0.95,
        frequency_penalty=0.1,
        presence_penalty=0.0,
        stream_flag=True
    )
    print(f"\n\n{idx} \n")
    print(profile_summary)

    return profile_summary


def main(args):
    # Load prompts
    prompts_dict = read_prompts_to_dict(args.prompts_dir)

    # Ensure the output directories exist
    os.makedirs(args.datasets_output_folder, exist_ok=True)

    #
    extracted_successful_profiles_df = pd.read_csv(args.extracted_info_successful_path)
    extracted_unsuccessful_profiles_df = pd.read_csv(args.extracted_info_successful_path)

    # Load data from CSV files specified in command line arguments
    successful_profiles_df = pd.read_csv(args.successful_profiles_filepath)
    unsuccessful_profiles_df = pd.read_csv(args.unsuccessful_profiles_filepath)

    #
    extracted_successful_profiles_df['summary'] = successful_profiles_df.iloc[:200].apply(lambda row: generate_summary_from_row(row, extracted_successful_profiles_df, prompts_dict), axis=1)
    successful_export_filename = 'summarised_extracted_info_successful_company_and_founder_profiles_dataset.csv'
    extracted_successful_profiles_df.to_csv(os.path.join(args.datasets_output_folder, successful_export_filename), index=False)
    print(f"Successful profiles exported to {os.path.join(args.datasets_output_folder, successful_export_filename)}")

    extracted_unsuccessful_profiles_df['summary'] = unsuccessful_profiles_df.iloc[:200].apply(lambda row: generate_summary_from_row(row, extracted_unsuccessful_profiles_df, prompts_dict), axis=1)
    unsuccessful_export_filename = 'summarised_extracted_info_unsuccessful_company_and_founder_profiles_dataset.csv'
    extracted_unsuccessful_profiles_df.to_csv(os.path.join(args.datasets_output_folder, unsuccessful_export_filename), index=False)
    print(f"Unsuccessful profiles exported to {os.path.join(args.datasets_output_folder, unsuccessful_export_filename)}")


if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Get information about employers')
    parser.add_argument('--extracted_info_successful_path', type=str, required=True, help="Path to the CSV file containing extracted info for successful profiles.")
    parser.add_argument('--extracted_info_unsuccessful_path', type=str, required=True, help="Path to the CSV file containing extracted info for unsuccessful profiles.")
    parser.add_argument('--successful_profiles_filepath', type=str, required=True, help='Path to the CSV file with successful company and founder profiles.')
    parser.add_argument('--unsuccessful_profiles_filepath', type=str, required=True, help='Path to the CSV file with unsuccessful company and founder profiles.')
    parser.add_argument('--prompts_dir', type=str, required=True, help="Directory containing the prompts files.")
    parser.add_argument('--datasets_output_folder', type=str, default='data', help='Folder to save the extracted info datasets.')

    # Parse the arguments
    args = parser.parse_args()

    # # Provide arguments directly
    # args = argparse.Namespace(
    #     extracted_info_successful_path='../data/new_extracted_info_successful_company_and_founder_profiles_dataset.csv',
    #     extracted_info_unsuccessful_path='../data/new_extracted_info_unsuccessful_company_and_founder_profiles_dataset.csv',
    #     successful_profiles_filepath='../data/preprocessed/successful_company_and_founder_profiles.csv',
    #     unsuccessful_profiles_filepath='../data/preprocessed/unsuccessful_company_and_founder_profiles.csv',
    #     prompts_dir='../prompts',
    #     datasets_output_folder='../data',
    # )

    main(args)

import argparse
import json
import random
import re
import os

import numpy as np
import pandas as pd
import requests

from sentence_transformers import SentenceTransformer
from typing import List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()


# Function to save employer_dict to data/ directory
def save_employer_dict(employer_dict):
    with open('data/employer_dict.json', 'w') as f:
        json.dump(employer_dict, f)

# Function to load employer_dict from data/ directory if it exists
def load_employer_dict():
    if os.path.exists('data/employer_dict.json'):
        with open('data/employer_dict.json', 'r') as f:
            return json.load(f)
    else:
        return {}


# Load employer_dict from file
employer_dict = load_employer_dict()

idx = 0
num_bing = 0


def get_employer_info_from_row(row: pd.Series, list_big_tech_list: list, list_unicorn_list: list) -> str:
    """
    Extract employment information from a DataFrame row containing a JSON string with employment information.

    This function parses the 'json_string' from the provided Series (row of a DataFrame), extracts the 'employments'
    list, and then constructs a string representation of employment history. If an employment entry includes both
    'employer' and 'title', it formats these details into a readable string. If 'summary' is present, it's included in
    the description. The function aims to get the three most recent employments before founding, if applicable.

    Args:
        row (pd.Series): A pandas Series object representing a row of a DataFrame. The Series must contain a
                         'json_string' column with a JSON string that includes an 'employments' key.

    Returns:
        str: A string representation of the employment history, formatted and concatenated from the 'employments'
             field in the 'json_string'. If no employments are provided, returns "No previous experience".
    """
    global idx, num_bing, employer_dict

    # def is_unicorn_name(employer_name: str) -> bool:
    #     """
    #     Check if any of the keywords exist in the employer name.
    #
    #     Args:
    #         employer_name (str): The name of the employer.
    #
    #     Returns:
    #         bool: True if any keyword exists in the employer name, False otherwise.
    #     """
    #     keywords = ["Authority", "Government", "Agency", "Association", "Organization", "Institution",
    #                 "Bureau", "Foundation", "Department", "Council", "Commission", "Administration"]
    #
    #     for keyword in keywords:
    #         if keyword.lower() in employer_name.lower():
    #             return False
    #     return True


    try:
        employments = json.loads(row['json_string'])['employments']
        info = []

        CB_API_KEY = os.getenv("CB_API_KEY")
        BING_API_KEY = os.getenv("BING_API_KEY")
        SEARCH_ENDPOINT = 'https://api.bing.microsoft.com/v7.0/search'
        HEADERS = {'Ocp-Apim-Subscription-Key': BING_API_KEY}

        employer_names = []
        for employment in employments:
            employer_name = employment.get('employer', {}).get('name', {})
            if not employer_name or employer_name in employer_names:
                continue
            employer_names.append(employer_name)

            if employer_name in employer_dict:
                info.append(employer_dict[employer_name]) if employer_dict[employer_name] != None else None
                continue

            # Make a Bing Search API request
            params = {'q': f'{employer_name} site:crunchbase.com', 'count': 1}
            response = requests.get(SEARCH_ENDPOINT, headers=HEADERS, params=params)
            num_bing += 1
            print(f'Searching via Bing API for {num_bing} times') if num_bing % 10 == 0 else None

            if response.status_code == 200:
                search_results = response.json()
                if 'webPages' in search_results and 'value' in search_results['webPages']:
                    first_result = search_results['webPages']['value'][0]
                    if 'url' in first_result:
                        url = first_result['url']
                        # Extract domain name from URL
                        try:
                            domain = url.split('/')[4]
                        except:
                            continue
                    else:
                        continue
                else:
                    continue
            else:
                # Print the error message from the response content
                print('Error:', response.text)
                employer_dict[employer_name] = None
                continue

            # url = f'https://api.crunchbase.com/api/v4/entities/organizations/{domain}?field_ids=status,funding_total,valuation&user_key={CB_API_KEY}'
            url = f'https://api.crunchbase.com/api/v4/entities/organizations/{domain}?field_ids=status,funding_total,valuation&user_key={CB_API_KEY}'

            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                # print(data)
                status = data['properties'].get('status', {})
                funding_total = data['properties'].get('funding_total', {}).get('value_usd', {})
                valuation = data['properties'].get('valuation', {}).get('value_usd', {})
                # is_unicorn = True if valuation and valuation >= 1_000_000_000 and status and status != 'ipo' and is_unicorn_name(employer_name) else False
                is_unicorn = True if employer_name.lower() in list_unicorn_list else False
                is_big_tech = True if employer_name.lower() in list_big_tech_list else False
                status, funding_total, valuation = 'unknown' if not status else status, 'unknown' if not funding_total else funding_total, 'unknown' if not valuation else valuation
                employer = f"Company: {employer_name}, status: {status}, total funding in USD: {funding_total}, is unicorn: {is_unicorn}, is big tech: {is_big_tech}"
                info.append(employer)
                employer_dict[employer_name] = employer
            else:
                # Print the error message from the response content
                print('Error:', response.text)
                employer_dict[employer_name] = None

        info_string = "; ".join(info) + "." if info else "No information about employers"
    except KeyError:
        info_string = "No information about employers"

    print(str(idx) + ". " + row['founder_linkedin_url'] + ": " + info_string)
    save_employer_dict(employer_dict)
    idx += 1
    return info_string
    

def main(args):
    # Ensure the output directories exist
    os.makedirs(args.datasets_output_folder, exist_ok=True)

    list_big_tech_df = pd.read_csv(args.list_big_tech_path)
    list_big_tech_list = list_big_tech_df['Company'].str.strip().str.lower().tolist()
    list_unicorn_df = pd.read_csv(args.list_unicorn_path)
    list_unicorn_list = list_unicorn_df['Company'].str.strip().str.lower().tolist()

    #
    successful_company_and_founder_profiles_dataset = pd.read_csv(args.extracted_info_successful_path)
    unsuccessful_company_and_founder_profiles_dataset = pd.read_csv(args.extracted_info_successful_path)

    # Load data from CSV files specified in command line arguments
    successful_profiles_df = pd.read_csv(args.successful_profiles_filepath)
    unsuccessful_profiles_df = pd.read_csv(args.unsuccessful_profiles_filepath)

    # Apply function to extract employer info for successful and unsuccessful samples
    successful_company_and_founder_profiles_dataset['employer_info'] = successful_profiles_df.iloc[:200].apply(lambda row: get_employer_info_from_row(row, list_big_tech_list, list_unicorn_list), axis=1)
    unsuccessful_company_and_founder_profiles_dataset['employer_info'] = unsuccessful_profiles_df.iloc[:200].apply(lambda row: get_employer_info_from_row(row, list_big_tech_list, list_unicorn_list), axis=1)

    if args.debug:
        successful_company_and_founder_profiles_dataset = successful_company_and_founder_profiles_dataset
        unsuccessful_company_and_founder_profiles_dataset = unsuccessful_company_and_founder_profiles_dataset
        print(successful_company_and_founder_profiles_dataset)
        print()
        print(unsuccessful_company_and_founder_profiles_dataset)

    # Specify the file paths for the exported CSV files
    successful_export_filename = 'new_extracted_info_successful_company_and_founder_profiles_dataset.csv'
    unsuccessful_export_filename = 'new_extracted_info_unsuccessful_company_and_founder_profiles_dataset.csv'

    # Export the successful company and founder profiles dataset to CSV
    successful_company_and_founder_profiles_dataset.to_csv(os.path.join(args.datasets_output_folder, successful_export_filename), index=False)
    unsuccessful_company_and_founder_profiles_dataset.to_csv(os.path.join(args.datasets_output_folder, unsuccessful_export_filename), index=False)

    print(f"Successful profiles exported to {os.path.join(args.datasets_output_folder, successful_export_filename)}")
    print(f"Unsuccessful profiles exported to {os.path.join(args.datasets_output_folder, unsuccessful_export_filename)}")


if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Get information about employers')
    parser.add_argument('--extracted_info_successful_path', type=str, required=True, help="Path to the CSV file containing extracted info for successful profiles.")
    parser.add_argument('--extracted_info_unsuccessful_path', type=str, required=True, help="Path to the CSV file containing extracted info for unsuccessful profiles.")
    parser.add_argument('--successful_profiles_filepath', type=str, required=True, help='Path to the CSV file with successful company and founder profiles.')
    parser.add_argument('--unsuccessful_profiles_filepath', type=str, required=True, help='Path to the CSV file with unsuccessful company and founder profiles.')
    parser.add_argument('--list_big_tech_path', type=str, required=True, help='Path to the CSV file with a list of nig tech companies.')
    parser.add_argument('--list_unicorn_path', type=str, required=True, help='Path to the CSV file with a list of unicorn startups.')
    parser.add_argument('--datasets_output_folder', type=str, default='data', help='Folder to save the extracted info datasets.')
    parser.add_argument('--debug', type=int, default=0, help='Debug parameter')

    # Parse the arguments
    args = parser.parse_args()

    # # Provide arguments directly
    # args = argparse.Namespace(
    #     extracted_info_successful_path='../data/extracted_info_successful_company_and_founder_profiles_dataset.csv',
    #     extracted_info_unsuccessful_path='../data/extracted_info_unsuccessful_company_and_founder_profiles_dataset.csv',
    #     successful_profiles_filepath='../data/preprocessed/successful_company_and_founder_profiles.csv',
    #     unsuccessful_profiles_filepath='../data/preprocessed/unsuccessful_company_and_founder_profiles.csv',
    #     list_big_tech_path='../data/Big_Tech_Companies_by_Market_Cap.csv',
    #     list_unicorn_path='../data/List_of_Unicorn_Companies_from_CB_Insights.csv',
    #     datasets_output_folder='../data',
    #     debug=0,
    # )

    main(args)

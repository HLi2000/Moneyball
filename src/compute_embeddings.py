import argparse
import json
import random
import re
import os

import numpy as np
import pandas as pd 

from sentence_transformers import SentenceTransformer
from typing import List, Optional, Tuple


DEGREE_MAPPING = {
      "N/A": -1,
      "High School": 0,
      "O Level": 0,
      "GCE":0,
      "A Level": 0,
      "BTEC":0,
      "Certificate":0,
      "Pre-College": 0,
      "Military": 1,
      "Army": 1,
      "Navy": 1,
      "Naval": 1,
      "Air": 1,
      "Associate": 1,
      "Bachelor": 1,
      "BEng": 1,
      "BMath":1,
      "BMus":1,
      "B Eng": 1,
      "B.A.": 1,
      "B.A": 1,
      "BS": 1,
      "BHons": 1,
      "BSc": 1,
      "LLB": 1,
      "S.B.": 1,
      "SB": 1,
      "ScB": 1,
      "BCMS": 1,
      "Sc.B":1,
      "Master": 2,
      "Graduate": 2,
      "MSc": 2,
      "M.Sc": 2,
      "M.Sc.": 2,
      "MA": 2,
      "M.A.": 2,
      "MEng": 2,
      "MLA": 2,
      "PGCHE": 2,
      "MBA": 2,
      "MMath":2,
      "MMus": 2,
      "PhD": 3,
      "Ph.D":3,
      "Ph D":3,
      "DPhil":3,
      "Postgraduate": 3,
      "Doctor of Philosophy": 3,
      "Visiting": 4,
      "Postdoc": 4,
      "Fellow": 4,
      "Post Doc": 4,
    }


TOP_INSTITUTIONS = [
    "Harvard", "MIT", "Massachusetts Institute of Technology", "Stanford", "Oxford", "Cambridge",
    "University of California Berkeley", "Columbia", "California Institute of Technology", "Caltech",
    "Johns Hopkins", "Yale", "UCL", "University College London", "ICL", "Imperial College London",
    "University of California Los Angeles", "Pennsylvania", "UPenn", "Princeton", "Toronto",
    "Cornell", "Tsinghua", "National University of Singapore", "NUS", "ETH", "LSE", "Nanyang"
]


def find_degree_level(degree: str) -> int:
    """
    Find the education level for a given degree using regex matching.
    
    Args:
        degree (str): The degree string to search for.
    
    Returns:
        int: The education level corresponding to the degree.
    """
    def normalize_degree(degree: str) -> str:
        """
        Normalize the degree string by removing punctuation and converting to lower case.
        """
        return re.sub(r'\W+', '', degree).lower()

    normalized_degree = normalize_degree(degree)
    for key, value in DEGREE_MAPPING.items():
        # Normalize the key for comparison
        key_normalized = normalize_degree(key)
        if re.search(key_normalized, normalized_degree):
            return value
    return -1  # Return -1 if no match is found


def extract_education_info(row: pd.Series) -> List[Tuple[Optional[int], Optional[str]]]:
    """
    Extract education levels and majors from a DataFrame row containing a JSON string with education information.

    Args:
        row (pd.Series): A pandas Series object representing a row of a DataFrame.

    Returns:
        List[Tuple[Optional[int], Optional[str]]]: A list of tuples, each containing the education level and major.
    """
    try:
        educations = json.loads(row['json_string'])['educations']
        education_info = []
        for education in educations:
            # Extract the degree and map it to the education level using regex
            degree = education.get('degree', "N/A")
            level = find_degree_level(degree)
            
            # Extract the major, default to None if not present
            major = education.get('major', 'Unknown')
            if major == '':
                major = 'Unknown'
            
            education_info.append((level, major))
    except KeyError:
        education_info = []

    return education_info


def extract_majors_from_row(row: pd.Series) -> List[str]:
    """
    Extract majors from a DataFrame row containing a JSON string with education information.

    This function parses the 'json_string' from the provided Series (row of a DataFrame), extracts the 'educations'
    list, and then collects the 'major' from each education entry into a list. If an education entry does not have
    a 'major', or if the 'major' is an empty string, "Unknown" is added to the list instead.

    Args:
        row (pd.Series): A pandas Series object representing a row of a DataFrame. The Series must contain a
                         'json_string' column with a JSON string that includes an 'educations' key.

    Returns:
        List[str]: A list of majors extracted from the 'educations' field in the 'json_string'. If a 'major' is not
                   specified or is an empty string, "Unknown" is added to the list for that education entry.
    """
    try:
        educations = json.loads(row['json_string'])['educations']
        majors = []
        for education in educations:
            major = education.get('major')
            if major:  # Checks if major is not None and not an empty string
                majors.append(major.replace('unknown', 'Unknown'))
            else:
                majors.append('Unknown')  # Assigns 'Unknown' if major is None or an empty string
    except KeyError:
        majors = []

    return majors


def get_embedding_for_major(major, majors_embeddings_dict):
    """
    Retrieve the embedding vector for a given major from the provided embeddings dictionary. 
    If the major is 'Unknown', returns a zero vector of the same dimension as the embeddings.

    Args:
        major (str): The name of the major for which to retrieve the embedding.
        majors_embeddings_dict (dict): A dictionary mapping major names to their embedding vectors.

    Returns:
        np.ndarray: The embedding vector for the specified major. Returns a zero vector if the major is 'Unknown'
                    or not found in the dictionary.
    """
    # Assuming all embeddings have the same dimension, get the dimension from a random embedding
    embedding_dim = len(list(majors_embeddings_dict.values())[0])
    if major == 'Unknown':
        return np.zeros(embedding_dim)  # Return a zero vector for 'Unknown' major
    else:
        return majors_embeddings_dict.get(major, np.zeros(embedding_dim))  # Lookup the embedding, or return zero vector if not found


def embed_education_info(row, majors_embeddings_dict):
    """
    Embeds each major listed in a DataFrame row using a provided dictionary of major embeddings. 
    It pairs the education level with the embedding for each major. If a major is 'Unknown' or not found,
    it pairs with a zero vector of the appropriate embedding dimension.

    Args:
        row (list of tuples): A list where each element is a tuple containing the education level and the name of the major.
        majors_embeddings_dict (dict): A dictionary mapping major names to their embedding vectors.

    Returns:
        list of tuples: A list where each tuple contains the education level and the embedding vector for the corresponding major.
    """
    embedded_info = [(education_level, get_embedding_for_major(major, majors_embeddings_dict)) for education_level, major in row]
    return embedded_info


def extract_top_university_flags(row: pd.Series) -> List[int]:
    """
    Determine if educations listed in a DataFrame row are from top universities.

    Args:
        row (pd.Series): A pandas Series object representing a row of a DataFrame.

    Returns:
        Int: A flag indicating whether there is an education entry is from a top university (1 for yes, 0 for no).
    """
    try:
        educations = json.loads(row['json_string'])['educations']
        top_university_flags = []
        for education in educations:
            # Extract the institution name
            institution_name = education.get('institution', {}).get('name', '').lower()
            # Check if the institution is listed in top_institutions
            top_university_flag = 1 if any(top_inst.lower() in institution_name for top_inst in TOP_INSTITUTIONS) else 0
            top_university_flags.append(top_university_flag)
    except KeyError:
        top_university_flags = []

    return max(top_university_flags) if len(top_university_flags) != 0 else 0


def extract_employments_from_row(row: pd.Series) -> str:
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
    try:
        employments = json.loads(row['json_string'])['employments']
        jobs = []

        for employment in employments:
            if 'title' in employment and 'employer' in employment:
                employer_name = employment['employer'].get('name', 'Unknown Employer')
                employer_summary = employment['employer'].get('summary', '')
                title = employment.get('title', 'Unknown Title')
                job = f"{employer_name} ({employer_summary}) as {title}" if employer_summary else f"{employer_name} as {title}"
                jobs.append(job)

        job_string = ", ".join(jobs) + "." if jobs else "No previous experience"
    except KeyError:
        job_string = "No previous experience"

    return job_string


def extract_longer_description_from_row(row: pd.Series) -> Optional[str]:
    """
    Extract the longer string between 'allDescriptions' or 'description' from a DataFrame row containing a JSON string.

    This function parses the 'json_string' from the provided Series (row of a DataFrame) and compares the lengths
    of the strings found in 'allDescriptions' and 'description' keys, returning the longer of the two. If only one
    of these keys exists, the function returns the string from that key. If neither key exists, the function returns None.

    Args:
        row (pd.Series): A pandas Series object representing a row of a DataFrame. The Series must contain a
                         'json_string' column with a JSON string that may include 'allDescriptions' or 'description' keys.

    Returns:
        Optional[str]: The longer string found between the 'allDescriptions' and 'description' entries. If neither key
                       is present, returns None. If both keys are missing, or the JSON string does not contain these keys,
                       the function returns an empty string or a default message indicating the absence of a description.
    """
    data = json.loads(row['json_string'])
    all_descriptions = data.get('allDescriptions', '')
    description = data.get('description', '')

    # Return the longer of the two descriptions
    if all_descriptions and description:
        return all_descriptions if len(all_descriptions) > len(description) else description
    elif all_descriptions:
        return all_descriptions
    elif description:
        return description
    else:
        return "No description available" 


def extract_longer_company_description(row: pd.Series) -> Optional[str]:
    """
    Extract the longer string between 'short_description' or 'long_description' from a DataFrame row.

    This function compares the lengths of the strings found in 'short_description' and 'long_description' keys,
    returning the longer of the two. If only one of these keys exists, the function returns the string from that key.
    If neither key exists or if they are both empty, the function returns None.

    Args:
        row (pd.Series): A pandas Series object representing a row of a DataFrame. The Series must contain
                         'short_description' and 'long_description' keys.

    Returns:
        Optional[str]: The longer string found between the 'short_description' and 'long_description' entries.
                       Returns None if both keys are missing or their values are empty.
    """
    # Extract short and long descriptions from the row
    short_description = row.get('short_description', '')
    long_description = row.get('long_description', '')

    # Determine which description is longer and return it
    if len(short_description) >= len(long_description):
        return short_description if short_description else None
    else:
        return long_description
    

def generate_embeddings(texts: List[Optional[str]], model: SentenceTransformer, num_rows: int) -> np.ndarray:
    """
    Generate embeddings for a list of text strings using a specified SentenceTransformer model, ensuring
    the output array matches the number of input rows, with zero vectors for missing texts.
    
    Args:
        texts (List[Optional[str]]): A list of text strings for which to generate embeddings.
        model (SentenceTransformer): A SentenceTransformer model used for generating embeddings.
        num_rows (int): The total number of rows/entries in the dataset.
    
    Returns:
        np.ndarray: A NumPy array of embeddings with the shape `(num_rows, embedding_dimension)`.
    """
    # Initialize a placeholder array of zeros
    embedding_dimension = model.get_sentence_embedding_dimension()
    embeddings = np.zeros((num_rows, embedding_dimension))
    
    # Identify indices of valid (non-None and non-empty) texts
    valid_indices = [i for i, text in enumerate(texts) if text]
    valid_texts = [text for text in texts if text]

    # Generate embeddings only for valid texts
    if valid_texts:
        valid_embeddings = model.encode(valid_texts, show_progress_bar=True, convert_to_numpy=True)
        # Insert valid embeddings into the corresponding positions in the placeholder array
        for idx, embedding in zip(valid_indices, valid_embeddings):
            embeddings[idx] = embedding

    return embeddings


def serialize_embedded_education_info(row):
    # Convert each numpy array in the tuple to a list
    return [(level, emb.tolist()) if isinstance(emb, np.ndarray) else (level, emb) for level, emb in row]
    

def main(args):
    # Ensure the output directories exist
    os.makedirs(args.embeddings_output_folder, exist_ok=True)
    os.makedirs(args.datasets_output_folder, exist_ok=True)

    # Initialize empty DataFrames for datasets
    successful_company_and_founder_profiles_dataset = pd.DataFrame()
    unsuccessful_company_and_founder_profiles_dataset = pd.DataFrame()

    # Load data from CSV files specified in command line arguments
    successful_company_and_founder_profiles_df = pd.read_csv(args.successful_profiles_filepath)
    unsuccessful_company_and_founder_profiles_df = pd.read_csv(args.unsuccessful_profiles_filepath)

    # Apply functions to extract majors, employment history, profile descriptions and company description from the successful dataset
    successful_company_and_founder_profiles_dataset['education_info'] = successful_company_and_founder_profiles_df.apply(extract_education_info, axis=1)
    successful_company_and_founder_profiles_dataset['majors'] = successful_company_and_founder_profiles_df.apply(extract_majors_from_row, axis=1)
    successful_company_and_founder_profiles_dataset['top_university'] = successful_company_and_founder_profiles_df.apply(extract_top_university_flags, axis=1)
    successful_company_and_founder_profiles_dataset['employment_history'] = successful_company_and_founder_profiles_df.apply(extract_employments_from_row, axis=1)
    successful_company_and_founder_profiles_dataset['profile_description'] = successful_company_and_founder_profiles_df.apply(extract_longer_description_from_row, axis=1)   
    successful_company_and_founder_profiles_dataset['company_description'] = successful_company_and_founder_profiles_df.apply(extract_longer_company_description, axis=1) 

    # Apply functions to extract majors, employment history, profile descriptions and company description from the unsuccessful dataset
    unsuccessful_company_and_founder_profiles_dataset['education_info'] = unsuccessful_company_and_founder_profiles_df.apply(extract_education_info, axis=1)
    unsuccessful_company_and_founder_profiles_dataset['majors'] = unsuccessful_company_and_founder_profiles_df.apply(extract_majors_from_row, axis=1)
    unsuccessful_company_and_founder_profiles_dataset['top_university'] = unsuccessful_company_and_founder_profiles_df.apply(extract_top_university_flags, axis=1)
    unsuccessful_company_and_founder_profiles_dataset['employment_history'] = unsuccessful_company_and_founder_profiles_df.apply(extract_employments_from_row, axis=1)
    unsuccessful_company_and_founder_profiles_dataset['profile_description'] = unsuccessful_company_and_founder_profiles_df.apply(extract_longer_description_from_row, axis=1) 
    unsuccessful_company_and_founder_profiles_dataset['company_description'] = unsuccessful_company_and_founder_profiles_df.apply(extract_longer_company_description, axis=1) 

    # Load SentenceTransformer models for embedding generation
    major_model = SentenceTransformer("flax-sentence-embeddings/multi-QA_v1-mpnet-asymmetric-A")
    general_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Determine the number of rows in each dataset to ensure correct embedding dimensions
    num_rows_successful = successful_company_and_founder_profiles_dataset.shape[0]
    num_rows_unsuccessful = unsuccessful_company_and_founder_profiles_dataset.shape[0]

    # Generate and save embeddings for majors, using a model specifically chosen for this purpose
    majors = list(
        set([item for sublist in successful_company_and_founder_profiles_dataset['majors'] for item in sublist if item is not None]) | \
        set([item for sublist in unsuccessful_company_and_founder_profiles_dataset['majors'] for item in sublist if item is not None])
    )
    majors_embeddings = generate_embeddings(
        majors,
        major_model,
        len(majors))

    majors_embeddings_dict = {major: embedding for major, embedding in zip(majors, majors_embeddings)}

    successful_company_and_founder_profiles_dataset['embedded_education_info'] = successful_company_and_founder_profiles_dataset['education_info'].apply(lambda x: embed_education_info(x, majors_embeddings_dict))
    unsuccessful_company_and_founder_profiles_dataset['embedded_education_info'] = unsuccessful_company_and_founder_profiles_dataset['education_info'].apply(lambda x: embed_education_info(x, majors_embeddings_dict))
    # Serialize numpy arrays to lists
    successful_company_and_founder_profiles_dataset['embedded_education_info'] = successful_company_and_founder_profiles_dataset['embedded_education_info'].apply(serialize_embedded_education_info)
    unsuccessful_company_and_founder_profiles_dataset['embedded_education_info'] = unsuccessful_company_and_founder_profiles_dataset['embedded_education_info'].apply(serialize_embedded_education_info)


    # Generate and save embeddings for employment history, using a general-purpose SentenceTransformer model
    successful_employment_embeddings = generate_embeddings(
        successful_company_and_founder_profiles_dataset['employment_history'].tolist(),
        general_model,
        num_rows_successful)

    unsuccessful_employment_embeddings = generate_embeddings(
        unsuccessful_company_and_founder_profiles_dataset['employment_history'].tolist(),
        general_model,
        num_rows_unsuccessful)

    # Generate and save embeddings for profile descriptions, also using the general-purpose model
    successful_profile_descriptions_embeddings = generate_embeddings(
        successful_company_and_founder_profiles_dataset['profile_description'].tolist(),
        general_model,
        num_rows_successful)

    unsuccessful_profile_descriptions_embeddings = generate_embeddings(
        unsuccessful_company_and_founder_profiles_dataset['profile_description'].tolist(),
        general_model,
        num_rows_unsuccessful)

    # Generate and save embeddings for profile descriptions, also using the general-purpose model
    successful_company_descriptions_embeddings = generate_embeddings(
        successful_company_and_founder_profiles_dataset['company_description'].tolist(),
        general_model,
        num_rows_successful)

    unsuccessful_company_descriptions_embeddings = generate_embeddings(
        unsuccessful_company_and_founder_profiles_dataset['company_description'].tolist(),
        general_model,
        num_rows_unsuccessful)


    if args.debug:
        successful_company_and_founder_profiles_dataset = successful_company_and_founder_profiles_dataset
        unsuccessful_company_and_founder_profiles_dataset = unsuccessful_company_and_founder_profiles_dataset
        print(successful_company_and_founder_profiles_dataset)
        print()
        print(unsuccessful_company_and_founder_profiles_dataset)

    # Specify the file paths for the exported CSV files
    successful_export_filename = 'extracted_info_successful_company_and_founder_profiles_dataset.csv'
    unsuccessful_export_filename = 'extracted_info_unsuccessful_company_and_founder_profiles_dataset.csv'

    # Export the successful company and founder profiles dataset to CSV
    successful_company_and_founder_profiles_dataset.to_csv(os.path.join(args.datasets_output_folder, successful_export_filename), index=False)

    # Export the unsuccessful company and founder profiles dataset to CSV
    unsuccessful_company_and_founder_profiles_dataset.to_csv(os.path.join(args.datasets_output_folder, unsuccessful_export_filename), index=False)

    print(f"Successful profiles exported to {os.path.join(args.datasets_output_folder, successful_export_filename)}")
    print(f"Unsuccessful profiles exported to {os.path.join(args.datasets_output_folder, unsuccessful_export_filename)}")

    # Demonstrate the shape of the generated embeddings for verification
    print(majors_embeddings.shape)
    print(successful_employment_embeddings.shape, unsuccessful_employment_embeddings.shape)
    print(successful_profile_descriptions_embeddings.shape, unsuccessful_profile_descriptions_embeddings.shape)
    print(successful_profile_descriptions_embeddings.shape, unsuccessful_profile_descriptions_embeddings.shape)

    # Export the generated embeddings to .npy files within the specified output directory
    np.save(os.path.join(args.embeddings_output_folder, 'majors_embeddings.npy'), majors_embeddings)
    np.save(os.path.join(args.embeddings_output_folder, 'successful_employment_embeddings.npy'), successful_employment_embeddings)
    np.save(os.path.join(args.embeddings_output_folder, 'unsuccessful_employment_embeddings.npy'), unsuccessful_employment_embeddings)
    np.save(os.path.join(args.embeddings_output_folder, 'successful_profile_descriptions_embeddings.npy'), successful_profile_descriptions_embeddings)
    np.save(os.path.join(args.embeddings_output_folder, 'unsuccessful_profile_descriptions_embeddings.npy'), unsuccessful_profile_descriptions_embeddings)
    np.save(os.path.join(args.embeddings_output_folder, 'successful_company_descriptions_embeddings.npy'), successful_company_descriptions_embeddings)
    np.save(os.path.join(args.embeddings_output_folder, 'unsuccessful_company_descriptions_embeddings.npy'), unsuccessful_company_descriptions_embeddings)


if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Generate and export embeddings for company and founder profiles.')

    # Add arguments to the parser
    parser.add_argument('--successful_profiles_filepath', type=str, required=True, help='Path to the CSV file with successful company and founder profiles.')
    parser.add_argument('--unsuccessful_profiles_filepath', type=str, required=True, help='Path to the CSV file with unsuccessful company and founder profiles.')
    parser.add_argument('--datasets_output_folder', type=str, default='data', help='Folder to save the extracted info datasets.')
    parser.add_argument('--embeddings_output_folder', type=str, default='embeddings', help='Folder to save the embeddings.')
    parser.add_argument('--debug', type=int, default=0, help='Debug parameter')

    # Parse the arguments
    args = parser.parse_args()

    main(args)

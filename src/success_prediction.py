import argparse
import ast
import json
import logging
import os

import concurrent.futures

import numpy as np
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score


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

embedding_file_names = [
    'successful_profile_descriptions_embeddings.npy',
    'unsuccessful_profile_descriptions_embeddings.npy',
    'successful_employment_embeddings.npy',
    'unsuccessful_employment_embeddings.npy',
    'successful_company_descriptions_embeddings.npy',
    'unsuccessful_company_descriptions_embeddings.npy'
]


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

    # Load and stack embeddings
    embeddings = load_and_stack_embeddings(embedding_file_names, args.embeddings_dir)

    founder_segmentation_results_df = pd.read_csv(args.founder_segmentation_results)

    return extracted_info_combined_df, preprocessed_combined_df, embeddings, founder_segmentation_results_df

class FounderDataset(Dataset):
    def __init__(self, extracted_info_df, preprocessed_df, embeddings_dict, segmentation_df):
        self.extracted_info_df = extracted_info_df
        self.preprocessed_df = preprocessed_df
        self.embeddings_dict = embeddings_dict
        self.segmentation_df = segmentation_df

    def __len__(self):
        return len(self.extracted_info_df)

    def __getitem__(self, idx):
        extracted_data = self.extracted_info_df.iloc[idx]
        profile_data = self.preprocessed_df.iloc[idx]
        segmentation_data = self.segmentation_df.iloc[idx]
        embeddings = {key: value[idx] for key, value in self.embeddings_dict.items()}

        input_profile_dict = {
            'level': [np.float64(segmentation_data['level'][-1])],
            'top_university': [np.float64(extracted_data['top_university'])],
            'profile_descriptions_embeddings': embeddings['profile_descriptions_embeddings'],
            'employment_embeddings': embeddings['employment_embeddings'],
        }
        input_profile_features = np.concatenate(list(input_profile_dict.values()))

        return idx, profile_data['founder_linkedin_url'], segmentation_data['level'], profile_data['source'], input_profile_features

def collate_fn(batch):
    # summaries, profiles, extracted_data, embeddings = zip(*batch)
    indices, urls, levels, sources, input_profile_features = zip(*batch)
    profile_batch = {
        'indices': indices,
        'urls': urls,
        'levels': levels,
        'sources': sources,
        'input_profile_features': input_profile_features,
    }
    return profile_batch


def main(args):
    #
    logger.info("Success Prediction")
    logger.info("\nargs: " + json.dumps(vars(args), indent=4))

    # Ensure the output directories exist
    os.makedirs(args.datasets_output_folder, exist_ok=True)

    # Load data
    logger.info("\nLoading data ...")
    extracted_info_combined_df, preprocessed_combined_df, embeddings, founder_segmentation_results_df = load_data_and_embeddings(args)
    logger.info("Loading data completed.\n")

    logger.info(f"\nSeed set to {args.seed} \n")
    torch.manual_seed(args.seed)

    dataset = FounderDataset(extracted_info_combined_df, preprocessed_combined_df, embeddings, founder_segmentation_results_df)
    dataloader = DataLoader(dataset, batch_size=len(extracted_info_combined_df), shuffle=True, collate_fn=collate_fn)

    results = []  # List to store tuples of (url, level, source)
    original_indices = []  # List to store original indices

    for batch_idx, batch in enumerate(dataloader):
        indices = batch['indices']
        urls = batch['urls']
        levels = batch['levels']
        sources = batch['sources']
        input_features = batch['input_profile_features']

        labels = [1.0 if x == 'Successful' else 0.0 for x in sources]
        X_train, X_test, y_train, y_test = train_test_split(input_features, labels, test_size=0.2, shuffle=False)

        # Initialize random forest classifier
        # clf = RandomForestClassifier(n_estimators=1000, random_state=args.seed)
        clf = LogisticRegression()
        # clf = GradientBoostingClassifier(random_state=args.seed)
        # clf = SVC(probability=True)
        # clf = KNeighborsClassifier()

        # Fit the model on training data
        clf.fit(X_train, y_train)

        # Predict on test data
        y_pred = clf.predict(X_test)

        # Predict probabilities for test data
        y_pred_proba = clf.predict_proba(X_test)

        # Calculate accuracy score
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Accuracy on test set: {accuracy}")

        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")

        # Calculate precision
        precision = precision_score(y_test, y_pred)
        logger.info(f"Precision: {precision}")

        # Calculate recall
        recall = recall_score(y_test, y_pred)
        logger.info(f"Recall (Sensitivity): {recall}")

        # Calculate F1 score
        f1 = f1_score(y_test, y_pred)
        logger.info(f"F1 Score: {f1}")

        # Calculate AUC-ROC score
        auc_roc = roc_auc_score(y_test, y_pred_proba[:, 1])
        logger.info(f"AUC-ROC Score: {auc_roc}")

        success_predictions = ['Successful' if x == 1.0 else 'Unsuccessful' for x in y_pred]
        success_rates = y_pred_proba[:, 1]
        for index, url, level, source, success_prediction, success_rate in zip(indices[320:], urls[320:], levels[320:], sources[320:], success_predictions, success_rates):
            original_indices.append(index)
            results.append((url, level, source, success_prediction, success_rate))


    # Convert the list of tuples into a DataFrame
    results_df = pd.DataFrame(results, columns=['url', 'level', 'source', 'success_prediction', 'success_rate'])
    results_df.index = original_indices
    results_df = results_df.reindex([*range(len(original_indices))])

    # Save the DataFrame as a CSV file
    results_filename = 'success_prediction_results.csv'
    results_df.to_csv(os.path.join(args.datasets_output_folder, results_filename), index=True)
    print(f"\n\nFounder segmentation results exported to {os.path.join(args.datasets_output_folder, results_filename)}")



if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="FounderSegmentation")
    # parser.add_argument('--extracted_info_successful_path', type=str, required=True, help="Path to the CSV file containing extracted info for successful profiles.")
    # parser.add_argument('--extracted_info_unsuccessful_path', type=str, required=True, help="Path to the CSV file containing extracted info for unsuccessful profiles.")
    # parser.add_argument('--preprocessed_successful_path', type=str, required=True, help="Path to the CSV file containing preprocessed data for successful profiles.")
    # parser.add_argument('--preprocessed_unsuccessful_path', type=str, required=True, help="Path to the CSV file containing preprocessed data for unsuccessful profiles.")
    # parser.add_argument('--embeddings_dir', type=str, required=True, help="Directory containing the embeddings files.")
    # parser.add_argument('--prompts_dir', type=str, required=True, help="Directory containing the prompts files.")
    # parser.add_argument('--founder_segmentation_results', type=str, required=True, help="Path to the CSV file containing segmentation results for profiles.")
    # parser.add_argument('--datasets_output_folder', type=str, default='data', help='Folder to save the extracted info datasets.')
    # parser.add_argument('--batch_size', type=int, default=4, help="Batch size for founder segmentation.")
    # parser.add_argument('--seed', type=int, default=2024, help="Seed for random process in PyTorch.")
    #
    # # Parse the arguments
    # args = parser.parse_args()

    # Provide arguments directly
    args = argparse.Namespace(
        extracted_info_successful_path='../data/summarised_extracted_info_successful_company_and_founder_profiles_dataset.csv',
        extracted_info_unsuccessful_path='../data/summarised_extracted_info_unsuccessful_company_and_founder_profiles_dataset.csv',
        preprocessed_successful_path='../data/preprocessed/successful_company_and_founder_profiles.csv',
        preprocessed_unsuccessful_path='../data/preprocessed/unsuccessful_company_and_founder_profiles.csv',
        embeddings_dir='../embeddings',
        prompts_dir='../prompts',
        founder_segmentation_results='../data/founder_segmentation_results.csv',
        datasets_output_folder='../data',
        batch_size=4,
        seed=2024,
    )

    main(args)

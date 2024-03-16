# Moneyball: Large Language Model-Based Feature Engineering for Predicting Startup Success

## Setting Up the Conda Environment

1. **Install Conda**: Download and install it from [Anaconda's official website](https://www.anaconda.com/products/individual).

2. **Clone the Repository**:

   ```bash
   git clone <url>
   cd Moneyball

3. **Create the Environment**:
   ```bash
   conda env create -f environment.yml
    ```
   
4. **Activate the Environment**:
   ```bash
    conda activate founder-gpt
   

## Run 
```bash
python src/get_employer_info.py --extracted_info_successful_path data/extracted_info_successful_company_and_founder_profiles_dataset.csv --extracted_info_unsuccessful_path data/extracted_info_unsuccessful_company_and_founder_profiles_dataset.csv --successful_profiles_filepath data/preprocessed/successful_company_and_founder_profiles.csv --unsuccessful_profiles_filepath data/preprocessed/unsuccessful_company_and_founder_profiles.csv --list_big_tech_path data/Big_Tech_Companies_by_Market_Cap.csv --list_unicorn_path data/List_of_Unicorn_Companies_from_CB_Insights.csv

python src/generate_summaries.py --extracted_info_successful_path data/new_extracted_info_successful_company_and_founder_profiles_dataset.csv --extracted_info_unsuccessful_path data/new_extracted_info_unsuccessful_company_and_founder_profiles_dataset.csv --successful_profiles_filepath data/preprocessed/successful_company_and_founder_profiles.csv --unsuccessful_profiles_filepath data/preprocessed/unsuccessful_company_and_founder_profiles.csv --prompts_dir prompts

python src/founder_segmentation.py --extracted_info_successful_path data/summarised_extracted_info_successful_company_and_founder_profiles_dataset.csv --extracted_info_unsuccessful_path data/summarised_extracted_info_unsuccessful_company_and_founder_profiles_dataset.csv --preprocessed_successful_path data/preprocessed/successful_company_and_founder_profiles.csv --preprocessed_unsuccessful_path data/preprocessed/unsuccessful_company_and_founder_profiles.csv --prompts_dir prompts --batch_size 4 --seed 2024

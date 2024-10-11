# data_preparation.py
import pandas as pd

def prepare_dataset_for_generator(df, indices_csv_path, indices_column_name, text_col, class_col=None, sample_size=100, supp_columns=None, leading_columns=None):
    """
    Prepare a dataset for generator tasks by excluding specific indices, sampling, and optionally concatenating text columns,
    handling empty or NaN values gracefully by skipping them in concatenation.
    """
    # Load indices to exclude from sampling
    exclude_df = pd.read_csv(indices_csv_path)
    exclude_indices = exclude_df[indices_column_name].tolist()

    # Exclude specified indices from the DataFrame
    df_filtered = df.drop(exclude_indices, errors='ignore')

    # Sample the DataFrame
    sampled_df = df_filtered.sample(n=sample_size, replace=False)

    # Initialize combined text with leading columns if present
    if leading_columns:
        sampled_df['combined_text'] = sampled_df[leading_columns].astype(str).agg(lambda x: ' '.join(x.dropna()), axis=1)
        sampled_df['combined_text'] += " " + sampled_df[text_col].astype(str)
    else:
        sampled_df['combined_text'] = sampled_df[text_col].astype(str)

    # Append supplementary columns, skipping blanks and NaNs
    if supp_columns:
        for col in supp_columns:
            sampled_df['combined_text'] += sampled_df[col].astype(str).replace(r'^\s*$', '', regex=True).apply(lambda x: ' ' + x if x != '' else '')

    x_data = sampled_df['combined_text'].tolist()

    # Prepare the data for generator tasks
    data_for_generator = {'x': x_data, 'Index': sampled_df.index.tolist()}
    if class_col:
        data_for_generator['y'] = sampled_df[class_col].tolist()

    return data_for_generator

def convert_csv_to_parquet(csv_file_path, parquet_file_path, **kwargs):
    df = pd.read_csv(csv_file_path)
    data_for_generator = prepare_dataset_for_generator(df, **kwargs)
    df_to_store = pd.DataFrame(data_for_generator)
    df_to_store.to_parquet(parquet_file_path)

import pandas as pd
import pgeocode

class DataManager:
    """
    A utility class designed to handle common data loading and merging operations.

    This class provides static methods to load data from CSV files, enrich data
    with geographical information based on postal codes, and merge columns
    from another DataFrame. Like DataMetrics, it uses static methods, meaning
    no instance of the class is needed to use its functions.
    """
    @staticmethod
    def load_csv(path_to_csv:str,verbose:int=0) -> pd.DataFrame:
       """
        Loads data from a CSV file into a pandas DataFrame.

        Args:
            path_to_csv (str): The full or relative path to the CSV file.
            verbose (int, optional): If > 0, prints a loading message. Defaults to 0.

        Returns:
            pd.DataFrame: The DataFrame loaded from the CSV file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            pd.errors.EmptyDataError: If the specified file is empty.
            pd.errors.ParserError: If the specified file cannot be parsed.
            # Other potential exceptions from pd.read_csv
        """
       if verbose:
           print(f"DataManager::load_csv -> Load csv file : {path_to_csv}")
       return pd.read_csv(path_to_csv)
    
    @staticmethod
    def get_lat_lng_for_zipcode(df,verbose=0):
        """
        Enriches the input DataFrame by adding 'zipcode_Latitude' and
        'zipcode_Longitude' columns based on the 'postCode' column
        using the pgeocode library and Nominatim service for Belgium.

        Assumes the DataFrame contains a 'postCode' column.

        Args:
            df (pd.DataFrame): The input DataFrame with a 'postCode' column.
            verbose (int, optional): If > 0, prints a message about fetching data.
                                     Defaults to 0.

        Returns:
            pd.DataFrame: The DataFrame with added 'zipcode_Latitude' and
                          'zipcode_Longitude' columns.

        Dependencies:
            - pgeocode: Used to query geographical information for postal codes.
                        Specifically configured for Belgium ('BE').
        """
        if verbose:
            print(f"DataManager::get_lat_lng_for_zipcode -> Get lat/Lng for poastal code from Nominatim")
        
        # Initialize Nominatim for Belgium
        nomi = pgeocode.Nominatim('BE')
        # Call Nominatim API and extract lat/lng to dataframe columns
        df["zipcode_Latitude"] = (nomi.query_postal_code(list(map(str,df["postCode"].tolist()))).latitude)
        df["zipcode_Longitude"] = (nomi.query_postal_code(list(map(str,df["postCode"].tolist()))).longitude)
        return df
    
    @staticmethod   
    def merge_columnsFrom(main_df,path_to_csv,id_col,from_id_col,from_columns_to_merge,verbose=0):
        """
        Loads a secondary dataset from a CSV, selects specified columns, and
        merges them into the main DataFrame based on a common ID column.

        Useful for combining data from different sources based on a shared key.

        Args:
            main_df (pd.DataFrame): The primary DataFrame to merge into.
            path_to_csv (str): The path to the CSV file containing the secondary data.
            id_col (str): The name of the ID column in the `main_df` to use for merging.
            from_id_col (str): The name of the ID column in the secondary CSV file
                               to use for merging.
            from_columns_to_merge (list): A list of column names (strings) from the
                                         secondary CSV file that should be merged
                                         into the `main_df`. (Excluding the ID column).
            verbose (int, optional): If > 0, prints messages about the merge process.
                                     Defaults to 0.

        Returns:
            pd.DataFrame: The `main_df` with the specified columns from the secondary
                          dataset merged in.

        Raises:
            FileNotFoundError: If the specified CSV file (`path_to_csv`) does not exist.
            KeyError: If `id_col`, `from_id_col`, or any column in `from_columns_to_merge`
                      is not found in the respective DataFrames.
            # Other potential exceptions from pd.read_csv or pd.merge
        """
        if verbose:
            print(f"DataManager::merge_columnsFrom -> Columns to merge : {from_columns_to_merge}")
            print(f"DataManager::merge_columnsFrom -> columns before merge : {main_df.columns.to_list}")
        main_df[id_col] = pd.to_numeric(main_df[id_col], errors='coerce').astype('Int64')

        from_df = pd.read_csv(path_to_csv)
        from_df = from_df[[from_id_col]+from_columns_to_merge]
        from_df = from_df.rename(columns={from_id_col: 'from_id'})
        from_df = from_df.drop_duplicates(subset=['from_id'], keep='first')
      
        main_df = main_df.merge(from_df, left_on=id_col, right_on='from_id', how='left')
        main_df = main_df.drop(columns=['from_id'])

        if verbose:
            print(f"DataManager::merge_columnsFrom -> columns merge successfully: {main_df.columns.to_list}")

        return main_df

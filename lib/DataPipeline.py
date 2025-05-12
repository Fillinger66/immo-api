from pandas import DataFrame
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from lib.DataClustering import DataClustering
import pickle

class PipelineNotInitialized(Exception):
    """Custom exception for uninitialized pipeline."""
    pass

class DataPipeline:
    """
    A class designed to handle the end-to-end data processing pipeline
    for a machine learning task, likely involving real estate or property data
    based on the column names and transformations.

    It encapsulates steps such as data loading (implicitly via constructor),
    cleaning, handling missing values, feature transformation, splitting data,
    and saving/loading the trained preprocessing pipeline.

    Attributes:
        df (pd.DataFrame): The pandas DataFrame containing the data to be processed.
        target_columns_name (str): The name of the target variable column (e.g., 'price').
        path_to_pipeline_file (str): The file path to save/load the scikit-learn pipeline.
        verbose (int): Controls the verbosity of logging messages. 0 for silent,
                       higher values for more messages.
    """

    def __init__(self,df:DataFrame,target_columns_name:str,path_to_save_pipeline:str="pipeline/xgboost_pipeline.pipeline",verbose:int=0):
        """
        Initializes the DataPipeline with the input DataFrame, target column name,
        and an optional path for saving/loading the pipeline.

        Args:
            df (pd.DataFrame): The input pandas DataFrame.
            target_columns_name (str): The name of the target column.
            path_to_save_pipeline (str): The default path for the pipeline file.
            verbose (int): Verbosity level (0 for silent).
        """
        self.df = df
        self.target_columns_name = target_columns_name
        self.path_to_pipeline_file = path_to_save_pipeline
        self.verbose = verbose


    def log(self,text):
        if text:
            print(f"DataPipeline::"+text)

    def load_pipeline(self):
        """
        Load a previously saved pipeline from a file using Python's pickle module.
        Ensures that the file and its directory exist before attempting to load.

        Parameters:
            path_to_pipeline (str): The file path from which to load the pipeline.

        Raises:
            FileNotFoundError: If the specified file or its directory does not exist.
        """
        pipeline = None
        # --- 1. Check if directory exists ---
        dir_path = os.path.dirname(self.path_to_pipeline_file)
        if dir_path and not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory does not exist: {dir_path}")

        # --- 2. Check if model file exists ---
        if not os.path.isfile(self.path_to_pipeline_file):
            raise FileNotFoundError(f"Model file not found: {self.path_to_pipeline_file}")

        # --- 3. Load the model using pickle ---
        with open(self.path_to_pipeline_file, 'rb') as file:
            pipeline = pickle.load(file)
        if self.verbose:
            self.log(f"Pipeline loaded successful from {dir_path}")
        return pipeline

    def save_pipeline(self,pipeline):
        """
        Saves the pipeline to a file using Python's pickle module.
        Ensures the target directory exists before saving.

        Parameters:
            path_to_pipeline (str): Full file path where the model will be saved.
            pipeline (object, optional): The model object to save. If None, saves self.model.

        Raises:
            ModelNotInitialized: If no model is provided and self.model is not set.
        """

        # --- 1. Ensure the directory exists ---
        # Extract directory from full path
        dir_path = os.path.dirname(self.path_to_pipeline_file)
        if self.verbose:
            self.log(f"Extracted path : {dir_path}")
        # If directory path is not empty and doesn't exist, create it
        if dir_path and not os.path.exists(dir_path):
            self.log(f"directory path doesn't exist, create it : {dir_path}")
            os.makedirs(dir_path, exist_ok=True)  # Create intermediate directories if needed

        # --- 2. Save the pipeline using pickle ---
        with open(self.path_to_pipeline_file, 'wb') as file:
            if pipeline:
                pickle.dump(pipeline, file)  # Save the provided pipeline
            else:
                # Raise an error if neither model is provided nor internal model is set
                raise PipelineNotInitialized("ERROR -> model is not initialized")
        self.log(f"Pipeline saved successful under : {self.path_to_pipeline_file}")

    def remove_unrevelant_columns(self,unrevelant_columns:list):
        """
        Removes specified irrelevant columns from the DataFrame.

        This method modifies the DataFrame stored in `self.df` by dropping
        the columns whose names are provided in the `unrevelant_columns` list.
        This is typically done as an initial cleaning step in a data processing
        pipeline to remove features that are not needed for analysis or modeling.

        Args:
            unrevelant_columns (list): A list of column names (strings) to be
                                    removed from the DataFrame.

        """
        self.df = self.df.drop(columns=unrevelant_columns)

    def dropna_target(self,target_columns:str):
        """
        Remove rows without value for target columns
        """
        self.df = self.df.dropna(subset=[target_columns])

    def dropna_for_columns(self,columns):
        """
        Drops rows from the DataFrame that have missing values (NaN) in specified columns.

        This method modifies the DataFrame stored in `self.df` by removing any rows
        where *at least one* of the columns listed in the `columns` argument has a
        missing value. This is useful for ensuring that essential features used
        later in the pipeline are complete.

        Args:
            columns (list): A list of column names (strings). Rows with NaN values
                            in any of these columns will be dropped.

        """
        self.df = self.df.dropna(subset=columns)

    def transform_boolean_to_int(self):
        boolean_features = ['hasAttic', 'hasBasement', 'hasDressingRoom', 'hasDiningRoom', 'hasLift',
                    'hasHeatPump', 'hasPhotovoltaicPanels', 'hasThermicPanels',
                    'hasLivingRoom', 'hasGarden', 'hasAirConditioning',
                    'hasArmoredDoor', 'hasVisiophone', 'hasOffice', 'hasSwimmingPool',
                    'hasFireplace', 'hasTerrace']
        boolean_features = self.df.select_dtypes(include=["bool"]).columns.tolist()
        # Assurez-vous que toutes les colonnes booléennes sont dans le dataframe
        boolean_features = [col for col in boolean_features if col in self.df.columns]

        # Convertir les caractéristiques booléennes en entiers (0 ou 1) et remplir les NaN avec False (0)
        for col in boolean_features:
            self.df[col].astype(int)
            self.df[col] = self.df[col].dropna()
        
        return boolean_features

    def manage_missing_values(self,df):
        """
        Manages missing values in specific columns of the input DataFrame.

        This function applies various strategies to fill missing (NaN) values
        in different property-related columns based on assumptions about their
        meaning (e.g., filling counts/boolean flags with 0 or 1).

        Args:
            df (pd.DataFrame): The input pandas DataFrame containing property data.

        Returns:
            pd.DataFrame: The DataFrame with missing values handled in the specified columns.
                        Note: The function modifies the DataFrame in place as well.
        """
        df.loc[(df['hasTerrace'] != 1) & (df['terraceSurface'].isna()), 'terraceSurface'] = 0

        df["terraceSurface"] = df["terraceSurface"].fillna(0)

        df['bedroomCount'] = df['bedroomCount'].fillna(1)
        df['bathroomCount'] = df['bathroomCount'].fillna(1)
        
        df["hasHeatPump"] = df["hasThermicPanels"].fillna(0)
        df["hasPhotovoltaicPanels"] = df["hasThermicPanels"].fillna(0)
        df["hasThermicPanels"] = df["hasThermicPanels"].fillna(0)
        df["hasAirConditioning"] = df["hasAirConditioning"].fillna(0)
        df["hasArmoredDoor"] = df["hasArmoredDoor"].fillna(0)
        df["hasVisiophone"] = df["hasVisiophone"].fillna(0)
        df["hasOffice"] = df["hasOffice"].fillna(0)
        df["hasLift"] = df["hasLift"].fillna(0)
        
        #df["facedeCount"] = df["facedeCount"].fillna(2)


        df["buildingConstructionYear"] = df["buildingConstructionYear"].fillna(0)
        df["parkingCountOutdoor"] = df["parkingCountOutdoor"].fillna(0)

        return df
    
    def transform_label_data(self,df):
        """Transform the label data to match the main dataframe."""
        # Filter only valid EPC scores 
        epc_order = ['A++', 'A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
        df = df[df['epcScore'].isin(epc_order)].copy()

        # Map EPC scores to ordinal values using .loc
        df.loc[:, "epcScore"] = df["epcScore"].map({
            'A++': 8,
            'A+': 7,
            'A': 6,
            'B': 5,
            'C': 4,
            'D': 3,
            'E': 2,
            'F': 1,
            'G': 0
        })

        # Fill any potential missing values and convert to int
        df.loc[:, "epcScore"] = df["epcScore"].infer_objects(copy=False).fillna(-1).astype(int)
        
        # floodZoneType lts assume that missing values are NON_FLOOD_ZONE
        df.loc[:, "floodZoneType"] = df["floodZoneType"].fillna("NON_FLOOD_ZONE")
        df.loc[:, "floodZoneType"] = df["floodZoneType"].map({
            'NON_FLOOD_ZONE': 0,
            'CIRCUMSCRIBED_FLOOD_ZONE': 1,
            'CIRCUMSCRIBED_WATERSIDE_ZONE': 1,
            'POSSIBLE_FLOOD_ZONE': 1,
            'POSSIBLE_N_CIRCUMSCRIBED_FLOOD_ZONE': 1,
            'POSSIBLE_N_CIRCUMSCRIBED_WATERSIDE_ZONE': 1,
            'RECOGNIZED_FLOOD_ZONE': 1,
            'RECOGNIZED_N_CIRCUMSCRIBED_FLOOD_ZONE': 1,
            'RECOGNIZED_N_CIRCUMSCRIBED_WATERSIDE_FLOOD_ZONE': 1
        }).astype(int)
       
        # Keep only rows with known building condition and make an explicit copy
        building_condition_text = [
            'AS_NEW', 'JUST_RENOVATED', 'GOOD', 'TO_BE_DONE_UP',
            'TO_RESTORE', 'TO_RENOVATE', 'NOT_MENTIONED'
        ]
        df = df[df['buildingCondition'].isin(building_condition_text)].copy()

        # Map text labels to integers safely
        df.loc[:, "buildingCondition"] = df["buildingCondition"].map({
            'AS_NEW': 6,
            'JUST_RENOVATED': 5,
            'GOOD': 4,
            'TO_BE_DONE_UP': 3,
            'TO_RESTORE': 2,
            'TO_RENOVATE': 1,
            'NOT_MENTIONED': -1
        })

        # Fill any remaining NAs just in case and convert to int
        df.loc[:, "buildingCondition"] = df["buildingCondition"].infer_objects(copy=False).fillna(-1).astype(int)

        # Filter kitchen types and make an explicit copy
        kitchen_type_text = [
            'HYPER_EQUIPPED', 'USA_HYPER_EQUIPPED', 'INSTALLED', 'USA_INSTALLED',
            'SEMI_EQUIPPED', 'USA_SEMI_EQUIPPED', 'NOT_INSTALLED', 'USA_UNINSTALLED'
        ]
        df = df[df['kitchenType'].isin(kitchen_type_text)].copy()

        # Map kitchen types to ordinal integers
        df.loc[:, "kitchenType"] = df["kitchenType"].map({
            'HYPER_EQUIPPED': 3,
            'USA_HYPER_EQUIPPED': 4,
            'INSTALLED': 2,
            'USA_INSTALLED': 3,
            'SEMI_EQUIPPED': 1,
            'USA_SEMI_EQUIPPED': 2,
            'NOT_INSTALLED': 0,
            'USA_UNINSTALLED': 0
        })

        # Fill any missing values and convert to int
        df.loc[:, "kitchenType"] = df["kitchenType"].infer_objects(copy=False).fillna(-1).astype(int)
        print(df[["kitchenType","buildingCondition","floodZoneType","epcScore"]].head(10))
        return df

    def execute_pipeline(self,unreavelant_columns,k=2,debug=False) -> list:
        """
        Executes a complete data processing and splitting pipeline.

        This method orchestrates a series of steps to clean, transform, and prepare
        the dataset for machine learning. It handles missing values, filters outliers,
        potentially engineers new features (though some steps are commented out),
        and prepares the data for model training and evaluation using scikit-learn
        preprocessing and splitting utilities.

        Steps:
            1. Remove irrelevant columns.
            2. Drop rows with missing target values ('price').
            3. Drop rows with missing values in specific essential columns.
            4. Filter out potential price outliers.
            5. Run pipeline and get splitted data into train/test
            6. Return processed training/testing features, original training/testing target, and initial feature list.

        Parameters:
            unreavelant_columns (list): List of column names to remove.
            k (int): Parameter for the clustering step (passed to clusterize_zipcode_sqrt_meter).
            debug (bool): If True, prints shapes of processed data.

        Returns:
            list: [X_train_processed, X_test_processed, y_train, y_test, initial_features_list]
                  - X_train_processed (np.ndarray): Processed training features.
                  - X_test_processed (np.ndarray): Processed testing features.
                  - y_train (pd.Series): Original training target.
                  - y_test (pd.Series): Original testing target.
                  - initial_features_list (list): List of feature names before processing.
        """
        # --- Step 1: Remove irrelevant columns ---
        # Calls a helper method to remove columns specified in unreavelant_columns list.
        self.remove_unrevelant_columns(unreavelant_columns)

        print(f"{self.df.head(10)}")

        # --- Step 2: Drop rows with missing target values ('price') ---
        # Calls a helper method to remove rows where the target column has NaN.
        self.dropna_target(self.target_columns_name)

        # --- Step 3: Drop rows with missing values in specific essential columns ---
        # Removes rows that still have NaNs in critical columns required for
        # subsequent steps or that are deemed essential features.
        self.dropna_for_columns(['bedroomCount','habitableSurface','terraceSurface',"postCode","buildingConstructionYear"])

        # --- Step 4: Filter out potential price outliers ---
        # Removes rows where the price is above a certain threshold (1,000,000).
        # This is a form of outlier removal based on the target variable.
        print(f"DataPipeline::execute_pipeline -> Filtering prices > 1,000,000...")
        initial_rows = len(self.df)
        self.df = self.df[(self.df['price'] <= 1000000) & (self.df['price'] > 50000)]
        print(f"DataPipeline::execute_pipeline -> Filtered {initial_rows - len(self.df)} rows. New shape: {self.df.shape}")

        print(self.df.columns.to_list)
        # --- Step 5: Run pipeline and get splitted data into train/test ---
        X_train_processed, X_test_processed, y_train, y_test, features_list = self.run_pipeline(debug)
        

        # --- Step 6 : Return processed data and initial features list ---
        # Returns the processed feature arrays (NumPy arrays), the original target Series,
        # and the list of feature names from the DataFrame BEFORE processing.
        # Note: features_list contains names before ColumnTransformer.
        # Getting names AFTER ColumnTransformer requires more complex logic (as shown previously).
        return [X_train_processed, X_test_processed, y_train, y_test, features_list]


    def create_pipeline(self):
        """
        Creates a scikit-learn preprocessing pipeline using ColumnTransformer.

        This pipeline is designed to handle numerical and categorical features
        separately, applying appropriate imputation and scaling/encoding steps
        before training a machine learning model.

        The function identifies numerical and categorical columns, defines
        preprocessing steps for each type using scikit-learn Pipelines,
        combines these steps with a ColumnTransformer, and returns the
        complete preprocessing pipeline.

        Returns:
            sklearn.pipeline.Pipeline: A scikit-learn Pipeline object containing
                                    the configured ColumnTransformer as the
                                    first step.
        """
         # --- Step 1: Identify column types ---
        # Separates columns into lists based on their data type after previous cleaning/transformations.
        # The target column is explicitly excluded from the numerical features list.
        if self.target_columns_name in self.df.columns:
            numeric_cols = self.df.select_dtypes(include=["float64", "int64"]).drop(columns=[self.target_columns_name]).columns.tolist()
        else:
            numeric_cols = self.df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        # Includes object (string) and boolean types in categorical columns.
        # Note: If previous steps converted some original categoricals to numerical (like 0/1),
        # they will end up in numeric_cols.
        categorical_cols = self.df.select_dtypes(include=["object", "bool"]).columns.tolist()

        print(f"\nDataPipeline::create_pipeline -> Identified {len(numeric_cols)} numerical columns and {len(categorical_cols)} categorical columns.")
        # print(f"DataPipeline::create_pipeline -> Numerical columns: {numeric_cols[:5]}...") # Print first few
        # print(f"DataPipeline::create_pipeline -> Categorical columns: {categorical_cols[:5]}...") # Print first few


        # --- Step 2: Prepare scikit-learn transformers ---
        # Defines preprocessing steps for different column types using Pipelines.
        # Numerical Transformer: Impute missing numerical values with the mean, then scale features to have zero mean and unit variance.
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        # Categorical Transformer: Impute missing categorical values with the most frequent value, then apply Ordinal Encoding.
        # OrdinalEncoder converts categories to integers. handle_unknown="use_encoded_value" and unknown_value=-1
        # handle categories seen during testing but not training.
        # Note: OneHotEncoder and TopKOneHotEncoder are commented out, meaning OrdinalEncoder is currently active.
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ])

        print("DataPipeline::create_pipeline -> Numerical and Categorical transformers defined.")

        # --- Step 3: Prepare the ColumnTransformer ---
        # Combines the numerical and categorical transformers and specifies which columns each applies to.
        # Columns not listed in 'transformers' are dropped by default if remainder is not specified or is 'drop'.
        # In this configuration, columns NOT in numeric_cols or categorical_cols will be DROPPED.
        # If you intended to keep other columns (like boolean features already encoded), remainder='passthrough' is needed.
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
                # ('bool', 'passthrough', boolean_features) # This line is commented out
            ],
            # remainder='passthrough' # This line is commented out. Columns not in num_cols or cat_cols will be dropped.
        )
        full_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor)
        ])

        print("\nDataPipeline::create_pipeline -> Preprocessing Pipeline created.")
        print("DataPipeline::create_pipeline -> ColumnTransformer defined.")

        

        return full_pipeline
    
    def prepare_data_for_prediction(self):
        """
        Load the pipeline and run the transform on 
        the dataframe to prepare forwarding it to the model
        """
        pipeline = self.load_pipeline()

        X_test_processed = pipeline.transform(self.df)

        return X_test_processed



    def run_pipeline(self,debug=0):
        # --- Step 8: Identify column types ---
        # Separates columns into lists based on their data type after previous cleaning/transformations.
        # The target column is explicitly excluded from the numerical features list.
        numeric_cols = self.df.select_dtypes(include=["float64", "int64"]).drop(columns=[self.target_columns_name]).columns.tolist()
        # Includes object (string) and boolean types in categorical columns.
        # Note: If previous steps converted some original categoricals to numerical (like 0/1),
        # they will end up in numeric_cols.
        categorical_cols = self.df.select_dtypes(include=["object", "bool"]).columns.tolist()

        print(f"\nDatapipeline::run_pipeline -> Identified {len(numeric_cols)} numerical columns and {len(categorical_cols)} categorical columns.")


        # --- Step 9: Prepare scikit-learn transformers ---
        # Defines preprocessing steps for different column types using Pipelines.

        # Numerical Transformer: Impute missing numerical values with the mean, then scale features to have zero mean and unit variance.
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        # Categorical Transformer: Impute missing categorical values with the most frequent value, then apply Ordinal Encoding.
        # OrdinalEncoder converts categories to integers. handle_unknown="use_encoded_value" and unknown_value=-1
        # handle categories seen during testing but not training.
        # Note: OneHotEncoder and TopKOneHotEncoder are commented out, meaning OrdinalEncoder is currently active.
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ])

        print("Datapipeline::run_pipeline -> Numerical and Categorical transformers defined.")

        # --- Step 10: Prepare the ColumnTransformer ---
        # Combines the numerical and categorical transformers and specifies which columns each applies to.
        # Columns not listed in 'transformers' are dropped by default if remainder is not specified or is 'drop'.
        # In this configuration, columns NOT in numeric_cols or categorical_cols will be DROPPED.
        # If you intended to keep other columns (like boolean features already encoded), remainder='passthrough' is needed.
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
                # ('bool', 'passthrough', boolean_features) # This line is commented out
            ],
            # remainder='passthrough' # This line is commented out. Columns not in num_cols or cat_cols will be dropped.
        )

        print("Datapipeline::run_pipeline -> ColumnTransformer defined.")

        # Saving feature names for later use
        features_list = list(self.df.columns)
        
        # --- Step 11: Split data into features (X) and target (y) ---
        # Separates the DataFrame into the feature set (all columns except target) and the target variable.
        X = self.df.drop(columns=[self.target_columns_name])
        y = self.df[self.target_columns_name]

        # --- Step 12: Split data into training and testing sets ---
        # Splits the feature and target data into subsets for training and evaluating the model.
        # test_size=0.2 means 20% of the data is used for testing, 80% for training.
        # random_state=42 ensures the split is the same each time the code is run.
        print(f"\nDatapipeline::run_pipeline -> Splitting data into train (80%) and test (20%)...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Datapipeline::run_pipeline -> Shapes after split: X_train={X_train.shape}, X_test={X_test.shape}, y_train={y_train.shape}, y_test={y_test.shape}")


        # --- Step 13: Create the pipeline with preprocessor ---
        # Creates a simple pipeline that just applies the preprocessor.
        # This is done after splitting because the preprocessor needs to be fitted
        # ONLY on the training data, then used to transform both train and test.
        full_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor)
        ])

        
        print("\nDatapipeline::run_pipeline -> Preprocessing Pipeline created.")

        # --- Step 14: Apply transformers ---
        # Fits the preprocessor on the training data (learning imputation values,
        # encoder mappings, scaler parameters) and then transforms both the training
        # and testing feature sets.
        print("Datapipeline::run_pipeline -> Applying preprocessor to training data (fit_transform)...")
        X_train_processed = full_pipeline.fit_transform(X_train)
        
        print("Datapipeline::run_pipeline -> Applying preprocessor to testing data (transform)...")
        # Use transform, NOT fit_transform, on the test set to apply the transformations
        # learned from the training data.
        X_test_processed = full_pipeline.transform(X_test)

        print(f"Datapipeline::run_pipeline -> save pipeline")
        self.save_pipeline(full_pipeline)


        # --- Step 15: Debugging output ---
        # If debug is True, print the shapes of the processed feature arrays.
        if debug:
            print(f"{'-'*20}\nDatapipeline::run_pipeline -> X_Train_processed shape: {X_train_processed.shape}\n{'-'*20}\nX_test_processed shape: {X_test_processed.shape}")
#            print(X_train[['postCode', 'avg_price_per_sqm', 'cluster_label']].head(5))
        
        return [X_train_processed, X_test_processed, y_train, y_test, features_list]


    def determine_kmean(self,unreavelant_columns):

         # Remove urevelant columns                
        self.remove_unrevelant_columns(unreavelant_columns)
        
        # Drop empty row for the target (price)
        self.dropna_target(self.target_columns_name)

        self.dropna_for_columns(['bedroomCount','habitableSurface','postCode','price'])
        

        cluster = DataClustering()
        cluster.determine_kmean_by_silhouete(self.df)

    def determine_kmean_by_elbow(self,unreavelant_columns):

         # Remove urevelant columns                
        self.remove_unrevelant_columns(unreavelant_columns)
        
        # Drop empty row for the target (price)
        self.dropna_target(self.target_columns_name)

        self.dropna_for_columns(['bedroomCount','habitableSurface','postCode','price'])
        

        cluster = DataClustering()
        cluster.determine_kmean_elbow_method(self.df)

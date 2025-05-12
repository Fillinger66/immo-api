from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class TopKOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn transformer for performing One-Hot Encoding,
    but only on the top K most frequent categories for each feature.

    This is particularly useful for handling categorical features with high
    cardinality (many unique values), where standard One-Hot Encoding would
    create a very large number of new columns, potentially leading to
    increased memory usage and model complexity.

    Categories that are not among the top K for a given feature will effectively
    be grouped together, resulting in a row having 0 across all the one-hot
    encoded columns for that feature.

    Inherits from BaseEstimator and TransformerMixin to be compatible
    with scikit-learn pipelines and utilities.
    """
    def __init__(self, top_k=10):
        """
        Initializes the TopKOneHotEncoder.

        Args:
            top_k (int, optional): The number of top most frequent categories
                                   to encode for each feature. Defaults to 10.
        """
        self.top_k = top_k
        self.top_categories_ = {}
        self.columns_ = []

    def fit(self, X, y=None):
        """
        Learns the top K most frequent categories for each column in the input data.

        This method calculates the frequency of each category in each column
        and identifies the `self.top_k` most frequent ones. These identified
        categories are stored in `self.top_categories_` for use during the
        `transform` step.

        Handles both pandas DataFrames and numpy arrays as input.

        Args:
            X (pd.DataFrame or np.ndarray): The input data containing the
                                          categorical features to encode.
                                          Expected shape (n_samples, n_features).
            y (ignored): Not used in this transformer, included for
                         scikit-learn pipeline compatibility.

        Returns:
            self: The fitted transformer instance.
        """
        # Convert to DataFrame if it's a numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
        else:
            X = X.copy()
        # Get columns list
        self.columns_ = X.columns
        # Get index of the K top values
        for col in self.columns_:
            top = X[col].value_counts().nlargest(self.top_k).index
            self.top_categories_[col] = top
        return self

    def transform(self, X):
        """
        Applies the Top K One-Hot Encoding to the input data.

        For each column processed during the `fit` step, this method creates
        binary columns (0 or 1) for each of the learned top K categories.
        A value of 1 in a new column indicates that the original data had
        that specific top category.

        Handles both pandas DataFrames and numpy arrays as input, ensuring
        that the column names match those seen during `fit` if a numpy array
        is provided.

        Args:
            X (pd.DataFrame or np.ndarray): The input data to transform.
                                          Expected shape (n_samples, n_features)
                                          with the same number of features as the
                                          data used during `fit`.

        Returns:
            pd.DataFrame: A new DataFrame containing the one-hot encoded features
                          for the top K categories. The column names will be
                          in the format "{original_column_name}_{category_value}".
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.columns_)
        else:
            X = X.copy()

        X_encoded = pd.DataFrame(index=X.index)
        
        for col in self.columns_:
            top = self.top_categories_[col]
            for category in top:
                X_encoded[f"{col}_{category}"] = (X[col] == category).astype(int)

        return X_encoded

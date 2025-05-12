from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_error, make_scorer,mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np

class DataMetrics:
    """
    A utility class designed to calculate and optionally print common regression
    evaluation metrics.

    This class is implemented using static methods, meaning you don't need to
    create an instance of the class to use its functionality. It serves as a
    container for functions related to evaluating the performance of regression models.
    """

    @staticmethod
    def get_score(test_labels, predictions,to_print=False):
        """
        Calculates R2 Score, Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE)
        for regression model predictions compared to the actual test labels.

        Optionally prints the calculated metrics in a formatted way.

        Args:
            test_labels (array-like): The true target values for the test dataset.
                                     Expected to be a 1-dimensional array or pandas Series.
            predictions (array-like): The predicted target values from the model.
                                      Expected to be a 1-dimensional array or pandas Series
                                      with the same shape as test_labels.
            to_print (bool, optional): If True, the calculated metrics will be printed
                                       to the console. Defaults to False.

        Returns:
            tuple: A tuple containing the three calculated metrics:
                   (r2, rmse, mae)
                   - r2 (float): The R2 score.
                   - rmse (float): The Root Mean Squared Error.
                   - mae (float): The Mean Absolute Error.

        Dependencies:
            - numpy (as np) for square root calculation (in RMSE).
            - sklearn.metrics.r2_score for R2 calculation.
            - sklearn.metrics.mean_squared_error for MSE calculation (used for RMSE).
            - sklearn.metrics.mean_absolute_error for MAE calculation.
        """
        # 1. Calculate R2 Score
        r2 = r2_score(test_labels, predictions)
        # 2. Calculate RMSE (Root Mean Squared Error)
        rmse = np.sqrt(mean_squared_error(test_labels, predictions))
        # 3. Calculate MAE (Mean Absolute Error)
        mae = mean_absolute_error(test_labels, predictions)
        # --- Affichage des Résultats ---
        if to_print:
            print(f"Model Test R2 Score : {r2:.4f}") 
            print(f"Model Test R2 (en %) : {r2 * 100:.2f} %")
            print(f"Model Test RMSE : {rmse:.2f} €")
            print(f"Model Test MAE : {mae:.2f} €")

        return r2,rmse,mae
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd

def scale_data_with_StandardScaler(X_train, X_test):
    """
    Scales input data using Standard Scaler.
    Returns scaled DataFrames of both input dfs
    """
    # Create the StandardScaler instance
    scaler = StandardScaler()

    # Fit the Standard Scaler with the training data
    scaler.fit(X_train)

    # Scale the training data
    X_train_scaled = scaler.transform(X_train)
    print(f"Scaled X_train data: {X_train_scaled}")

    # Transforming the test dataset based on the fit from the training dataset
    X_test_scaled = scaler.transform(X_test)
    print(f"Scaled X_train data: {X_test_scaled}")

    # Return the scaled data
    return X_train_scaled, X_test_scaled

def logistic_regression_model_generator(X_train, X_test, y_train, y_test, r_state):
    """
    Generates and fits a Logistic Regression model.
    Uses training and testing data passed in as arguments.
    Makes predictions using testing data and prints model accuracy score.
    Does not return anything.
    """
    model = LogisticRegression(random_state=r_state)

    # Train a Logistic Regression model and print the model score
    model.fit(X_train, y_train)
    print(f"Logistic Regression Training Data Score: {model.score(X_train, y_train)}")
    print(f"Logistic Regression Testing Data Score: {model.score(X_test, y_test)}")
    
    # Make and save testing predictions with the saved logistic regression model using the test data
    predictions = model.predict(X_test)
    
    # Review the predictions
    print(f"Logistic Regression Predictions: {predictions}")

    # Calculate the accuracy score by evaluating `y_test` vs. `testing_predictions`
    print(f"Logistic Regression Predictions: {accuracy_score(y_test, predictions)}")

def random_forest_model_generator(X_train, X_test, y_train, y_test, r_state, estimator_count, X_columns):
    """
    Generates and fits a Random Forest model.
    Uses training and testing data passed in as arguments.
    Makes predictions using testing data.
    Prints model accuracy score and top 10 most important features.
    Does not return anything.
    """
    # Create the random forest classifier instance
    model = RandomForestClassifier(n_estimators=estimator_count, random_state=r_state)

    # Fit the model and print the training and testing scores
    model.fit(X_train, y_train)
    print(f"Random Forest Training Data Score: {model.score(X_train, y_train)}")
    print(f"Random Forest Testing Data Score: {model.score(X_test, y_test)}")

    # Make predictions using testing data
    predictions = model.predict(X_test)
    print(f"Random Forest Predictions: {predictions}")

    # Print the accuracy score
    print(f"Random Forest Predictions: {accuracy_score(y_test, predictions)}")
    
    # Get the feature importance array
    importances = model.feature_importances_

    # List the top 10 most important features
    importances_sorted = sorted(zip(model.feature_importances_, X_columns), reverse=True)
    importances_sorted[:10]

if __name__ == "__main__":
    print("This script should not be run directly! Import these functions for use in another file.")


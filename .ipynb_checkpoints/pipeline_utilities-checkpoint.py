from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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

def display_accuracy_scores(model_name, test, predictions):
    """
    Uses model_name, test and predictions passed in as arguments
    Prints different accuracy scores.
    Does not return anything at this time.
    """
    # Calculate the accuracy score by evaluating `test` versus `predictions`
    print(f"{model_name} Predictions Accuracy Score: {accuracy_score(test, predictions)}")
    print(classification_report(test, predictions, labels = [1, 0]))
    print(f"{model_name} Balanced Accuracy Score: {balanced_accuracy_score(test, predictions)}")

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
    #print(f"Logistic Regression Predictions: {predictions}")

    # Print accuracy scores
    display_accuracy_scores("Logistic Regression", y_test, predictions)

    # Print roc_auc_score
    pred_probas = model.predict_proba(X_train)
    pred_probas_firsts = [prob[1] for prob in pred_probas]
    print(f"Logistic Regression roc_auc_score: {roc_auc_score(y_train, pred_probas_firsts)}")

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
    #print(f"Random Forest Predictions: {predictions}")

    # Print accuracy scores
    display_accuracy_scores("Random Forest", y_test, predictions)
    
    # Get the feature importance array
    importances = model.feature_importances_

    # List the top 10 most important features
    importances_sorted = sorted(zip(model.feature_importances_, X_columns), reverse=True)
    print(f"{importances_sorted[:10]}")

def svm_model_generator(X_train, X_test, y_train, y_test, m_type):
    """
    Generates and fits a Support Vector Machine model.
    Uses training and testing data passed in as arguments.
    Makes predictions using testing data and prints model accuracy score.
    Does not return anything.
    """
    model = SVC(kernel=m_type)

    # Train an SVM model and print the model score
    model.fit(X_train, y_train)
    print(f"SVM Training Data Score: {model.score(X_train, y_train)}")
    print(f"SVM Testing Data Score: {model.score(X_test, y_test)}")
    
    # Make and save testing predictions using the test data
    predictions = model.predict(X_test)
    
    # Print accuracy scores
    display_accuracy_scores("SVM", y_test, predictions)

def decision_tree_model_generator(X_train, X_test, y_train, y_test):
    """
    Generates and fits a Decision Tree model.
    Uses training and testing data passed in as arguments.
    Makes predictions using testing data and prints model accuracy score.
    Does not return anything.
    """
    model = tree.DecisionTreeClassifier()

    # Train a Decision Tree model and print the model score
    model.fit(X_train, y_train)
    print(f"Decision Tree Training Data Score: {model.score(X_train, y_train)}")
    print(f"Decision Tree Testing Data Score: {model.score(X_test, y_test)}")
    
    # Make and save testing predictions with the model using the test data
    predictions = model.predict(X_test)
    
    # Print accuracy scores
    display_accuracy_scores("Decision Tree", y_test, predictions)

def gradient_boost_model_generator(X_train, X_test, y_train, y_test, r_state):
    """
    Generates and fits a Gradient Boosting CLassifier model.
    Uses training and testing data and random_state passed in as arguments.
    Makes predictions using testing data and prints model accuracy score.
    Does not return anything.
    """
    model = GradientBoostingClassifier(random_state=r_state)

    # Train a Gradient Boosting model and print the model score
    model.fit(X_train, y_train)
    print(f"Gradient Boosting Training Data Score: {model.score(X_train, y_train)}")
    print(f"Gradient Boosting Testing Data Score: {model.score(X_test, y_test)}")
    
    # Make and save testing predictions using the test data
    predictions = model.predict(X_test)
    
    # Print accuracy scores
    display_accuracy_scores("Gradient Boosting", y_test, predictions)

def ada_boost_model_generator(X_train, X_test, y_train, y_test, r_state):
    """
    Generates and fits an ADA Boosting CLassifier model.
    Uses training and testing data and random_state passed in as arguments.
    Makes predictions using testing data and prints model accuracy score.
    Does not return anything.
    """
    model = AdaBoostClassifier(random_state=r_state)

    # Train an ADA Boosting model and print the model score
    model.fit(X_train, y_train)
    print(f"Ada Boosting Training Data Score: {model.score(X_train, y_train)}")
    print(f"Ada Boosting Testing Data Score: {model.score(X_test, y_test)}")
    
    # Make and save testing predictions using the test data
    predictions = model.predict(X_test)
    
    # Print accuracy scores
    display_accuracy_scores("Ada Boosting", y_test, predictions)

def extra_trees_model_generator(X_train, X_test, y_train, y_test, r_state):
    """
    Generates and fits an Extra Trees Classifier model.
    Uses training and testing data and random_state passed in as arguments.
    Makes predictions using testing data and prints model accuracy score.
    Does not return anything.
    """
    model = ExtraTreesClassifier(random_state=r_state)

    # Train a Extra Trees model and print the model score
    model.fit(X_train, y_train)
    print(f"Extra Trees Training Data Score: {model.score(X_train, y_train)}")
    print(f"Extra Trees Testing Data Score: {model.score(X_test, y_test)}")
    
    # Make and save testing predictions using the test data
    predictions = model.predict(X_test)
    
    # Print accuracy scores
    display_accuracy_scores("Extra Trees", y_test, predictions)

if __name__ == "__main__":
    print("This script should not be run directly! Import these functions for use in another file.")

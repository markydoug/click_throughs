from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import sklearn.preprocessing as pre

###################################################################################
############################ PREP DATA FOR MODELING ###############################
###################################################################################

def scale_data(train, validate, test, target):
    '''
    Takes in train, validate, test and the target variable.
    Returns df with new columns with scaled data for the numeric
    columns besides the target variable
    '''
    scale_features=list(train.select_dtypes(include=np.number).columns)
    scale_features.remove(target)
    
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    minmax = pre.MinMaxScaler()
    minmax.fit(train[scale_features])
    
    train_scaled[scale_features] = pd.DataFrame(minmax.transform(train[scale_features]),
                                                  columns=train[scale_features].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[scale_features] = pd.DataFrame(minmax.transform(validate[scale_features]),
                                               columns=validate[scale_features].columns.values).set_index([validate.index.values])
    
    test_scaled[scale_features] = pd.DataFrame(minmax.transform(test[scale_features]),
                                                 columns=test[scale_features].columns.values).set_index([test.index.values])
    
    return train_scaled, validate_scaled, test_scaled

def get_dumdum(train, validate, test, cols_to_encode):
    '''
    Takes in a dataframe and creates dummy variables for each 
    categorical variable.
    '''

    dummy_train = pd.get_dummies(train[cols_to_encode], dummy_na=False)
    train = pd.concat([train, dummy_train], axis=1)

    dummy_validate = pd.get_dummies(validate[cols_to_encode], dummy_na=False)
    validate = pd.concat([validate, dummy_validate], axis=1)

    dummy_test = pd.get_dummies(test[cols_to_encode], dummy_na=False)
    test = pd.concat([test, dummy_test], axis=1)

    return train, validate, test

def pre_prep(train, validate, test, cols_to_encode, target):
    for col in cols_to_encode:
        train[col] = train[col].astype('category')
        validate[col] = validate[col].astype('category')
        test[col] = test[col].astype('category')
    
    train, validate, test = scale_data(train, validate, test, target)

    train, validate, test = get_dumdum(train, validate, test, cols_to_encode)

    return train, validate, test


def prep_for_model(train, validate, test, target, drivers):
    '''
    Takes in train, validate, and test data frames, the target variable, 
    and a list of the drivers/features we want to model
    It splits each dataframe into X (all variables but target variable) 
    and y (only target variable) for each data frame
    '''

    X_train = train[drivers]
    y_train = train[target]

    X_validate = validate[drivers]
    y_validate = validate[target]

    X_test = test[drivers]
    y_test = test[target]

    return X_train, y_train, X_validate, y_validate, X_test, y_test


###################################################################################
################### MODEL EVALUATION ON TRAIN AND VALIDATE DATA ###################
###################################################################################


def decision_tree_results(X_train, y_train, X_validate, y_validate):
    '''
    Takes in train and validate data and returns decision tree model results
    '''
    # create classifier object
    clf = DecisionTreeClassifier(max_depth=8, random_state=27)

    #fit model on training data
    clf.fit(X_train, y_train)

    #print results
    print("Decision Tree")
    print(f"Train Accuracy: {clf.score(X_train, y_train):.2%}")
    print(f"Validate Accuracy: {clf.score(X_validate, y_validate):.2%}")
    print(f"Difference: {(clf.score(X_train, y_train)-clf.score(X_validate, y_validate)):.2%}")


def random_forest_results(X_train, y_train, X_validate, y_validate):
    '''
    Takes in train and validate data and returns random forest model results
    '''
    # create classifier object
    rf = RandomForestClassifier(max_depth=4, random_state=27)

    #fit model on training data
    rf.fit(X_train, y_train)

    #print results
    print('Random Forest')
    print(f"Train Accuracy: {rf.score(X_train, y_train):.2%}")
    print(f"Validate Accuracy: {rf.score(X_validate, y_validate):.2%}")
    print(f"Difference: {(rf.score(X_train, y_train)-rf.score(X_validate, y_validate)):.2%}")

def knn_results(X_train, y_train, X_validate, y_validate):
    '''
    Takes in train and validate data and returns knn model results
    '''
    # create classifier object
    knn = KNeighborsClassifier()

    #fit model on training data
    knn.fit(X_train, y_train)

    #print results
    print('KNN')
    print(f"Train Accuracy: {knn.score(X_train, y_train):.2%}")
    print(f"Validate Accuracy: {knn.score(X_validate, y_validate):.2%}")
    print(f"Difference: {(knn.score(X_train, y_train)-knn.score(X_validate, y_validate)):.2%}")

def log_results(X_train, y_train, X_validate, y_validate):
    '''
    Takes in train and validate data and returns logistic regression model results
    '''
    # create classifier object
    logit = LogisticRegression(random_state=27)

    #fit model on training data
    logit.fit(X_train, y_train)

    #print results
    print('Logistic Regression')
    print(f"Train Accuracy: {logit.score(X_train, y_train):.2%}")
    print(f"Validate Accuracy: {logit.score(X_validate, y_validate):.2%}")
    print(f"Difference: {(logit.score(X_train, y_train)-logit.score(X_validate, y_validate)):.2%}")

def best_model(X_train, y_train, X_test, y_test):
    '''
    Takes in train and test data and returns random forest model results
    '''
    # create classifier object
    rf = RandomForestClassifier(max_depth=4, random_state=27)

    #fit model on training data
    rf.fit(X_train, y_train)

    #run the best overall model on test data
    print('Best Model: Random Forest')
    print(f"Test Accuracy: {rf.score(X_test, y_test):.2%}")

def best_model_comparison(X_train, y_train, X_validate, y_validate, X_test, y_test):
    '''
    Takes in train, validate and test data and returns random forest model results
    '''
    # create classifier object
    rf = RandomForestClassifier(max_depth=4, random_state=27)

    #fit model on training data
    rf.fit(X_train, y_train)

    #print results
    print('Random Forest')
    print(f"Train Accuracy: {rf.score(X_train, y_train):.2%}")
    print(f"Validate Accuracy: {rf.score(X_validate, y_validate):.2%}")
    print(f"Test Accuracy: {rf.score(X_test, y_test):.2%}")
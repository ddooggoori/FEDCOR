import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import *
from imblearn.over_sampling import *
from imblearn.combine import *
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import *
from tensorflow.keras.metrics import * 
from sklearn.preprocessing import *
from fancyimpute import IterativeImputer
from imblearn.over_sampling import *
from sklearn.utils.class_weight import compute_class_weight
import joblib
import warnings
warnings.filterwarnings('ignore')



def Imputation(train, *args, binary=None, save=False, imputer=None):
    """
    Impute missing values in the dataset.

    Args:
        train (DataFrame): The training dataset.
        args (DataFrame): Additional datasets to apply the same imputation.
        binary (list): List of binary columns.
        save (bool or list): Whether to save the imputer or not.
        imputer (tuple): Tuple containing imputers for continuous and binary variables.

    Returns:
        DataFrame or list of DataFrame: Imputed dataset(s).
    """

    # Extract original variable names
    original_variable = train.columns
    
    # Identify binary columns if not provided
    if binary is None:
        binary = list()
        for i in train.columns:
            if ((train[i].min() == 0) & (train[i].max() == 1)) or (len(train[i].unique()) == 1):
                binary.append(i)
    else:
        binary = binary
    
   
    # Separate binary columns from the dataset
    train_binary = train[binary].reset_index(drop=True)
    
    # Set maximum values for imputation
    max_values = train.max()    
    min_values = train.min()
    max_values[(max_values == 0) & (min_values == 0)] = 1
    
    # Apply imputation if imputer is provided
    if imputer is not None:
        conti_imputer = imputer[0]
        binary_imputer = imputer[1]

        train_imputed = pd.DataFrame(conti_imputer.transform(train), columns=train.columns)  

        train_imputed[binary] = train_binary
            
        train_imputed = pd.DataFrame(binary_imputer.transform(train_imputed), columns=train.columns)[original_variable].astype(float)  
    else:
        # Create iterative imputers for continuous and binary variables
        conti_imputer = IterativeImputer(estimator=LinearRegression(), random_state=42, initial_strategy='median', skip_complete=True, min_value=0, max_value=max_values)
        train_imputed = pd.DataFrame(conti_imputer.fit_transform(train), columns=train.columns)  
        train_imputed[binary] = train_binary
        
        # If binary columns have missing values, use Logistic Regression imputer
        if train_imputed[binary].isnull().values.any():
            binary_imputer = IterativeImputer(estimator=LogisticRegression(multi_class='ovr'), random_state=42, initial_strategy='most_frequent', skip_complete=True, min_value=0)
        else: 
            binary_imputer = IterativeImputer(estimator=LogisticRegression(multi_class='ovr'), random_state=42, initial_strategy='most_frequent', skip_complete=False, min_value=0)
            
        train_imputed = pd.DataFrame(binary_imputer.fit_transform(train_imputed), columns=train.columns)[original_variable].astype(float)  

    # Impute additional datasets if provided
    imputed_args = [] 
    if args:   
        for data in args:
            data_binary = data[binary].reset_index(drop=True)
            data_imputed = pd.DataFrame(conti_imputer.transform(data), columns=train.columns).astype(float)  
            data_imputed[binary] = data_binary
            if data_imputed[binary].isnull().values.any():
                data_imputed = pd.DataFrame(binary_imputer.transform(data_imputed), columns=train.columns)[original_variable].astype(float)
            imputed_args.append(data_imputed)
        
    # Save imputers if specified
    if save != False:
        joblib.dump(conti_imputer, save[0] + '.joblib')          
        joblib.dump(binary_imputer, save[1] + '.joblib')      
            
    # Return imputed datasets
    if args:
        return [train_imputed, *imputed_args]
    else:
        return train_imputed
    


def Scaling(train, *args, method='scaling', binary=None, exception=None, scaler=StandardScaler(), save=False):
    """
    Scale features in the dataset.

    Args:
        train (DataFrame): The training dataset.
        args (DataFrame): Additional datasets to apply the same scaling.
        method (str): Scaling method ('scaling', 'apply', or 'inverse').
        binary (str or list): Binary columns or 'all' for all binary columns.
        exception (str): Exception column to exclude from scaling.
        scaler (object or str): Scaler object or filename for saved scaler.
        save (bool or str): Whether to save the scaler or not.

    Returns:
        DataFrame or list of DataFrame: Scaled dataset(s).
    """

    # Extract original variable names
    original_variable = train.columns

    # Handle exception column if provided
    if exception is not None:
        train_exception = train[exception]
        train = train.drop(exception, axis=1)

    # Handle scaling for all binary columns
    if binary == 'all':
        if method == 'scaling':
            train_total = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
        elif method == 'apply':
            scaler = joblib.load(scaler + '.joblib')
            train_total = pd.DataFrame(scaler.transform(train), columns=train.columns)
        elif method == 'inverse':
            scaler = joblib.load(scaler + '.joblib')
            train_total = pd.DataFrame(scaler.inverse_transform(train), columns=train.columns)   
    else:    
        # Identify binary columns if not provided
        if binary is None:
            binary = list()
            for i in train.columns:
                if ((train[i].min() == 0) & (train[i].max() == 1)) or (len(train[i].unique()) == 1):
                    binary.append(i)
        else:
            binary = binary
        
        # Separate binary and continuous columns
        train_binary = train[binary].reset_index(drop=True)
        train_conti = train.drop(binary, axis=1).reset_index(drop=True)
        
        # Apply scaling to continuous columns
        if method == 'scaling':
            train_conti = pd.DataFrame(scaler.fit_transform(train_conti), columns=train_conti.columns)
        elif method == 'apply':
            scaler = joblib.load(scaler + '.joblib')
            train_conti = pd.DataFrame(scaler.transform(train_conti), columns=train_conti.columns)
        elif method == 'inverse':
            scaler = joblib.load(scaler + '.joblib')
            train_conti = pd.DataFrame(scaler.inverse_transform(train_conti), columns=train_conti.columns
        
        # Concatenate scaled continuous and binary columns
        train_total = pd.concat([train_conti, train_binary], axis=1).astype(float)
        
    # Include exception column back if provided
    if exception is not None:
        train_total[exception] = train_exception
        
    train_total = train_total[original_variable]
                
    # Scale additional datasets if provided
    scaled_args = list()    
    if args:
        for data in args:
            if exception is not None:
                data_exception = data[exception]
                data = data.drop(exception, axis=1)

            if binary == 'all':
                data_total = pd.DataFrame(scaler.transform(data), columns=train.columns)        

            else:            
                data_binary = data[binary].reset_index(drop=True)
                data_conti = data.drop(binary, axis=1).reset_index(drop=True)

                if method == 'scaling' or method == 'apply':
                    data_conti = pd.DataFrame(scaler.transform(data_conti), columns=train_conti.columns)        
                elif method == 'inverse':
                    data_conti = pd.DataFrame(scaler.inverse_transform(data_conti), columns=train_conti.columns)   
                
                data_total = pd.concat([data_conti, data_binary], axis=1).astype(float)
                
            if exception is not None:
                data_total[exception] = data_exception
            
            data_total = data_total[original_variable]
            
            scaled_args.append(data_total)
        
    # Save scaler if specified
    if save != False:
        joblib.dump(scaler, save + '.joblib')
    
    # Return scaled datasets
    if args:
        return [train_total, *scaled_args]
    else:
        return train_total



def Number(data_dict):
    """
    Display class distribution for each dataset in the dictionary.

    Args:
        data_dict (dict): Dictionary containing dataset names as keys and datasets as values.
    """

    for name, data in data_dict.items():
        percentage = round((data['Target'].value_counts()[1] / len(data)) * 100, 1)
        print(f"{name} : {len(data)} ({percentage}%)")
        print('---------------------------------------------')



def Resampling(X_train, y_train, sampler=SMOTE()):
    """
    Resample dataset to balance classes.

    Args:
        X_train (DataFrame): Training features.
        y_train (Series): Training target.
        sampler (object): Resampling technique.

    Returns:
        DataFrame: Resampled features.
        Series: Resampled target.
    """
    
    ori = len(X_train)
    
    # Resample dataset
    sampler = sampler
    X_train, y_train = sampler.fit_resample(X_train, y_train)
    
    print('Changed Num :', ori - len(X_train))
    
    return X_train, y_train



def Performance(model, X_data, y_data=['Target'], cut_off=0.5, classification = True, verbose=0):    
    """
    Evaluate model performance.

    Args:
        model (object): Trained model.
        X_data (DataFrame): Test features.
        y_data (str or DataFrame): Test target column name or DataFrame.
        cut_off (float): Threshold for binary classification.
        classification (bool): Threshold for binary classification.
        verbose (int): Verbosity level.

    Returns:
        DataFrame: Performance metrics.
    """

    # Extract features and target
    if isinstance(y_data, (pd.DataFrame, pd.Series)):
        X_test = X_data.values
        y_test = y_data.values
    elif isinstance(y_data, list):
        X_test = X_data.drop(y_data, axis=1).values
        y_test = X_data[y_data].values
        
    # Predict probabilities
    if hasattr(model, 'predict_proba'):
        pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        pred_proba = model.predict(X_test, verbose=0)
        
    # Convert probabilities to binary predictions
    y_pred = [1 if x >= cut_off else 0 for x in pred_proba]  

    if classification == True:
        # Calculate class distribution
        num_samples = (len(y_test))
        num_ones = np.sum(y_test == 1)
        ratio_ones = (num_ones / num_samples) * 100
        
        results = pd.DataFrame({'N (incidence)': [f"{num_samples} ({ratio_ones:.1f})"],
                                'Accuracy': accuracy_score(y_test, y_pred),
                                'Precision': precision_score(y_test, y_pred, zero_division=0),
                                'Recall': recall_score(y_test, y_pred, zero_division=0),
                                'F1-score': f1_score(y_test, y_pred, zero_division=0),
                                'AUPRC': average_precision_score(y_test, pred_proba),
                                'AUROC': roc_auc_score(y_test, pred_proba)
                                }, index=[0])    
        results = round(results, 2)
    
    else:
        results = pd.DataFrame({
                        'MSE': mean_squared_error(y_test, y_pred),
                        'MAE': mean_absolute_error(y_test, y_pred),
                        'RMSE': sqrt(mean_squared_error(y_test, y_pred)),
                        }, index=[0])   
        
        results = round(results, 4)    
        
    if verbose == 1:
        print('---------------------------------------------')
        print(results)
    
    return results



def Class_weights(y_train):
    """
    Compute class weights for imbalanced classes.

    Args:
        y_train (Series): Training target.

    Returns:
        dict: Class weights.
    """

    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weights_dict = {0: class_weights[0], 1: class_weights[1]}
    
    return class_weights_dict




def Initial_bias(y_train, verbose=1):
    """
    Compute initial bias for binary classification.

    Args:
        y_train (Series): Training target.
        verbose (int): Verbosity level.

    Returns:
        array: Initial bias.
    """

    # Calculate class distribution
    neg = np.sum(y_train == 0)
    pos = np.sum(y_train == 1) 
    total = len(y_train)

    # Compute initial bias
    initial_bias = np.log([pos/neg])
    
    # Print details if verbose
    if verbose == 1:
        print('Nums:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
            total, pos, 100 * pos / total))
        print('Inintial bias :', initial_bias)
    
    return initial_bias




def Log_transformation(train, *args, binary=None):
    """
    Apply log transformation to the dataset.

    Args:
        train (DataFrame): The training dataset.
        args (DataFrame): Additional datasets to apply the same transformation.
        binary (list): List of binary columns.

    Returns:
        DataFrame or list of DataFrame: Transformed dataset(s).
    """

    # Extract original variable names
    original_variable = train.columns
    
    # Identify binary columns if not provided
    if binary is None:
        binary = list()
        for i in train.columns:
            if ((train[i].min() == 0) & (train[i].max() == 1)) or (len(train[i].unique()) == 1):
                binary.append(i)
    else:
        binary = binary
            
    # Separate binary and continuous columns
    train_binary = train[binary].reset_index(drop=True)
    train_conti = train.drop(binary, axis=1).reset_index(drop=True)
    
    # Apply log transformation to continuous columns
    train_conti = pd.DataFrame(np.log(train_conti), columns=train_conti.columns)
    
    # Concatenate transformed continuous and binary columns
    train_total = pd.concat([train_conti, train_binary], axis=1).astype(float)[original_variable]
    
    # Replace infinite values with 0 and fill missing values with 0
    train_total = pd.DataFrame(np.where(np.isinf(train_total), 0, train_total), columns=original_variable)
    train_total = train_total.fillna(0)
            
    scaled_args = list()    
    if args:
        for data in args:
            # Separate binary and continuous columns for additional datasets
            data_binary = data[binary].reset_index(drop=True)
            data_conti = data.drop(binary, axis=1).reset_index(drop=True)
            
            # Apply log transformation to continuous columns
            data_conti = pd.DataFrame(np.log(data_conti), columns=train_conti.columns)        
            
            # Concatenate transformed continuous and binary columns
            data_total = pd.concat([data_conti, data_binary], axis=1).astype(float)[original_variable]
            
            # Replace infinite values with 0 and fill missing values with 0
            data_total = pd.DataFrame(np.where(np.isinf(data_total), 0, data_total), columns=original_variable)
            data_total = data_total.fillna(0)
        
            scaled_args.append(data_total)
    
    # Return transformed datasets
    if args:
        return [train_total, *scaled_args]
    else:
        return train_total


          
            
def One_hot_encoder(train, *args, variable, drop = True):    
    """
    Perform one-hot encoding for categorical variables.

    Args:
        train (DataFrame): The training dataset.
        args (DataFrame): Additional datasets to apply the same encoding.
        variable (list): List of categorical variable names.

    Returns:
        DataFrame or list of DataFrame: Encoded dataset(s).
    """

    new_args = [df.copy() for df in args]

    for feature in variable:

        unique = sorted(train[feature].unique().tolist())

        for i in unique:
            train[feature + '_' + str(i)] = np.where(train[feature] == i, 1, 0).astype(float)

        if drop == True:
            train = train.drop(feature, axis = 1)

        if args:
            for df in new_args:
                for i in unique:
                    df[feature + '_' + str(i)] = np.where(df[feature] == i, 1, 0).astype(float)
                
                if drop == True:
                    df = df.drop(feature, axis = 1)
        
    if args:
        return [train, *new_args]
    else:
        return train




def Youden_index(model, X_val, y_val):

    pred_proba = model.predict(X_val, verbose=0)

    fpr, tpr, thresholds = roc_curve(y_val, pred_proba)

    J = tpr - fpr

    youden_cutoff = thresholds[np.argmax(J)] 
    
    print(youden_cutoff)
    
    return youden_cutoff
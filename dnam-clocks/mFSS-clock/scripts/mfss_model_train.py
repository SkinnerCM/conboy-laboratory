from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy import stats

def fit_and_evaluate_model(x_train, y_train, x_val, y_val, x_test, test_labels, flag=False):
    """
    Fit a Linear Regression model and evaluate performance on validation and test sets.

    Parameters:
    - x_train: Training features
    - y_train: Training labels
    - x_val: Validation features
    - y_val: Validation labels
    - x_test: Test features
    - test_labels: Test labels
    - flag: Whether to use positive constraints in Linear Regression

    Returns:
    - val_mse: Mean squared error on validation set
    - test_mse: Mean squared error on test set
    - val_r_val: Correlation coefficient on validation set
    - test_r_val: Correlation coefficient on test set
    """
    regr = LinearRegression(positive=flag, n_jobs=-1).fit(x_train, y_train)
    val_preds = regr.predict(x_val)
    test_preds = regr.predict(x_test)

    # Calculate performance metrics
    val_mse = mean_squared_error(y_val, val_preds)
    test_mse = mean_squared_error(test_labels, test_preds)

    # Calculate r-values
    val_r_val = compute_r_value(val_preds, y_val)
    test_r_val = compute_r_value(test_preds, test_labels)

    return val_mse, test_mse, val_r_val, test_r_val

def compute_r_value(preds, actuals):
    """
    Compute the correlation coefficient between predictions and actual values.
    
    Parameters:
    - preds: Predicted values
    - actuals: Actual values

    Returns:
    - r_value: Correlation coefficient
    """
    if len(set(preds)) < 2 or len(set(actuals)) < 2:
        return 0
    return stats.linregress(preds, actuals).rvalue

def mf_stepwise(cg_list, train, train_labels, test, test_labels, threshold, flag=False):
    """
    Perform stepwise model selection.

    Parameters:
    - cg_list: List of CpG sites
    - train: Training data
    - train_labels: Labels for training data
    - test: Test data
    - test_labels: Labels for test data
    - threshold: Number of unsuccessful iterations before stopping
    - flag: Whether to use positive constraints in Linear Regression

    Returns:
    - model_cgs: List of selected CpG sites
    - best_iter: Iteration number of the best model
    - val_mse: List of validation mean squared errors
    - val_r_val: List of validation r-values
    - test_mse: List of test mean squared errors
    - test_r_val: List of test r-values
    """
    countdown = threshold
    iteration = 0
    best_mse = float('inf')
    model_cgs = []
    val_mse = []
    val_r_val = []
    best_iter = 0
    test_mse = []
    test_r_val = []

    while countdown > 0 and iteration < len(cg_list):
        print(f'Iteration: {iteration}')
        
        # Update model with the next CpG site
        model_cgs.append(cg_list[iteration])
        temp_data = train[model_cgs]

        # Split the dataset
        x_train, x_val, y_train, y_val = train_test_split(temp_data, train_labels.age,
                                                          test_size=0.15, random_state=42)
        
        # Fit and evaluate model
        curr_val_mse, curr_test_mse, curr_val_r_val, curr_test_r_val = fit_and_evaluate_model(
            x_train, y_train, x_val, y_val, test[model_cgs], test_labels.age, flag
        )
        
        # Store performance metrics
        val_mse.append(curr_val_mse)
        test_mse.append(curr_test_mse)
        val_r_val.append(curr_val_r_val)
        test_r_val.append(curr_test_r_val)

        # Update best model if current model is better
        if curr_test_mse < best_mse:
            best_mse = curr_test_mse
            best_iter = iteration
            countdown = threshold
        else:
            countdown -= 1
        
        iteration += 1
        
    return model_cgs, best_iter, val_mse, val_r_val, test_mse, test_r_val
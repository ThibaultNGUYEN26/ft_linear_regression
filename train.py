import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def normalize_features(x):
    """
    Normalize input features to have a mean of 0 and a standard deviation of 1.

    Parameters:
        x (array-like): The input feature values to be normalized.

    Returns:
        array-like: The normalized feature values, where each value has been scaled to mean 0 and standard deviation 1.
    """
    mean_x = np.mean(x)  # Compute the mean of the input feature.
    std_x = np.std(x)    # Compute the standard deviation of the input feature.
    return (x - mean_x) / std_x  # Scale the input to mean 0 and std dev 1.


def denormalize_theta(theta, mean_x, std_x):
    """
    Convert normalized theta values back to their original scale for easier interpretation.

    Parameters:
        theta (array-like): Normalized theta values (parameters) from gradient descent.
        mean_x (float): The mean of the original input feature.
        std_x (float): The standard deviation of the original input feature.

    Returns:
        tuple: The slope and intercept of the linear model in the original scale.
    """
    slope = theta[0, 0] / std_x  # Adjust the slope by reversing the standardization.
    intercept = theta[1, 0] - (slope * mean_x)  # Adjust the intercept to account for the mean of x.
    return slope, intercept


def model(X, theta):
    """
    Compute the predictions of a linear model using the input features and parameters.

    Parameters:
        X (array-like): The input features (matrix) with a bias column for the intercept.
        theta (array-like): The model parameters (slope and intercept).

    Returns:
        array-like: The predicted values based on the linear model.
    """
    return X.dot(theta)  # Perform matrix multiplication to calculate predictions.


def cost(X, y, theta):
    """
    Compute the cost (error) of the model's predictions compared to the true values.

    Parameters:
        X (array-like): The input features (matrix) with a bias column for the intercept.
        y (array-like): The true output values.
        theta (array-like): The model parameters (slope and intercept).

    Returns:
        float: The mean squared error (MSE) cost of the model.
    """
    m = len(y)  # Number of training examples.
    return 1 / (2 * m) * np.sum((model(X, theta) - y)**2)  # Calculate the MSE cost.


def gradient(X, y, theta):
    """
    Compute the gradient of the cost function with respect to the model parameters.

    Parameters:
        X (array-like): The input features (matrix) with a bias column for the intercept.
        y (array-like): The true output values.
        theta (array-like): The model parameters (slope and intercept).

    Returns:
        array-like: The gradients for the slope (theta_1) and intercept (theta_0).
    """
    m = len(y)  # Number of training examples.
    error = model(X, theta) - y  # Compute the difference between predictions and true values.

    # Gradient for the slope (theta_1).
    grad_theta1 = (1 / m) * np.sum(error * X[:, 0].reshape(-1, 1))

    # Gradient for the intercept (theta_0).
    grad_theta0 = (1 / m) * np.sum(error)

    # Combine gradients into a single array.
    return np.array([[grad_theta1], [grad_theta0]])


def gradient_descent(X, y, theta, learning_rate, n_iterations):
    """
    Perform gradient descent optimization to minimize the cost function.

    Parameters:
        X (array-like): The input features (matrix) with a bias column for the intercept.
        y (array-like): The true output values.
        theta (array-like): The initial model parameters (slope and intercept).
        learning_rate (float): The step size for updating the parameters.
        n_iterations (int): The number of iterations to run the optimization.

    Returns:
        tuple: The optimized parameters (theta) and the history of cost values for each iteration.
    """
    cost_history = np.zeros(n_iterations)  # Initialize an array to store the cost at each iteration.
    for i in range(n_iterations):
        # Compute the gradient and update the parameters.
        theta = theta - learning_rate * gradient(X, y, theta)
        # Store the current cost value.
        cost_history[i] = cost(X, y, theta)

    return theta, cost_history  # Return the optimized parameters and the cost history.


def coef_determination(y, pred):
    """
    Calculate the coefficient of determination (R-squared) for the model's predictions.

    R-squared measures the proportion of variance in the target variable (y) that is explained
    by the model's predictions. It ranges from 0 to 1, where:
        - 0 indicates the model explains none of the variance.
        - 1 indicates the model perfectly explains the variance.

    Parameters:
        y (array-like): The actual target values.
        pred (array-like): The predicted target values from the model.

    Returns:
        float: The R-squared value indicating the proportion of explained variance.
    """
    u = ((y - pred) ** 2).sum()  # Residual sum of squares (unexplained variance).
    v = ((y - y.mean()) ** 2).sum()  # Total sum of squares (total variance).
    return 1 - u / v  # Proportion of variance explained by the model.


def mean_squared_error(y_actual, y_predicted):
    """
    Calculate the Mean Squared Error (MSE).

    MSE represents the average squared difference between the actual and predicted values.
    It penalizes larger errors more heavily than smaller ones, making it sensitive to outliers.

    Parameters:
        y_actual (array-like): The actual target values.
        y_predicted (array-like): The predicted target values from the model.

    Returns:
        float: The mean squared error.
    """
    return np.mean((y_actual - y_predicted) ** 2)


def mean_absolute_error(y_actual, y_predicted):
    """
    Calculate the Mean Absolute Error (MAE).

    MAE represents the average of the absolute differences between the actual and predicted values.
    It provides a straightforward measure of model prediction accuracy, less sensitive to outliers
    compared to MSE.

    Parameters:
        y_actual (array-like): The actual target values.
        y_predicted (array-like): The predicted target values from the model.

    Returns:
        float: The mean absolute error.
    """
    return np.mean(np.abs(y_actual - y_predicted))


def root_mean_squared_error(y_actual, y_predicted):
    """
    Calculate the Root Mean Squared Error (RMSE).

    RMSE is the square root of the Mean Squared Error (MSE). It provides a measure of how well
    the model's predictions match the actual values, in the same units as the target variable.

    Parameters:
        y_actual (array-like): The actual target values.
        y_predicted (array-like): The predicted target values from the model.

    Returns:
        float: The root mean squared error.
    """
    return np.sqrt(mean_squared_error(y_actual, y_predicted))


def main():
    """
    Main function to perform linear regression using gradient descent,
    visualize the results, and compare with Numpy's Polyfit.

    The function supports multiple command-line options:
        - `--visualizer`: Plot the final results of the linear regression.
        - `--compare`: Compare the gradient descent results with Polyfit.
        - `--steps`: Visualize intermediate steps during gradient descent.

    The dataset is expected to be loaded from "data.csv".
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Apply a Linear Regression to a Dataset.")
    parser.add_argument("-v", "--visualizer", action="store_true", help="Plot the results of the Linear Regression")
    parser.add_argument("-c", "--compare", action="store_true", help="Compare the results of the Linear Regression with Polyfit")
    parser.add_argument("-s", "--steps", action="store_true", help="Visualize the steps of Gradient Descent")

    args = parser.parse_args()

    # Step 1: Load and preprocess the dataset
    # Load data from "data.csv" and drop rows with missing values
    data = pd.read_csv("data.csv")
    data = data.dropna()
    x = data['km'].values  # Feature: Mileage
    y = data['price'].values  # Target: Price

    # Step 2: Reshape and normalize the data
    x = x.reshape(-1, 1)  # Reshape to 2D array for compatibility with matrix operations
    y = y.reshape(-1, 1)
    x_normalized = normalize_features(x)  # Normalize x to have mean 0 and std deviation 1

    # Step 3: Add a bias column to the feature matrix (for intercept calculation)
    X = np.hstack((x_normalized, np.ones(x_normalized.shape)))  # Add column of 1s for intercept

    # Step 4: Initialize model parameters
    theta = np.zeros((2, 1))  # Initialize slope and intercept to 0

    # Step 5: Compute the initial cost
    print("Initial Cost:", cost(X, y, theta))  # Display cost before optimization

    # Step 6: Perform gradient descent
    n = 500  # Number of iterations
    rate = 0.01  # Learning rate
    final_theta, cost_history = gradient_descent(X, y, theta, learning_rate=rate, n_iterations=n)

    # Step 7: Compute the final cost
    final_cost = cost(X, y, final_theta)
    print(f"Final Cost: {final_cost:.2f}")

    # Step 8: Denormalize the model parameters for interpretability
    slope_manual, intercept_manual = denormalize_theta(final_theta, mean_x=np.mean(x), std_x=np.std(x))
    print(f"Theta (Manual Gradient Descent): Intercept (theta0) = {intercept_manual}, Slope (theta1) = {slope_manual}")

    # Step 9: Generate predictions using the trained model
    predictions_manual = model(X, final_theta)

    if args.visualizer:
        # Step 10: Compute error metrics and display results
        mse_manual = mean_squared_error(y, predictions_manual)
        mae_manual = mean_absolute_error(y, predictions_manual)
        rmse_manual = root_mean_squared_error(y, predictions_manual)
        r2_manual = coef_determination(y, predictions_manual) * 100

        print(f"Mean Squared Error: {mse_manual:.2f}")
        print(f"Mean Absolute Error: {mae_manual:.2f}")
        print(f"Root Mean Squared Error: {rmse_manual:.2f}")
        print(f"Coefficient of Determination: {r2_manual:.2f}%")

        # Visualize the dataset and the manual regression results
        plt.scatter(x, y, color='blue', label='Data')  # Scatter plot of the dataset
        plt.plot(x, predictions_manual, color='red', label='Manual Regression')  # Regression line
        plt.title('Manual Linear Regression')
        plt.xlabel('Mileage')
        plt.ylabel('Price')
        plt.legend()
        plt.tight_layout()
        plt.show()

    elif args.steps:
        # Step 10: Compute error metrics
        # Calculate and display error metrics for manual gradient descent
        mse_manual = mean_squared_error(y, predictions_manual)
        mae_manual = mean_absolute_error(y, predictions_manual)
        rmse_manual = root_mean_squared_error(y, predictions_manual)
        r2_manual = coef_determination(y, predictions_manual) * 100

        print(f"Mean Squared Error: {mse_manual:.2f}")
        print(f"Mean Absolute Error: {mae_manual:.2f}")
        print(f"Root Mean Squared Error: {rmse_manual:.2f}")
        print(f"Coefficient of Determination: {r2_manual:.2f}%")

        # Step 11: Create a 2x2 grid of subplots to visualize intermediate steps and final results
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Top Left: Scatter plot of the original dataset
        axs[0, 0].scatter(x, y, label='Data', color='blue')
        axs[0, 0].set_title('Dataset')
        axs[0, 0].set_xlabel('Mileage')
        axs[0, 0].set_ylabel('Price')
        axs[0, 0].legend()

        # Top Right: Initial regression (with theta initialized to zeros)
        predictions_initial = model(X, theta)  # Initial predictions
        axs[0, 1].scatter(x, y, label='Data', color='blue')
        axs[0, 1].plot(x, predictions_initial, label='Initial Regression (Theta=0)', color='red')
        axs[0, 1].set_title('Initial Linear Regression')
        axs[0, 1].set_xlabel('Mileage')
        axs[0, 1].set_ylabel('Price')
        axs[0, 1].legend()

        # Bottom Left: Intermediate regressions during gradient descent
        intermediate_lines = []  # Store intermediate predictions for plotting
        for i in range(n):  # Perform gradient descent
            theta = theta - rate * gradient(X, y, theta)
            if i % 50 == 0:  # Store predictions every 50 iterations
                intermediate_lines.append((i, model(X, theta)))

        axs[1, 0].scatter(x, y, label='Data', color='blue')
        for iteration, predictions_step in intermediate_lines:
            axs[1, 0].plot(x, predictions_step, label=f'Iteration {iteration}', linestyle=':')
        axs[1, 0].set_title('Intermediate Linear Regressions')
        axs[1, 0].set_xlabel('Mileage')
        axs[1, 0].set_ylabel('Price')
        axs[1, 0].legend()

        # Bottom Right: Final regression after gradient descent
        predictions_final = model(X, theta)
        axs[1, 1].scatter(x, y, label='Data', color='blue')
        axs[1, 1].plot(x, predictions_final, label='Final Regression', color='red')
        axs[1, 1].set_title('Final Linear Regression')
        axs[1, 1].set_xlabel('Mileage')
        axs[1, 1].set_ylabel('Price')
        axs[1, 1].legend()

        # Adjust layout and display the subplots
        plt.tight_layout()
        plt.show()

        # Step 12: Plot Cost Over Iterations in a Separate Figure
        # Initialize parameters and storage for theta history
        theta = np.zeros((2, 1))
        theta_history = []
        denormalized_theta_history = []
        cost_history = []

        # Compute the mean and standard deviation of the original feature
        mean_x = np.mean(x)
        std_x = np.std(x)

        # Perform gradient descent and store theta, cost, and denormalized theta at each iteration
        for i in range(n):  # Number of iterations
            grad = gradient(X, y, theta)
            theta = theta - rate * grad  # Learning rate
            theta_history.append(theta.copy())  # Store a copy of normalized theta
            cost_history.append(cost(X, y, theta))  # Store the cost

            # Denormalize the parameters
            slope_denormalized = theta[0, 0] / std_x
            intercept_denormalized = theta[1, 0] - (slope_denormalized * mean_x)
            denormalized_theta_history.append([slope_denormalized, intercept_denormalized])

        # Convert histories to NumPy arrays for easy indexing
        theta_history = np.array(theta_history).squeeze()
        denormalized_theta_history = np.array(denormalized_theta_history)

        # Create subplots
        fig = plt.figure(figsize=(12, 12))

        # Subplot 1: Cost Function Over Iterations (Top Center)
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)  # Spans both columns
        ax1.plot(range(len(cost_history)), cost_history, color='blue')
        ax1.set_title('Cost Function Over Iterations', fontsize=14)
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Cost')
        ax1.grid(True)

        # Subplot 2: Evolution of Normalized Parameters (Bottom Left)
        ax2 = plt.subplot2grid((2, 2), (1, 0))  # Bottom left
        ax2.plot(range(len(theta_history)), theta_history[:, 0], label=r'$\theta_1$ (Slope)', color='red')
        ax2.plot(range(len(theta_history)), theta_history[:, 1], label=r'$\theta_0$ (Intercept)', color='green')
        ax2.set_title('Evolution of Normalized Parameters ($\\theta_0$ and $\\theta_1$)', fontsize=12)
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Parameter Value')
        ax2.legend()
        ax2.grid(True)

        # Subplot 3: Evolution of Denormalized Parameters (Bottom Right)
        ax3 = plt.subplot2grid((2, 2), (1, 1))  # Bottom right
        ax3.plot(range(len(denormalized_theta_history)), denormalized_theta_history[:, 0], label=r'$\theta_1$ (Slope)', color='red')
        ax3.plot(range(len(denormalized_theta_history)), denormalized_theta_history[:, 1], label=r'$\theta_0$ (Intercept)', color='green')
        ax3.set_title('Evolution of Denormalized Parameters ($\\theta_0$ and $\\theta_1$)', fontsize=12)
        ax3.set_xlabel('Iterations')
        ax3.set_ylabel('Parameter Value')
        ax3.legend()
        ax3.grid(True)

        # Adjust layout
        plt.tight_layout(h_pad=2)  # Add padding between rows
        plt.show()

    elif args.compare:
        # Step 11: Compare gradient descent results with Numpy's Polyfit
        # Perform linear regression using Polyfit (degree 1 for a straight line)
        theta_polyfit = np.polyfit(x.flatten(), y.flatten(), deg=1)
        print(f"Theta (Polyfit Gradient Descent): Intercept (theta0) = {theta_polyfit[1]}, Slope (theta1) = {theta_polyfit[0]}")

        # Generate predictions for Polyfit and manual gradient descent models
        predictions_polyfit = theta_polyfit[0] * x + theta_polyfit[1]  # Polyfit predictions

        # Step 12: Compute and compare error metrics for both models
        mse_manual = mean_squared_error(y, predictions_manual)
        mse_polyfit = mean_squared_error(y, predictions_polyfit)

        mae_manual = mean_absolute_error(y, predictions_manual)
        mae_polyfit = mean_absolute_error(y, predictions_polyfit)

        rmse_manual = root_mean_squared_error(y, predictions_manual)
        rmse_polyfit = root_mean_squared_error(y, predictions_polyfit)

        r2_manual = coef_determination(y, predictions_manual) * 100
        r2_polyfit = coef_determination(y, predictions_polyfit) * 100

        # Display the error metrics
        print(f"Mean Squared Error (Manual Gradient Descent): {mse_manual:.2f}")
        print(f"Mean Squared Error (Polyfit Gradient Descent): {mse_polyfit:.2f}")

        print(f"Mean Absolute Error (Manual Gradient Descent): {mae_manual:.2f}")
        print(f"Mean Absolute Error (Polyfit Gradient Descent): {mae_polyfit:.2f}")

        print(f"Root Mean Squared Error (Manual Gradient Descent): {rmse_manual:.2f}")
        print(f"Root Mean Squared Error (Polyfit Gradient Descent): {rmse_polyfit:.2f}")

        print(f"Coefficient of Determination (Manual Gradient Descent): {r2_manual:.2f}%")
        print(f"Coefficient of Determination (Polyfit Gradient Descent): {r2_polyfit:.2f}%")

        # Step 13: Visualize the regression results for both models
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left Subplot: Manual Gradient Descent Regression
        axes[0].scatter(x, y, color='blue', label='Data')
        axes[0].plot(x, predictions_manual, color='red', label='Manual Regression')
        axes[0].set_title('Manual Linear Regression')
        axes[0].set_xlabel('Mileage')
        axes[0].set_ylabel('Price')
        axes[0].legend()

        # Right Subplot: Polyfit Regression
        axes[1].scatter(x, y, color='blue', label='Data')
        axes[1].plot(x, predictions_polyfit, color='green', label='Polyfit Regression')
        axes[1].set_title('Polyfit Linear Regression')
        axes[1].set_xlabel('Mileage')
        axes[1].set_ylabel('Price')
        axes[1].legend()

        # Adjust layout and display the plots
        plt.tight_layout()
        plt.show()

    # Step 13: Save the Denormalized Model Parameters
    with open("model.txt", "w") as file:
        file.write(f"{intercept_manual},{slope_manual}")
    print("Denormalized model parameters (theta) saved to 'model.txt'.")

if __name__ == "__main__":
    main()

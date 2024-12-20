import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def normalize_features(x):
    """ Normalize input features to have mean 0 and standard deviation 1 """
    mean_x = np.mean(x)
    std_x = np.std(x)
    return (x - mean_x) / std_x


def denormalize_theta(theta, mean_x, std_x):
    """ Denormalize theta values for comparison with polyfit """
    slope = theta[0, 0] / std_x
    intercept = theta[1, 0] - (slope * mean_x)
    return slope, intercept


def model(X, theta):
    return X.dot(theta)


def cost(X, y, theta):
    m = len(y)
    return 1 / (2 * m) * np.sum((model(X, theta) - y)**2)


def gradient(X, y, theta):
    m = len(y)
    return (1 / m) * X.T.dot(model(X, theta) - y)


def gradient_descent(X, y, theta, learning_rate, n_iterations):
    cost_history = np.zeros(n_iterations)
    for i in range(0, n_iterations):
        theta = theta - learning_rate * gradient(X, y, theta)
        cost_history[i] = cost(X, y, theta)
    return theta, cost_history


def coef_determination(y, pred):
    u = ((y - pred)**2).sum()
    v = ((y - y.mean())**2).sum()
    return 1 - u / v


def mean_squared_error(y_actual, y_predicted):
    """ Calculate the Mean Squared Error """
    return np.mean((y_actual - y_predicted) ** 2)


def mean_absolute_error(y_actual, y_predicted):
    """ Calculate the Mean Absolute Error """
    return np.mean(np.abs(y_actual - y_predicted))


def root_mean_squared_error(y_actual, y_predicted):
    """ Calculate the Root Mean Squared Error """
    return np.sqrt(mean_squared_error(y_actual, y_predicted))


# Main Function
def main():
    # Step 1: Load the Dataset
    data = pd.read_csv("data.csv")
    data = data.dropna()
    x = data['km'].values
    y = data['price'].values

    # Step 2: Reshape and Normalize the Data
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    x_normalized = normalize_features(x)

    # Step 3: Add Bias Column to X
    X = np.hstack((x_normalized, np.ones(x_normalized.shape)))

    # Step 4: Initialize Parameters
    theta = np.zeros((2, 1))

    # Step 5: Compute Initial Cost
    print("Initial Cost:", cost(X, y, theta))

    # Step 6: Gradient Descent
    final_theta, cost_history = gradient_descent(X, y, theta, learning_rate=0.01, n_iterations=500)
    # print("Final Theta (Manual Gradient Descent):\n", final_theta)

    # Step 7: Compute Final Cost
    final_cost = cost(X, y, final_theta)
    print(f"Final Cost: {final_cost:.2f}")

    # Denormalize Manual Results
    mean_x = np.mean(x)  # Mean of original x
    std_x = np.std(x)    # Standard deviation of original x
    slope_manual = final_theta[0, 0] / std_x
    intercept_manual = final_theta[1, 0] - (slope_manual * mean_x)

    print(f"Theta (Manual Gradient Descent): Intercept (theta0) = {intercept_manual}, Slope (theta1) = {slope_manual}")

    # Step 8: Manual Model Predictions
    predictions_manual = model(X, final_theta)

    # Step 9: Compare with Numpy Polyfit
    # Using original x (not normalized) and flattened y
    theta_polyfit = np.polyfit(x.flatten(), y.flatten(), deg=1)  # Linear fit (degree 1)
    print(f"Theta (Polyfit Gradient Descent): Intercept (theta0) = {theta_polyfit[1]}, Slope (theta1) = {theta_polyfit[0]}")

    # Generate predictions using polyfit coefficients
    predictions_polyfit = theta_polyfit[0] * x + theta_polyfit[1]

    # Step 10: Plot Results Separately

    # Plot 1: Manual Gradient Descent Regression
    # Combined Figure with Subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns

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

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

    # Plot 2: Cost Function Over Iterations
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cost_history)), cost_history, color='blue')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function Over Iterations (Manual Gradient Descent)')
    plt.show()

    # Step 11: Error Comparison
    mse_manual = mean_squared_error(y, predictions_manual)
    mse_polyfit = mean_squared_error(y, predictions_polyfit)
    print(f"Mean Squared Error (Manual Gradient Descent): {mse_manual:.2f}")
    print(f"Mean Squared Error (Polyfit): {mse_polyfit:.2f}")

    mse_manual = mean_absolute_error(y, predictions_manual)
    mse_polyfit = mean_absolute_error(y, predictions_polyfit)
    print(f"Mean Absolute Error (Manual Gradient Descent): {mse_manual:.2f}")
    print(f"Mean Absolute Error (Polyfit): {mse_polyfit:.2f}")

    rmse_manual = root_mean_squared_error(y, predictions_manual)
    rmse_polyfit = root_mean_squared_error(y, predictions_polyfit)
    print(f"Root Mean Squared Error (Manual Gradient Descent): {rmse_manual:.2f}")
    print(f"Root Mean Squared Error (Polyfit): {rmse_polyfit:.2f}")

    # Step 12: Coefficient of Determination
    r2_manual = coef_determination(y, predictions_manual) * 100
    r2_polyfit = coef_determination(y, predictions_polyfit) * 100
    print(f"Coefficient of Determination (Manual): {r2_manual:.2f}%")
    print(f"Coefficient of Determination (Polyfit): {r2_polyfit:.2f}%")

   # Step 13: Save the Denormalized Model Parameters
    with open("model.txt", "w") as file:
        file.write(f"{intercept_manual},{slope_manual}")
    print("Denormalized model parameters saved to 'model.txt'.")

if __name__ == "__main__":
    main()

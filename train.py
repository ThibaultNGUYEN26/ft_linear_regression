import pandas as pd
import matplotlib.pyplot as plt

# Helper Function to Normalize Data
def normalize_data(X):
    """ Normalize X values between 0 and 1 """
    min_X = min(X)
    max_X = max(X)
    X_normalized = (X - min_X) / (max_X - min_X)
    return X_normalized, min_X, max_X

# Helper Function to Denormalize theta1
def denormalize_theta1(theta1, min_X, max_X):
    """ Adjust theta1 back after normalization """
    return theta1 / (max_X - min_X)

# Cost Function
def compute_cost(X, Y, theta0, theta1):
    """ Compute the Mean Squared Error """
    n = len(X)
    total_cost = sum((theta0 + theta1 * X[i] - Y[i]) ** 2 for i in range(n))
    return total_cost / (2 * n)

# Gradient Descent Function
def gradient_descent(X, Y, theta0, theta1, learning_rate, iterations):
    """ Perform Gradient Descent to optimize theta values """
    n = len(X)
    for _ in range(iterations):
        sum0 = sum((theta0 + theta1 * X[i] - Y[i]) for i in range(n))
        sum1 = sum((theta0 + theta1 * X[i] - Y[i]) * X[i] for i in range(n))
        theta0 -= (learning_rate / n) * sum0
        theta1 -= (learning_rate / n) * sum1
    return theta0, theta1

# Function to Plot Results
def plot_results(X, Y, theta0, theta1):
    """ Plot the regression line with training data """
    plt.scatter(X, Y, color='blue', label='Training Data')
    predicted_Y = [theta0 + theta1 * x for x in X]
    plt.plot(X, predicted_Y, color='red', label='Regression Line')
    plt.xlabel('Mileage (km)')
    plt.ylabel('Price')
    plt.legend()
    plt.title('Linear Regression: Mileage vs Price')
    plt.show()

# Main Function
def main():
    # Step 1: Load the Dataset
    data = pd.read_csv("data.csv")  # Ensure your CSV has 'km' and 'price' columns

    # Step 2: Process the Data
    data = data.dropna()  # Drop rows with missing data
    X = data['km'].values  # Independent variable (mileage)
    Y = data['price'].values  # Dependent variable (price)
    X_normalized, min_X, max_X = normalize_data(X)

    # Step 3: Initialize Parameters
    theta0, theta1 = 0, 0  # Initial values of theta0 (intercept) and theta1 (slope)
    learning_rate = 0.01  # Learning rate (controls step size)
    iterations = 10000  # Number of iterations for gradient descent

    # Step 4 and 5: Train the Model Using Gradient Descent and Track Cost
    print("Training the model...")
    costs = []  # Store cost values for each iteration
    for i in range(iterations):
        theta0, theta1 = gradient_descent(X_normalized, Y, theta0, theta1, learning_rate, 1)
        cost = compute_cost(X_normalized, Y, theta0, theta1)  # Calculate cost at each iteration
        costs.append(cost)
        if i % 1000 == 0:  # Print cost every 1000 iterations for monitoring
            print(f"Iteration {i}: Cost = {cost:.4f}")

    theta1 = denormalize_theta1(theta1, min_X, max_X)  # Scale theta1 back
    print(f"Training complete. theta0 = {theta0:.2f}, theta1 = {theta1:.2f}")

    # Step 6: Plot the Results
    plot_results(X, Y, theta0, theta1)

    # Step 7: Plot the Cost Over Iterations
    plt.plot(range(iterations), costs)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost Function Over Iterations")
    plt.show()

    # Step 8: Evaluate the Model
    mse = sum([(theta0 + theta1 * X[i] - Y[i]) ** 2 for i in range(len(X))]) / len(X)
    print("Mean Squared Error:", mse)

    # Step 9: Save the Trained Model Parameters
    with open("model.txt", "w") as file:
        file.write(f"{theta0},{theta1}")
    print("Model parameters saved to 'model.txt'.")

if __name__ == "__main__":
    main()

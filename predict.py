# Estimate price function
def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

def main():
    # Load the model parameters
    try:
        with open("model.txt", "r") as file:
            theta0, theta1 = map(float, file.read().split(","))
        print(f"Loaded model parameters: theta0 = {theta0:.2f}, theta1 = {theta1:.2f}")
    except FileNotFoundError:
        print("Error: 'model.txt' not found. Please run 'train.py' first to train the model.")
        return

    # Prompt user for mileage and predict the price
    while True:
        mileage_input = input("\nEnter the mileage (or 'exit' to quit): ")
        if mileage_input.lower() == 'exit':
            break
        try:
            mileage = float(mileage_input)
            price_estimate = estimate_price(mileage, theta0, theta1)
            print(f"Estimated price for mileage {mileage} km: {price_estimate:.2f}")
        except ValueError:
            print("Please enter a valid number for mileage.")

if __name__ == "__main__":
    main()

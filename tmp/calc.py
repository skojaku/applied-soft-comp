"""
Simple Calculator Script
This script has a bug that needs fixing!
"""

def add(a, b):
    """Add two numbers"""
    return a + b

def subtract(a, b):
    """Subtract b from a"""
    return a - b

def divide(a, b):
    """Divide a by b"""
    if b == 0:
        return "Error: Division by zero"
    return a / b

# SYNTAX ERROR BELOW - Students need to fix this!
def main()
    print("=== Simple Calculator ===")
    print("1. Addition")
    print("2. Subtraction")
    print("3. Division")
    # TODO: Students will add multiplication feature here

    choice = input("Enter choice (1/2/3): ")

    if choice in ['1', '2', '3']:
        num1 = float(input("Enter first number: "))
        num2 = float(input("Enter second number: "))

        if choice == '1':
            print(f"Result: {add(num1, num2)}")
        elif choice == '2':
            print(f"Result: {subtract(num1, num2)}")
        elif choice == '3':
            print(f"Result: {divide(num1, num2)}")
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()

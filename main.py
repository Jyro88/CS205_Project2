from forward_selection import forward_selection
from backwards_elimination import backward_elimination
from data_loader import load_data

def main():
    print("Welcome to Justin's Feature Selection Algorithm.")
    dataset_filename = input("Please enter the dataset filename: ")
    data, labels = load_data(dataset_filename)

    print("Type the number of the algorithm you want to run.")
    print("1. Forward Selection")
    print("2. Backward Elimination")
    algorithm_choice = int(input())
    
    if algorithm_choice == 1:
        print("Beginning search.")
        best_subset, best_accuracy = forward_selection(data, labels)
        print("\nFinished search!! The best feature subset is", best_subset, "which has an accuracy of", best_accuracy, "%")
    elif algorithm_choice == 2:
        print("Beginning search.")
        best_subset, best_accuracy = backward_elimination(data, labels)
        print("\nFinished search!! The best feature subset is", best_subset, "which has an accuracy of", best_accuracy, "%")
    else:
        print("You selected an invalid option.")

if __name__ == "__main__":
    main()

import time
import numpy as np
from validator import evaluate_subset

def forward_selection(data, labels):
    num_features = data.shape[1]
    best_subset = set()
    best_accuracy = evaluate_subset(best_subset, data, labels)
    print(f"Using no features, I get an accuracy of {best_accuracy * 100:.1f}% \n")

    overall_best_accuracy = best_accuracy
    trace_log = []
    start_time = time.time()

    for i in range(num_features):
        iteration_start_time = time.time()
        current_best_feature = None
        improved = False

        for feature in range(num_features):
            if feature not in best_subset:
                current_subset = best_subset.copy()
                current_subset.add(feature)

                accuracy = evaluate_subset(current_subset, data, labels)

                print(f"Using feature(s) {current_subset} accuracy is {accuracy * 100:.1f}%")

                if accuracy > best_accuracy:
                    current_best_feature = feature
                    best_accuracy = accuracy
                    improved = True

        iteration_end_time = time.time()
        iteration_time = iteration_end_time - iteration_start_time
        trace_log.append({
            "iteration": i + 1,
            "best_subset": best_subset.copy(),
            "best_accuracy": best_accuracy,
            "iteration_time": iteration_time
        })

        if not improved:
            print("No improvement from adding any features, terminating search.")
            break

        if current_best_feature is not None:
            best_subset.add(current_best_feature)
            overall_best_accuracy = best_accuracy
            print(f"\nFeature set {best_subset} was best, accuracy is {best_accuracy * 100:.1f}% \n")

    total_time = time.time() - start_time
    trace_log.append({
        "total_time": total_time
    })
    print_trace_log(trace_log)
    return best_subset, overall_best_accuracy * 100

def print_trace_log(trace_log):
    print("\n--- Trace Log ---")
    for entry in trace_log:
        if "iteration" in entry:
            print(f"Iteration {entry['iteration']}: Best subset = {entry['best_subset']}, Best accuracy = {entry['best_accuracy'] * 100:.1f}%, Time taken = {entry['iteration_time']:.4f} seconds")
        else:
            print(f"Total time taken: {entry['total_time']:.4f} seconds")
    print("-----------------\n")

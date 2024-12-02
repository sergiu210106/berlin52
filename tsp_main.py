import time
from berlin52 import coordinates
from tsp_algorithms import hill_climbing, simulated_annealing, tabu_search, genetic_algorithm, ant_colony_optimization
from tsp_visualization import plot_tsp_solution

def main():
    results = []

    # Algorithms
    algorithms = {
        "Hill Climbing": hill_climbing,
        "Simulated Annealing": simulated_annealing,
        "Tabu Search": tabu_search,
        "Genetic Algorithm": genetic_algorithm,
        "Ant Colony Optimization": ant_colony_optimization
    }

    for name, algorithm in algorithms.items():
        start_time = time.time()
        tour, length = algorithm(coordinates)
        end_time = time.time()
        results.append((name, length, end_time - start_time))
        plot_tsp_solution(coordinates, tour, title=f"{name} Solution")

    # Print Results
    print("Algorithm Comparison:")
    print(f"{'Algorithm':<25}{'Length':<15}{'Time (s)':<10}")
    for name, length, runtime in results:
        print(f"{name:<25}{length:<15.2f}{runtime:<10.2f}")

if __name__ == "__main__":
    main()

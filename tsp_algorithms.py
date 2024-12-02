import random
import numpy as np
from collections import deque

# Helper: Calculate distance between two cities
def calculate_distance(city1, city2):
    return np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)

# Helper: Calculate total path length
def total_path_length(tour, coordinates):
    return sum(
        calculate_distance(coordinates[tour[i]], coordinates[tour[(i + 1) % len(tour)]])
        for i in range(len(tour))
    )

# Hill Climbing
def hill_climbing(coordinates):
    n = len(coordinates)
    best_tour = list(range(n))
    random.shuffle(best_tour)
    best_length = total_path_length(best_tour, coordinates)

    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                new_tour = best_tour[:]
                new_tour[i:j + 1] = reversed(new_tour[i:j + 1])
                new_length = total_path_length(new_tour, coordinates)
                if new_length < best_length:
                    best_tour = new_tour
                    best_length = new_length
                    improved = True
    return best_tour, best_length

# Simulated Annealing
def simulated_annealing(coordinates, initial_temp=1000, cooling_rate=0.995, min_temp=1e-3):
    n = len(coordinates)
    current_tour = list(range(n))
    random.shuffle(current_tour)
    current_length = total_path_length(current_tour, coordinates)

    temp = initial_temp
    while temp > min_temp:
        i, j = sorted(random.sample(range(n), 2))
        new_tour = current_tour[:]
        new_tour[i:j + 1] = reversed(new_tour[i:j + 1])
        new_length = total_path_length(new_tour, coordinates)

        if new_length < current_length or random.random() < np.exp((current_length - new_length) / temp):
            current_tour, current_length = new_tour, new_length
        temp *= cooling_rate
    return current_tour, current_length

# Tabu Search
def tabu_search(coordinates, tabu_size=10, max_iter=100):
    n = len(coordinates)
    best_tour = list(range(n))
    random.shuffle(best_tour)
    best_length = total_path_length(best_tour, coordinates)
    tabu_list = deque(maxlen=tabu_size)

    current_tour = best_tour[:]
    for _ in range(max_iter):
        neighbors = []
        for i in range(n):
            for j in range(i + 1, n):
                new_tour = current_tour[:]
                new_tour[i:j + 1] = reversed(new_tour[i:j + 1])
                if new_tour not in tabu_list:
                    neighbors.append(new_tour)

        neighbors.sort(key=lambda x: total_path_length(x, coordinates))
        current_tour = neighbors[0]
        tabu_list.append(current_tour)

        current_length = total_path_length(current_tour, coordinates)
        if current_length < best_length:
            best_tour, best_length = current_tour, current_length

    return best_tour, best_length

# Genetic Algorithm
def genetic_algorithm(coordinates, population_size=50, generations=100, mutation_rate=0.1):
    n = len(coordinates)

    # Helper: Create random tour
    def create_tour():
        tour = list(range(n))
        random.shuffle(tour)
        return tour

    # Helper: Perform crossover
    def crossover(parent1, parent2):
        start, end = sorted(random.sample(range(n), 2))
        child = [None] * n
        child[start:end + 1] = parent1[start:end + 1]
        for city in parent2:
            if city not in child:
                child[child.index(None)] = city
        return child

    # Helper: Perform mutation
    def mutate(tour):
        i, j = random.sample(range(n), 2)
        tour[i], tour[j] = tour[j], tour[i]

    # Initialize population
    population = [create_tour() for _ in range(population_size)]
    for _ in range(generations):
        population = sorted(population, key=lambda x: total_path_length(x, coordinates))
        next_population = population[:2]  # Elitism (keep best two tours)
        for _ in range(population_size - 2):
            parent1, parent2 = random.choices(population[:10], k=2)
            child = crossover(parent1, parent2)
            if random.random() < mutation_rate:
                mutate(child)
            next_population.append(child)
        population = next_population

    best_tour = min(population, key=lambda x: total_path_length(x, coordinates))
    return best_tour, total_path_length(best_tour, coordinates)

# Ant Colony Optimization
def ant_colony_optimization(coordinates, n_ants=20, n_iterations=100, alpha=1, beta=5, evaporation_rate=0.5):
    n = len(coordinates)
    distances = np.array([[calculate_distance(coordinates[i], coordinates[j]) for j in range(n)] for i in range(n)])
    pheromones = np.ones((n, n))

    def select_next_city(current_city, visited):
        probabilities = []
        for city in range(n):
            if city not in visited:
                probabilities.append((pheromones[current_city][city] ** alpha) *
                                     ((1 / distances[current_city][city]) ** beta))
            else:
                probabilities.append(0)
        probabilities = np.array(probabilities) / sum(probabilities)
        return np.random.choice(range(n), p=probabilities)

    best_tour, best_length = None, float('inf')
    for _ in range(n_iterations):
        all_tours = []
        for _ in range(n_ants):
            current_city = random.randint(0, n - 1)
            visited = [current_city]
            while len(visited) < n:
                next_city = select_next_city(current_city, visited)
                visited.append(next_city)
                current_city = next_city
            all_tours.append(visited)

        for tour in all_tours:
            length = total_path_length(tour, coordinates)
            if length < best_length:
                best_tour, best_length = tour, length

        pheromones *= (1 - evaporation_rate)
        for tour in all_tours:
            for i in range(len(tour)):
                pheromones[tour[i]][tour[(i + 1) % n]] += 1 / total_path_length(tour, coordinates)

    return best_tour, best_length

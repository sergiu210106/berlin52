import matplotlib.pyplot as plt


def plot_tsp_solution(coordinates, tour, title="TSP Solution"):
    plt.figure(figsize=(8, 8))
    # Plot cities
    for x, y in coordinates:
        plt.scatter(x, y, color="red", s=50)
    for idx, (x, y) in enumerate(coordinates):
        plt.text(x, y, f"{idx}", fontsize=10, ha="center", va="center")

    # Plot path
    for i in range(len(tour)):
        x1, y1 = coordinates[tour[i]]
        x2, y2 = coordinates[tour[(i + 1) % len(tour)]]
        plt.plot([x1, x2], [y1, y2], color="blue")

    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid()
    plt.show()

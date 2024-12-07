import random
import networkx as nx
from node2vec import Node2Vec
import heapq
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from multiprocessing import Pool


class mine:
    ii = 1

print("Starting GWOخخخخخخ...")
all_accuracies = []  # List to store all accuracy scores
Score_list = []
P_list = []
Q_list = []
best_y = []
features = pd.read_csv('c:\\outputdolphin.csv')
G_data = pd.read_csv('c:\\dolphininput.csv')
nodes = features['node'].unique()  # Get unique node values
def custom_objective(x, G, nodes):
    p, q, d = x
    if p < 0.25 or p > 2 or q < 0.25 or q > 2 or d < 3 or d > 128:
        return 0
    d = int(round(d))
    model = Node2Vec(G, p=p, q=q, dimensions=d, walk_length=5, num_walks=40)
    model = model.fit(window=5, min_count=1, batch_words=4)
    embeddings = model.wv.vectors
    for i in range(len(embeddings)):
        print(embeddings[i])
        df2 = pd.DataFrame(data={'embedding': embeddings[i]})
        df2.to_csv('vector.csv', index=False)

    X_n2v = np.array([model.wv.get_vector(f'{node}') for node in nodes])
    X_orig = features[['Degree', 'closnesscentrality', 'betweenesscentrality', 'node']]
    X = np.concatenate((X_orig, X_n2v), axis=1)
    y = features['label']
    score = cross_val_score(LogisticRegression(), X, y).mean()

    Score_list.append(score)
    P_list.append(x[0])
    Q_list.append(x[1])
    print(20 * "-")
    print(f"iterate and p and q and score: {mine.ii, p, q, score}")
    with open("result55555.csv", 'a') as f:
        f.write(str(mine.ii) + ',' + str(p) + ',' + str(q) + ',' + str(score) + '\n')
        print(' ================= >>>>>  inserted n.o ', mine.ii)

    print(20 * "-")
    mine.ii += 1
    return score

class Wolf:
    def __init__(self, p, q, d, seed):
        self.rnd = random.Random(seed)
        self.position = [p, q, d]
        self.fitness = None

def main():
    # Load data
    features = pd.read_csv('c:\\outputdolphin.csv')
    G_data = pd.read_csv('c:\\dolphininput.csv')
    nodes = features['node'].unique()  # Get unique node values

    print("Data loaded successfully.")
    G = nx.from_pandas_edgelist(G_data, source='from', target='to')

    # Call GWO
    #best_solution = gwo(G, objective_func=lambda x: objective(x, G, nodes), n_wolves=10, n_iterations=100, dim=3, minx=[0.25, 0.25, 8], maxx=[2, 2, 128])
    best_solution = gwo(G, objective_func=custom_objective, n_wolves=5, n_iterations=20, dim=3, minx=[0.25, 0.25, 8], maxx=[2, 2, 128])

    # Final model
    best_p, best_q, best_d = best_solution
    model = Node2Vec(G, p=best_p, q=best_q, dimensions=int(best_d))
    model = model.fit(window=5, min_count=1, batch_words=4)

    X_n2v = np.array([model.wv.get_vector(f'{node}') for node in nodes])
    X_orig = features[['Degree', 'closnesscentrality', 'betweenesscentrality', 'node']]
    y = features['label']
    X = np.concatenate((X_orig, X_n2v), axis=1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    # Train and evaluate model
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')

    print("Best p:", best_p)
    print("Best q:", best_q)
    print("Best d:", best_d)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

    baseline_acc = 0.91
    improvement = (accuracy - baseline_acc) / baseline_acc * 100
    print("Improvement over baseline: %.2f%%" % improvement)

    df = pd.DataFrame(data={"P": P_list, "Q": Q_list, "Scores": Score_list})
    vector_array = np.ravel(model.wv.vectors)
    plt.plot(df)
    plt.autoscale(enable=True)

    plt.legend(['P', 'Q', 'Scores'], loc='upper right')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title('Scores over the PQ')
    plt.show()

    print("p:", P_list)
    print("q:", Q_list)
    print("Scores:", Score_list)

class Wolf:
    def __init__(self, p, q, d, seed):
        self.rnd = random.Random(seed)
        self.position = [p, q, d]
        self.fitness = None

# Grey Wolf Optimizer
# Grey Wolf Optimizer
def gwo(G, objective_func, n_wolves, n_iterations, dim, minx, maxx):
    rnd = random.Random(0)
    nodes = features['node'].unique()  # Get unique node values

    # Create n random wolves (parallelized)
    with Pool() as pool:
        population = pool.starmap(create_wolf, [(G, nodes, minx, maxx, rnd) for _ in range(n_wolves)])

    # Sort the population based on fitness
    population = sorted(population, key=lambda temp: temp.fitness, reverse=True)

    # Best 3 solutions will be called alpha, beta, and gamma
    alpha_wolf, beta_wolf, gamma_wolf = population[:3]

    # Store the best fitness values for convergence analysis
    best_fitness_values = [alpha_wolf.fitness]

    # Main loop of GWO
    for Iter in range(n_iterations):
        # Linearly decreased from 2 to 0
        a = 2 * (1 - Iter / n_iterations)

        # Update each population member with the help of best three members (parallelized)
        with Pool() as pool:
            updated_population = pool.starmap(update_wolf,
                                              [(wolf, alpha_wolf, beta_wolf, gamma_wolf, a, rnd, dim, G, objective_func)
                                               for wolf in population])
        population = updated_population

        # Sort the population based on fitness
        population = sorted(population, key=lambda temp: temp.fitness, reverse=True)

        # Update alpha, beta, and gamma
        alpha_wolf, beta_wolf, gamma_wolf = population[:3]

        # Store the best fitness value for convergence analysis
        best_fitness_values.append(alpha_wolf.fitness)

    # Plot the convergence curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(n_iterations + 1), best_fitness_values)
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.title('Convergence Curve')
    plt.show()

    # Return the best solution
    return alpha_wolf.position

# Helper functions for parallelization
def create_wolf(G, nodes, minx, maxx, rnd):
    # Generate random seed for each wolf
    seed = rnd.randint(0, 10000)
    wolf_rnd = random.Random(seed)

    # Generate random position within the search space
    p = wolf_rnd.uniform(minx[0], maxx[0])
    q = wolf_rnd.uniform(minx[1], maxx[1])
    d = wolf_rnd.randint(minx[2], maxx[2])

    # Create the wolf and calculate its fitness
    wolf = Wolf(p, q, d, seed)
    wolf.fitness = objective(wolf.position, G, nodes)
    return wolf
def update_wolf(wolf, alpha_wolf, beta_wolf, gamma_wolf, a, rnd, dim, G, objective_func):
    A1, A2, A3 = a * (2 * rnd.random() - 1), a * (2 * rnd.random() - 1), a * (2 * rnd.random() - 1)
    C1, C2, C3 = 2 * rnd.random(), 2 * rnd.random(), 2 * rnd.random()

    Xnew = [0.0 for _ in range(dim)]
    for j in range(dim):
        Xnew[j] = (alpha_wolf.position[j] - A1 * abs(C1 * alpha_wolf.position[j] - wolf.position[j])) + \
                  (beta_wolf.position[j] - A2 * abs(C2 * beta_wolf.position[j] - wolf.position[j])) + \
                  (gamma_wolf.position[j] - A3 * abs(C3 * gamma_wolf.position[j] - wolf.position[j]))
        Xnew[j] /= 3.0

    # Fitness calculation of new solution
    fnew = objective_func(Xnew, G, nodes)

    # Greedy selection
    if fnew > wolf.fitness:
        wolf.position = Xnew
        wolf.fitness = fnew

    return wolf
# Objective function
def objective(x, G, nodes):
    p, q, d = x

    if p < 0.25 or p > 2 or q < 0.25 or q > 2 or d < 3 or d > 128:
        return 0
    d = int(round(d))
    model = Node2Vec(G, p=p, q=q, dimensions=d, walk_length=5, num_walks=40)
    model = model.fit(window=5, min_count=1, batch_words=4)
    embeddings = model.wv.vectors
    for i in range(len(embeddings)):
        print(embeddings[i])
        df2 = pd.DataFrame(data={'embedding': embeddings[i]})
        df2.to_csv('vector.csv', index=False)

    X_n2v = np.array([model.wv.get_vector(f'{node}') for node in nodes])
    X_orig = features[['Degree', 'closnesscentrality', 'betweenesscentrality', 'node']]
    X = np.concatenate((X_orig, X_n2v), axis=1)
    y = features['label']
    score = cross_val_score(LogisticRegression(), X, y).mean()

    Score_list.append(score)
    P_list.append(x[0])
    Q_list.append(x[1])
    print(20 * "-")
    print(f"iterate and p and q and score: {mine.ii, p, q, score}")
    with open("result55555.csv", 'a') as f:
        f.write(str(mine.ii) + ',' + str(p) + ',' + str(q) + ',' + str(score) + '\n')
        print(' ================= >>>>>  inserted n.o ', mine.ii)

    print(20 * "-")
    mine.ii += 1
    return score


if __name__ == '__main__':
    main()
    
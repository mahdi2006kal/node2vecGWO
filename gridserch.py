import random
import networkx as nx
from node2vec import Node2Vec
import heapq
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class mine:
    ii = 1

print("Starting GWO and Grid Search...")
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

class Node2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, G, p=1, q=1, dimensions=128, walk_length=80, num_walks=10, workers=1):
        self.G = G
        self.p = p
        self.q = q
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        model = Node2Vec(self.G, p=self.p, q=self.q, dimensions=self.dimensions,
                         walk_length=self.walk_length, num_walks=self.num_walks, workers=self.workers)
        model = model.fit(window=5, min_count=1, batch_words=4)
        embeddings = [model.wv[str(node)] for node in X]
        return np.array(embeddings)

    def get_params(self, deep=True):
        return {
            'G': self.G,
            'p': self.p,
            'q': self.q,
            'dimensions': self.dimensions,
            'walk_length': self.walk_length,
            'num_walks': self.num_walks,
            'workers': self.workers
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

def main():
    # Load data
    features = pd.read_csv('c:\\outputdolphin.csv')
    G_data = pd.read_csv('c:\\dolphininput.csv')
    nodes = features['node'].unique()  # Get unique node values

    print("Data loaded successfully.")
    G = nx.from_pandas_edgelist(G_data, source='from', target='to')

    # Call GWO
    best_solution_gwo = gwo(G, objective_func=custom_objective, n_wolves=3, n_iterations=15, dim=3, minx=[0.25, 0.25, 8], maxx=[2, 2, 128])

    # Grid Search
    param_grid = {'p': np.linspace(0.25, 2, 8), 'q': np.linspace(0.25, 2, 8), 'dimensions': range(3, 129, 16)}
    model = Node2VecTransformer(G, walk_length=5, num_walks=40)  # Pass G as an argument
    grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=1)

    X_orig = features[['Degree', 'closnesscentrality', 'betweenesscentrality', 'node']]
    y = features['label']
    grid_search.fit(X_orig, y)

    best_p_gs, best_q_gs, best_d_gs = grid_search.best_params_['p'], grid_search.best_params_['q'], grid_search.best_params_['dimensions']
    best_score_gs = grid_search.best_score_
    print("Best p (Grid Search):", best_p_gs)
    print("Best q (Grid Search):", best_q_gs)
    print("Best d (Grid Search):", best_d_gs)
    print("Best Score (Grid Search):", best_score_gs)

    # Final models
    best_p_gwo, best_q_gwo, best_d_gwo = best_solution_gwo
    model_gwo = Node2Vec(G, p=best_p_gwo, q=best_q_gwo, dimensions=int(best_d_gwo))
    model_gwo = model_gwo.fit(window=5, min_count=1, batch_words=4)


    model_gs = Node2VecTransformer(G, p=best_p_gs, q=best_q_gs, dimensions=best_d_gs, walk_length=5, num_walks=40)
    model_gs.fit(X_orig)

    X_n2v_gwo = np.array([model_gwo.wv[f'{node}'] for node in nodes])
    X_n2v_gs = model_gs.transform(nodes)
    X_orig = features[['Degree', 'closnesscentrality', 'betweenesscentrality', 'node']]
    X_gwo = np.concatenate((X_orig, X_n2v_gwo), axis=1)
    X_gs = np.concatenate((X_orig, X_n2v_gs), axis=1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_gwo, y, test_size=0.30, random_state=42)
    
    # Train and evaluate models
    clf_gwo = LogisticRegression(max_iter=2000)
    clf_gwo.fit(X_train, y_train)
    y_pred_gwo = clf_gwo.predict(X_test)


    X_train, X_test, y_train, y_test = train_test_split(X_gs, y, test_size=0.30, random_state=42)
    clf_gs = LogisticRegression(max_iter=2000)
    clf_gs.fit(X_train, y_train)
    y_pred_gs = clf_gs.predict(X_test)


    #X_train, X_test, y_train, y_test = train_test_split(X_gwo, y, test_size=0.30, random_state=42)

   # clf.fit(X_train, y_train)
    #y_pred = clf.predict(X_test)





    accuracy_gwo = accuracy_score(y_test, y_pred_gwo)
    precision_gwo = precision_score(y_test, y_pred_gwo, average='micro')
    recall_gwo = recall_score(y_test, y_pred_gwo, average='micro')
    f1_gwo = f1_score(y_test, y_pred_gwo, average='micro')

    accuracy_gs = accuracy_score(y_test, y_pred_gs)
    precision_gs = precision_score(y_test, y_pred_gs, average='micro')
    recall_gs = recall_score(y_test, y_pred_gs, average='micro')
    f1_gs = f1_score(y_test, y_pred_gs, average='micro')

    #baseline_clf = LogisticRegression(max_iter=2000)
    #baseline_clf.fit(X_orig, y_train)
   # baseline_pred = baseline_clf.predict(X_test)
   # baseline_acc = accuracy_score(y_test, baseline_pred)

    #print("\nBaseline Accuracy:", baseline_acc)
    print("\nGWO Results:")
    print("Best p (GWO):", best_p_gwo)
    print("Best q (GWO):", best_q_gwo)
    print("Best d (GWO):", best_d_gwo)
    print("Accuracy (GWO):", accuracy_gwo)
    print("Precision (GWO):", precision_gwo)
    print("Recall (GWO):", recall_gwo)
    print("F1-score (GWO):", f1_gwo)
   # improvement_gwo = (accuracy_gwo - baseline_acc) / baseline_acc * 100
    #print("Improvement over baseline (GWO): %.2f%%" % improvement_gwo)

    print("\nGrid Search Results:")
    print("Best p (Grid Search):", best_p_gs)
    print("Best q (Grid Search):", best_q_gs)
    print("Best d (Grid Search):", best_d_gs)
    print("Accuracy (Grid Search):", accuracy_gs)
    print("Precision (Grid Search):", precision_gs)
    print("Recall (Grid Search):", recall_gs)
    print("F1-score (Grid Search):", f1_gs)
    #improvement_gs = (accuracy_gs - baseline_acc) / baseline_acc * 100
   # print("Improvement over baseline (Grid Search): %.2f%%" % improvement_gs)

    df = pd.DataFrame(data={"P": P_list, "Q": Q_list, "Scores": Score_list})
    #vector_array = np.ravel(model.wv.vectors)
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

    accuracy_list = []

# GWO Results
    accuracy_gwo = accuracy_score(y_test, y_pred_gwo)
    accuracy_list.append(("GWO", accuracy_gwo))

# Grid Search Results
    accuracy_gs = accuracy_score(y_test, y_pred_gs)
    accuracy_list.append(("Grid Search", accuracy_gs))

# Plot accuracy comparison
    models = [model for model, _ in accuracy_list]
    accuracies = [accuracy for _, accuracy in accuracy_list]

    plt.figure(figsize=(8, 6))
    plt.bar(models, accuracies)
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison: GWO vs. Grid Search')
    plt.show()






def create_wolf(G, nodes, minx, maxx, rnd):
    seed = rnd.randint(0, 10000)
    wolf_rnd = random.Random(seed)

    p = wolf_rnd.uniform(minx[0], maxx[0])
    q = wolf_rnd.uniform(minx[1], maxx[1])
    d = wolf_rnd.randint(minx[2], maxx[2])

    wolf = Wolf(p, q, d, seed)
    wolf.fitness = custom_objective(wolf.position, G, nodes)
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

    fnew = objective_func(Xnew, G, nodes)

    if fnew > wolf.fitness:
        wolf.position = Xnew
        wolf.fitness = fnew

    return wolf

def gwo(G, objective_func, n_wolves, n_iterations, dim, minx, maxx):
    rnd = random.Random(0)
    nodes = features['node'].unique()

    with Pool() as pool:
        population = pool.starmap(create_wolf, [(G, nodes, minx, maxx, rnd) for _ in range(n_wolves)])

    population = sorted(population, key=lambda temp: temp.fitness, reverse=True)

    alpha_wolf, beta_wolf, gamma_wolf = population[:3]

    best_fitness_values = [alpha_wolf.fitness]

    for Iter in range(n_iterations):
        a = 2 * (1 - Iter / n_iterations)

        with Pool() as pool:
            updated_population = pool.starmap(update_wolf,
                                              [(wolf, alpha_wolf, beta_wolf, gamma_wolf, a, rnd, dim, G, objective_func)
                                               for wolf in population])
        population = updated_population

        population = sorted(population, key=lambda temp: temp.fitness, reverse=True)

        alpha_wolf, beta_wolf, gamma_wolf = population[:3]

        best_fitness_values.append(alpha_wolf.fitness)

    plt.figure(figsize=(10, 6))
    plt.plot(range(n_iterations + 1), best_fitness_values)
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.title('Convergence Curve')
    plt.show()

    return alpha_wolf.position

if __name__ == '__main__':
    main()

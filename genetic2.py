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
class mine:
  ii = 1
print("Starting genetic...")
all_accuracies = []  # List to store all accuracy scores
Score_list = []
P_list = []
Q_list = []
best_y = []
class Individual:
  def __init__(self, chromosome, fitness):
    self.chromosome = chromosome
    self.fitness = fitness

def objective(chromosome):
  p, q, d = chromosome

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
  P_list.append(X[0])
  Q_list.append(X[1])
  print(20 * "-")
  print(f"iterate and p and q and score: {mine.ii , p, q, score}")
  with open("result55555.csv", 'a') as f:
      f.write(str(mine.ii) + ',' +  str(p)+ ','+ str(q)+ ' ,'+ str(score)+ '\n' )
  return score

def selection(population):
  # Select parents based on their fitness using Roulette Wheel Selection
  selected = []
  for _ in range(len(population) // 2):
    # Calculate the total fitness of the population
    total_fitness = sum(ind.fitness for ind in population)

    # Generate a random number between 0 and the total fitness
    r = random.uniform(0, total_fitness)

    # Select the first parent
    parent1 = None
    for ind in population:
      r -= ind.fitness
      if r <= 0:
        parent1 = ind
        break

    # Generate a random number between 0 and the total fitness
    r = random.uniform(0, total_fitness)

    # Select the second parent
    parent2 = None
    for ind in population:
      r -= ind.fitness
      if r <= 0:
        parent2 = ind
        break

    selected.append(parent1)
    selected.append(parent2)
  return selected

def crossover(parent1, parent2):
  # Perform single-point crossover
  crossover_point = random.randint(1, len(parent1.chromosome) - 2)
  child1 = parent1.chromosome[:crossover_point] + parent2.chromosome[crossover_point:]
  child2 = parent2.chromosome[:crossover_point] + parent1.chromosome[crossover_point:]
  return child1, child2

def mutation(chromosome, mutation_rate):
  # Perform bit flip mutation
  for i in range(len(chromosome)):
    if random.random() < mutation_rate:
      chromosome[i] = random.uniform(minx[i], maxx[i])
  return chromosome

def genetic_algorithm(objective_func, population_size, n_iterations, dim, minx, maxx, mutation_rate):
    population = [Individual([random.uniform(minx[i], maxx[i]) for i in range(dim)], 0) for _ in range(population_size)]
    best_fitness_values = []  # List to store best fitness values

    for _ in range(n_iterations):
        # Evaluate fitness of each individual
        for individual in population:
            individual.fitness = objective_func(individual.chromosome)

        # Select parents based on fitness
        selected_parents = selection(population)

        # Perform crossover
        children = []
        for i in range(0, len(selected_parents), 2):
            child1, child2 = crossover(selected_parents[i], selected_parents[i + 1])
            children.append(Individual(child1, 0))
            children.append(Individual(child2, 0))

        # Perform mutation
        for child in children:
            child.chromosome = mutation(child.chromosome, mutation_rate)

        # Combine parents and children for next generation
        population = population + children
        population = sorted(population, key=lambda ind: -ind.fitness, reverse=True)[:population_size]

        # Store the best fitness value for convergence analysis
        best_fitness_values.append(population[0].fitness)

    # Plot the convergence curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(n_iterations), best_fitness_values)
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.title('Convergence Curve')
    plt.show()

    # Return the best individual
    return population[0]

# Load data
features = pd.read_csv('c:\outputdolphin.csv')
G = pd.read_csv ('c:\dolphininput.csv')
nodes = features['node'].unique()

print("Data loaded successfully.")
G = nx.from_pandas_edgelist(G, source='from', target='to')

# Parameters
minx = [0.25, 0.25, 3]
maxx = [2, 2, 128]
population_size = 10
n_iterations = 50
mutation_rate = 0.2

# Run GA
best_solution = genetic_algorithm(objective_func=objective, population_size=population_size, n_iterations=n_iterations, dim=3, minx=minx, maxx=maxx, mutation_rate=mutation_rate)

# Final model
best_p, best_q, best_d = best_solution.chromosome
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
# Visualize the optimization process
P_list = []
Q_list = []
Score_list = []
for i in range(n_iterations):
    P_list.append(best_solution.chromosome[0])
    Q_list.append(best_solution.chromosome[1])
    Score_list.append(best_solution.fitness)

df = pd.DataFrame(data={"P":P_list ,"Q":Q_list, "Scores": Score_list})
vector_array = np.ravel(model.wv.vectors) 
plt.plot(df)
plt.autoscale(enable= True)

plt.legend(['P', 'Q' , 'Scores'], loc='upper right')
# plt.yscale("log")
# plt.xscale("log")
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('Scores over the PQ')
plt.show()

print("p:", P_list)
print("q:", Q_list)
print("Scores:",Score_list)

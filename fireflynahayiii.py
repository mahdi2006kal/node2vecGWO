import random
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import numpy as np
import matplotlib.pyplot as plt
from joblib import Memory, Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import seaborn as sns
import time
import os

# تنظیم مسیر فایل‌های ورودی
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KARATE_FILE = 'c:\dolphininput.csv'
DOLPHIN_FILE = 'c:\dolphininput.csv'
FACEBOOK_FILE = 'c:\dolphininput.csv'

memory = Memory(location='.cache', verbose=1)

def print_file_paths():
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"KARATE_FILE: {KARATE_FILE}")
    print(f"DOLPHIN_FILE: {DOLPHIN_FILE}")
    print(f"FACEBOOK_FILE: {FACEBOOK_FILE}")
    
    for file_path in [KARATE_FILE, DOLPHIN_FILE]:
        if os.path.exists(file_path):
            print(f"File exists: {file_path}")
        else:
            print(f"File does not exist: {file_path}")

@memory.cache
def create_graph(file_path):
    df = pd.read_csv(file_path)
    if 'from' in df.columns and 'to' in df.columns:
        return nx.from_pandas_edgelist(df, 'from', 'to')
    elif 'source' in df.columns and 'target' in df.columns:
        return nx.from_pandas_edgelist(df, 'source', 'target')
    else:
        return nx.from_pandas_edgelist(df, df.columns[0], df.columns[1])

@memory.cache
def node2vec_embedding(num_nodes, edges, dimensions, walk_length, num_walks, p, q):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, p=p, q=q, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    return np.array([model.wv[str(node)] for node in G.nodes()])

def evaluate_embedding(G, embeddings):
    communities = nx.community.louvain_communities(G)
    modularity = nx.community.modularity(G, communities)
    
    sample_size = min(1000, len(embeddings))
    sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
    sample_embeddings = embeddings[sample_indices]
    avg_distance = np.mean(np.linalg.norm(sample_embeddings[:, None] - sample_embeddings, axis=2))
    
    return modularity, avg_distance, 0.7 * modularity + 0.3 * (1 / (avg_distance + 1e-6))

class FireflyAlgorithm:
    def __init__(self, fitness_function, num_params, param_ranges, num_fireflies=10, max_iter=20, alpha=0.5, beta0=1, gamma=0.01):
        self.fitness_function = fitness_function
        self.num_params = num_params
        self.param_ranges = param_ranges
        self.num_fireflies = num_fireflies
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.best_pos = None
        self.best_score = float("-inf")
        self.convergence = []
        self.scaler = MinMaxScaler()

    def optimize(self):
        fireflies = self.scaler.fit_transform(np.random.uniform(low=[r[0] for r in self.param_ranges], 
                                              high=[r[1] for r in self.param_ranges], 
                                              size=(self.num_fireflies, self.num_params)))

        for t in range(self.max_iter):
            fitnesses = Parallel(n_jobs=-1)(delayed(self.fitness_function)(self.scaler.inverse_transform(firefly.reshape(1, -1))[0]) for firefly in fireflies)
            
            for i in range(self.num_fireflies):
                for j in range(self.num_fireflies):
                    if fitnesses[j] > fitnesses[i]:
                        r = np.linalg.norm(fireflies[i] - fireflies[j])
                        beta = self.beta0 * np.exp(-self.gamma * r**2)
                        fireflies[i] += beta * (fireflies[j] - fireflies[i]) + self.alpha * (np.random.rand(self.num_params) - 0.5)
                        fireflies[i] = np.clip(fireflies[i], 0, 1)
            
            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > self.best_score:
                self.best_score = fitnesses[best_idx]
                self.best_pos = fireflies[best_idx].copy()
            
            self.convergence.append(self.best_score)

            if t > 10 and self.convergence[-1] == self.convergence[-10]:
                print(f"توقف زودهنگام Firefly در تکرار {t}")
                break

        return self.scaler.inverse_transform(self.best_pos.reshape(1, -1))[0], self.best_score

class GreyWolfOptimizer:
    def __init__(self, fitness_function, num_params, param_ranges, num_wolves=10, max_iter=20):
        self.fitness_function = fitness_function
        self.num_params = num_params
        self.param_ranges = param_ranges
        self.num_wolves = num_wolves
        self.max_iter = max_iter
        self.alpha_pos = np.zeros(num_params)
        self.alpha_score = float("-inf")
        self.beta_pos = np.zeros(num_params)
        self.beta_score = float("-inf")
        self.delta_pos = np.zeros(num_params)
        self.delta_score = float("-inf")
        self.convergence = []
        self.scaler = MinMaxScaler()

    def optimize(self):
        wolves = self.scaler.fit_transform(np.random.uniform(low=[r[0] for r in self.param_ranges], 
                                   high=[r[1] for r in self.param_ranges], 
                                   size=(self.num_wolves, self.num_params)))

        for l in range(self.max_iter):
            fitnesses = Parallel(n_jobs=-1)(delayed(self.fitness_function)(self.scaler.inverse_transform(wolf.reshape(1, -1))[0]) for wolf in wolves)
            
            for i, fitness in enumerate(fitnesses):
                if fitness > self.alpha_score:
                    self.alpha_score = fitness
                    self.alpha_pos = wolves[i].copy()
                elif fitness > self.beta_score:
                    self.beta_score = fitness
                    self.beta_pos = wolves[i].copy()
                elif fitness > self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = wolves[i].copy()
            
            self.convergence.append(self.alpha_score)

            a = 2 - l * (2 / self.max_iter)
            for i in range(self.num_wolves):
                A = 2 * a * np.random.random(self.num_params) - a
                C = 2 * np.random.random(self.num_params)
                
                D_alpha = abs(C * self.alpha_pos - wolves[i])
                D_beta = abs(C * self.beta_pos - wolves[i])
                D_delta = abs(C * self.delta_pos - wolves[i])
                
                X1 = self.alpha_pos - A * D_alpha
                X2 = self.beta_pos - A * D_beta
                X3 = self.delta_pos - A * D_delta
                
                wolves[i] = np.clip((X1 + X2 + X3) / 3, 0, 1)

            if l > 10 and self.convergence[-1] == self.convergence[-10]:
                print(f"توقف زودهنگام GWO در تکرار {l}")
                break

        return self.scaler.inverse_transform(self.alpha_pos.reshape(1, -1))[0], self.alpha_score

class ParticleSwarmOptimizer:
    def __init__(self, fitness_function, num_params, param_ranges, num_particles=20, max_iter=30, w=0.5, c1=1, c2=2):
        self.fitness_function = fitness_function
        self.num_params = num_params
        self.param_ranges = param_ranges
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w  # inertia weight
        self.c1 = c1  # cognitive parameter
        self.c2 = c2  # social parameter
        self.best_pos = None
        self.best_score = float("-inf")
        self.convergence = []
        self.scaler = MinMaxScaler()

    def optimize(self):
        particles = self.scaler.fit_transform(np.random.uniform(low=[r[0] for r in self.param_ranges], 
                                              high=[r[1] for r in self.param_ranges], 
                                              size=(self.num_particles, self.num_params)))
        velocities = np.zeros((self.num_particles, self.num_params))
        personal_best_pos = particles.copy()
        personal_best_score = np.array([float("-inf")] * self.num_particles)

        for t in range(self.max_iter):
            fitnesses = Parallel(n_jobs=-1)(delayed(self.fitness_function)(self.scaler.inverse_transform(particle.reshape(1, -1))[0]) for particle in particles)
            
            for i in range(self.num_particles):
                if fitnesses[i] > personal_best_score[i]:
                    personal_best_score[i] = fitnesses[i]
                    personal_best_pos[i] = particles[i].copy()
                
                if fitnesses[i] > self.best_score:
                    self.best_score = fitnesses[i]
                    self.best_pos = particles[i].copy()
            
            self.convergence.append(self.best_score)
            
            r1, r2 = np.random.rand(2)
            velocities = (self.w * velocities + 
                          self.c1 * r1 * (personal_best_pos - particles) + 
                          self.c2 * r2 * (self.best_pos - particles))
            
            particles += velocities
            particles = np.clip(particles, 0, 1)

            if t > 10 and self.convergence[-1] == self.convergence[-10]:
                print(f"توقف زودهنگام PSO در تکرار {t}")
                break

        return self.scaler.inverse_transform(self.best_pos.reshape(1, -1))[0], self.best_score

def optimize_node2vec(G, dataset_name):
    param_ranges = [
        (50, 200),    # dimensions
        (10, 30),     # walk_length
        (50, 200),    # num_walks
        (0.5, 2.0),   # p
        (0.5, 2.0)    # q
    ]

    @memory.cache
    def fitness_function(params):
        dimensions, walk_length, num_walks, p, q = params
        embeddings = node2vec_embedding(G.number_of_nodes(), list(G.edges()), 
                                        int(dimensions), int(walk_length), int(num_walks), p, q)
        return evaluate_embedding(G, embeddings)[2]  # Return only the combined score

    firefly = FireflyAlgorithm(fitness_function, len(param_ranges), param_ranges, num_fireflies=10, max_iter=30)
    gwo = GreyWolfOptimizer(fitness_function, len(param_ranges), param_ranges, num_wolves=10, max_iter=30)
    pso = ParticleSwarmOptimizer(fitness_function, len(param_ranges), param_ranges, num_particles=10, max_iter=30)
    
    firefly_params, firefly_fitness = firefly.optimize()
    gwo_params, gwo_fitness = gwo.optimize()
    pso_params, pso_fitness = pso.optimize()

    print(f"\nبهترین پارامترهای Node2Vec برای {dataset_name} با الگوریتم Firefly:")
    print(f"dimensions: {int(firefly_params[0])}")
    print(f"walk_length: {int(firefly_params[1])}")
    print(f"num_walks: {int(firefly_params[2])}")
    print(f"p: {firefly_params[3]:.2f}")
    print(f"q: {firefly_params[4]:.2f}")
    print(f"بهترین امتیاز Firefly: {firefly_fitness:.4f}")

    print(f"\nبهترین پارامترهای Node2Vec برای {dataset_name} با الگوریتم GWO:")
    print(f"dimensions: {int(gwo_params[0])}")
    print(f"walk_length: {int(gwo_params[1])}")
    print(f"num_walks: {int(gwo_params[2])}")
    print(f"p: {gwo_params[3]:.2f}")
    print(f"q: {gwo_params[4]:.2f}")
    print(f"بهترین امتیاز GWO: {gwo_fitness:.4f}")

    print(f"\nبهترین پارامترهای Node2Vec برای {dataset_name} با الگوریتم PSO:")
    print(f"dimensions: {int(pso_params[0])}")
    print(f"walk_length: {int(pso_params[1])}")
    print(f"num_walks: {int(pso_params[2])}")
    print(f"p: {pso_params[3]:.2f}")
    print(f"q: {pso_params[4]:.2f}")
    print(f"بهترین امتیاز PSO: {pso_fitness:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(firefly.convergence) + 1), firefly.convergence, label='Firefly')
    plt.plot(range(1, len(gwo.convergence) + 1), gwo.convergence, label='GWO')
    plt.plot(range(1, len(pso.convergence) + 1), pso.convergence, label='PSO')
    plt.title(f"منحنی همگرایی بهینه‌سازی Firefly، GWO و PSO برای {dataset_name}")
    plt.xlabel("تکرار")
    plt.ylabel("بهترین امتیاز")
    plt.legend()
    plt.savefig(f"convergence_comparison_{dataset_name}.png")
    plt.close()

    return firefly_params, firefly_fitness, gwo_params, gwo_fitness, pso_params, pso_fitness

def compare_methods(G, dataset_name, firefly_params, gwo_params, pso_params):
   # default_params = [80, 80, 8, 2, 2]
   # hingrl_params = [150, 70, 15, 1.5, 0.8]
    
    results = []
    #for method, params in [("Default", default_params), ("HINGRL", hingrl_params), 
                          # ("Firefly", firefly_params), ("GWO", gwo_params), ("PSO", pso_params)]:
    for method, params in [ 
                           ("Firefly", firefly_params), ("GWO", gwo_params), ("PSO", pso_params)]:
        start_time = time.time()
        embeddings = node2vec_embedding(G.number_of_nodes(), list(G.edges()), 
                                        int(params[0]), int(params[1]), int(params[2]), params[3], params[4])
        modularity, avg_distance, combined_score = evaluate_embedding(G, embeddings)
        end_time = time.time()
        
        results.append({
            "Method": method,
            "Modularity": modularity,
            "Avg Distance": avg_distance,
            "Combined Score": combined_score,
            "Time (s)": end_time - start_time
        })
    
    df_results = pd.DataFrame(results)
    print(f"\nمقایسه نتایج برای {dataset_name}:")
    print(df_results.to_string(index=False))
    
    df_results.to_csv(f"comparison_results_{dataset_name}.csv", index=False)
    
    return df_results
def visualize_embeddings(G, embeddings, dataset_name):
    n_samples = len(embeddings)
    
    if n_samples < 5:
        print(f"تعداد گره‌ها در {dataset_name} برای تصویرسازی با t-SNE کافی نیست.")
        return
    
    perplexity = min(30, n_samples - 1)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    
    try:
        embeddings_2d = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
        plt.title(f"تصویرسازی Embedding‌ها برای {dataset_name}")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.savefig(f"embeddings_visualization_{dataset_name}.png")
        plt.close()
        
        print(f"تصویرسازی برای {dataset_name} با موفقیت انجام شد.")
    except Exception as e:
        print(f"خطا در تصویرسازی برای {dataset_name}: {str(e)}")

def main():
    print_file_paths()
    
    datasets = [
        ("Karate Club", KARATE_FILE),
        ("Dolphin", DOLPHIN_FILE),
        ("Facebook", FACEBOOK_FILE)
    ]
    
    all_results = []
    
    for dataset_name, file_path in datasets:
        print(f"\nپردازش دیتاست {dataset_name}...")
        G = create_graph(file_path)
        print(f"تعداد گره‌ها در {dataset_name}: {G.number_of_nodes()}")
        print(f"تعداد یال‌ها در {dataset_name}: {G.number_of_edges()}")
        
def main():
    print_file_paths()
    
    datasets = [
        ("Karate Club", KARATE_FILE),
        ("Dolphin", DOLPHIN_FILE),
        ("Facebook", FACEBOOK_FILE)
    ]
    
    all_results = []
    
    for dataset_name, file_path in datasets:
        print(f"\nپردازش دیتاست {dataset_name}...")
        G = create_graph(file_path)  # اینجا G را ایجاد می‌کنیم
        print(f"تعداد گره‌ها در {dataset_name}: {G.number_of_nodes()}")
        print(f"تعداد یال‌ها در {dataset_name}: {G.number_of_edges()}")
        
        firefly_params, firefly_fitness, gwo_params, gwo_fitness, pso_params, pso_fitness = optimize_node2vec(G, dataset_name)
        
        df_results = compare_methods(G, dataset_name, firefly_params, gwo_params, pso_params)
        df_results['Dataset'] = dataset_name
        all_results.append(df_results)
        
        # انتخاب بهترین پارامترها بر اساس بالاترین امتیاز
        best_fitness = max(firefly_fitness, gwo_fitness, pso_fitness)
        if best_fitness == firefly_fitness:
            best_params = firefly_params
        elif best_fitness == gwo_fitness:
            best_params = gwo_params
        else:
            best_params = pso_params
        
        best_embeddings = node2vec_embedding(G.number_of_nodes(), list(G.edges()), 
                                             int(best_params[0]), int(best_params[1]), int(best_params[2]), 
                                             best_params[3], best_params[4])
        visualize_embeddings(G, best_embeddings, dataset_name)
    
    df_all_results = pd.concat(all_results)
    df_all_results.to_csv("all_comparison_results.csv")
    
    plt.figure(figsize=(15, 10))
    sns.barplot(x='Dataset', y='Combined Score', hue='Method', data=df_all_results)
    plt.title("مقایسه عملکرد روش‌های مختلف برای همه دیتاست‌ها")
    plt.savefig("all_datasets_comparison.png")
    plt.close()

if __name__ == "__main__":
    main()


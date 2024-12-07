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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
from karateclub import DeepWalk
import networkx as nx
from networkx.algorithms import community
try:
    from karateclub import LINE
    use_line = True
except ImportError:
    from karateclub import HOPE
    use_line = False
    print("LINE algorithm not available. Using HOPE as an alternative.")


# Set up input file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KARATE_FILE = 'c:\KARATE_FILE.csv'
DOLPHIN_FILE = 'c:\DOLPHIN_FILE.csv'
FACEBOOK_FILE = 'c:\FACEBOOK_FILE.csv'



memory = Memory(location='.cache', verbose=1)

def print_file_paths():
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"KARATE_FILE: {KARATE_FILE}")
    print(f"DOLPHIN_FILE: {DOLPHIN_FILE}")
    print(f"FACEBOOK_FILE: {FACEBOOK_FILE}")

    for file_path in [KARATE_FILE,DOLPHIN_FILE,FACEBOOK_FILE]:
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
    communities = list(community.greedy_modularity_communities(G))
    modularity = community.modularity(G, communities)
    
    sample_size = min(1000, len(embeddings))
    sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
    sample_embeddings = embeddings[sample_indices]
    avg_distance = np.mean(np.linalg.norm(sample_embeddings[:, None] - sample_embeddings, axis=2))
    
    return modularity, avg_distance, 0.7 * modularity + 0.3 * (1 / (avg_distance + 1e-6))

class GreyWolfOptimizer:
    def __init__(self, fitness_function, num_params, param_ranges, num_wolves=10, max_iter=5):
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
                print(f"Early stopping GWO at iteration {l}")
                break

        return self.scaler.inverse_transform(self.alpha_pos.reshape(1, -1))[0], self.alpha_score

def optimize_node2vec(G, dataset_name):
    param_ranges = [
        (4, 128),    # dimensions
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

    gwo = GreyWolfOptimizer(fitness_function, len(param_ranges), param_ranges, num_wolves=10, max_iter=5)
    gwo_params, gwo_fitness = gwo.optimize()

    print(f"\nBest Node2Vec parameters for {dataset_name} with GWO algorithm:")
    print(f"dimensions: {int(gwo_params[0])}")
    print(f"walk_length: {int(gwo_params[1])}")
    print(f"num_walks: {int(gwo_params[2])}")
    print(f"p: {gwo_params[3]:.2f}")
    print(f"q: {gwo_params[4]:.2f}")
    print(f"Best GWO score: {gwo_fitness:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(gwo.convergence) + 1), gwo.convergence, label='GWO')
    plt.title(f"Convergence curve for {dataset_name}")
    plt.xlabel("Iteration")
    plt.ylabel("Best score")
    plt.legend()
    plt.savefig(f"convergence_comparison_{dataset_name}.png")
    plt.close()

    return gwo_params, gwo_fitness

def deepwalk_embedding(G, dimensions=128, walk_length=80, num_walks=10):
    model = DeepWalk(walk_number=num_walks, walk_length=walk_length, dimensions=dimensions)
    model.fit(G)
    return model.get_embedding()

def line_or_hope_embedding(G, dimensions=128):
    if use_line:
        model = LINE(dimensions=dimensions)
    else:
        # Adjust dimensions for HOPE
        n_nodes = G.number_of_nodes()
        adjusted_dimensions = min(dimensions, n_nodes - 1)
        if adjusted_dimensions != dimensions:
            print(f"Adjusting HOPE dimensions from {dimensions} to {adjusted_dimensions} due to graph size.")
        model = HOPE(dimensions=adjusted_dimensions)
    
    model.fit(G)
    embeddings = model.get_embedding()
    
    # If dimensions were adjusted, pad the embeddings with zeros
    if not use_line and adjusted_dimensions != dimensions:
        padding = np.zeros((embeddings.shape[0], dimensions - adjusted_dimensions))
        embeddings = np.hstack((embeddings, padding))
    
    return embeddings

def compare_methods(G, dataset_name, gwo_params):
    default_params = [128, 80, 10, 1, 1]
    
    results = []
    
    for method, params in [("Default", default_params), ("GWO", gwo_params)]:
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
    
    # Add DeepWalk, LINE/HOPE, and VERSE embeddings
    for method, embedding_func in [("DeepWalk", deepwalk_embedding), 
                                   ("LINE/HOPE", line_or_hope_embedding)]:
        start_time = time.time()
        try:
            embeddings = embedding_func(G)
            modularity, avg_distance, combined_score = evaluate_embedding(G, embeddings)
            end_time = time.time()
            
            results.append({
                "Method": method,
                "Modularity": modularity,
                "Avg Distance": avg_distance,
                "Combined Score": combined_score,
                "Time (s)": end_time - start_time
            })
        except Exception as e:
            print(f"Error in {method}: {str(e)}")
            results.append({
                "Method": method,
                "Modularity": np.nan,
                "Avg Distance": np.nan,
                "Combined Score": np.nan,
                "Time (s)": np.nan
            })
    
    df_results = pd.DataFrame(results)
    print(f"\nComparison results for {dataset_name}:")
    print(df_results.to_string(index=False))
    
    df_results.to_csv(f"comparison_results_{dataset_name}.csv", index=False)
    
    return df_results

#def verse_embedding(G, dimensions=128):
    #model = VERSE(dimensions=dimensions)
    #model.fit(G)
    #return model.get_embedding()


def create_graph(file_path):
    df = pd.read_csv(file_path)
    if 'from' in df.columns and 'to' in df.columns:
        G = nx.from_pandas_edgelist(df, 'from', 'to')
    elif 'source' in df.columns and 'target' in df.columns:
        G = nx.from_pandas_edgelist(df, 'source', 'target')
    else:
        G = nx.from_pandas_edgelist(df, df.columns[0], df.columns[1])
    
    # Relabel nodes to ensure they are integers from 0 to n-1
    G = nx.convert_node_labels_to_integers(G)
    return G

def deepwalk_embedding(G, dimensions=128, walk_length=80, num_walks=10):
    # Ensure the graph has consecutive integer node labels
    G = nx.convert_node_labels_to_integers(G)
    model = DeepWalk(walk_number=num_walks, walk_length=walk_length, dimensions=dimensions)
    model.fit(G)
    return model.get_embedding()

# Similarly, update other embedding functions (line_or_hope_embedding, verse_embedding) 
# to include the relabeling step if needed.

def compare_methods(G, dataset_name, gwo_params):
    # Standard Node2Vec parameters
    #G = nx.convert_node_labels_to_integers(G)

    standard_node2vec_params = [128, 80, 10, 1, 1]  # dimensions, walk_length, num_walks, p, q
    
    results = []
    
    for method, params in [("Standard Node2Vec", standard_node2vec_params), ("GWO", gwo_params)]:
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
    
    # Add DeepWalk and LINE/HOPE embeddings
    for method, embedding_func in [("DeepWalk", deepwalk_embedding), 
                                   ("LINE/HOPE", line_or_hope_embedding)]:
        start_time = time.time()
        try:
            embeddings = embedding_func(G)
            modularity, avg_distance, combined_score = evaluate_embedding(G, embeddings)
            end_time = time.time()
            
            results.append({
                "Method": method,
                "Modularity": modularity,
                "Avg Distance": avg_distance,
                "Combined Score": combined_score,
                "Time (s)": end_time - start_time
            })
        except Exception as e:
            print(f"Error in {method}: {str(e)}")
            results.append({
                "Method": method,
                "Modularity": np.nan,
                "Avg Distance": np.nan,
                "Combined Score": np.nan,
                "Time (s)": np.nan
            })
    
    df_results = pd.DataFrame(results)
    print(f"\nComparison results for {dataset_name}:")
    print(df_results.to_string(index=False))
    
    df_results.to_csv(f"comparison_results_{dataset_name}.csv", index=False)
    
    return df_results

def visualize_embeddings(G, embeddings, dataset_name):
    n_samples = len(embeddings)
    
    if n_samples < 5:
        print(f"Not enough nodes in {dataset_name} for t-SNE visualization.")
        return
    
    perplexity = min(30, n_samples - 1)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    
    try:
        embeddings_2d = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
        plt.title(f"Embedding visualization for {dataset_name}")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.savefig(f"embeddings_visualization_{dataset_name}.png")
        plt.close()
        
        print(f"Visualization for {dataset_name} completed successfully.")
    except Exception as e:
        print(f"Error in visualization for {dataset_name}: {str(e)}")

def main():
    print_file_paths()
    
    datasets = [
        ("Karate Club", KARATE_FILE),
        ("Dolphin", DOLPHIN_FILE),
        ("FACEBOOK_FILE", FACEBOOK_FILE),

    ]
    
    all_results = []
    
    for dataset_name, file_path in datasets:
        print(f"\nProcessing dataset {dataset_name}...")
        G = create_graph(file_path)
        G = nx.convert_node_labels_to_integers(G)  

        print(f"Number of nodes in {dataset_name}: {G.number_of_nodes()}")
        print(f"Number of edges in {dataset_name}: {G.number_of_edges()}")
        
        gwo_params, gwo_fitness = optimize_node2vec(G, dataset_name)
    
        df_results = compare_methods(G, dataset_name, gwo_params)
        df_results['Dataset'] = dataset_name
        all_results.append(df_results)
        
        # Visualize embeddings for both standard Node2Vec and GWO-optimized Node2Vec
        standard_params = [128, 80, 10, 1, 1]
        standard_embeddings = node2vec_embedding(G.number_of_nodes(), list(G.edges()), 
                                                 int(standard_params[0]), int(standard_params[1]), 
                                                 int(standard_params[2]), standard_params[3], standard_params[4])
        visualize_embeddings(G, standard_embeddings, f"{dataset_name}_standard")
        
        gwo_embeddings = node2vec_embedding(G.number_of_nodes(), list(G.edges()), 
                                            int(gwo_params[0]), int(gwo_params[1]), int(gwo_params[2]), 
                                            gwo_params[3], gwo_params[4])
        visualize_embeddings(G, gwo_embeddings, f"{dataset_name}_gwo")
    
    df_all_results = pd.concat(all_results)
    df_all_results.to_csv("all_comparison_results.csv")
    
    plt.figure(figsize=(15, 10))
    sns.barplot(x='Dataset', y='Combined Score', hue='Method', data=df_all_results)
    plt.title("Performance comparison of different methods for all datasets")
    plt.savefig("all_datasets_comparison.png")
    plt.close()

if __name__ == "__main__":
    main()
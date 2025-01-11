import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec

from pyswarms.single.global_best import GlobalBestPSO
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# بارگذاری داده‌ها
features = pd.read_excel('c:\\out2.xlsx')
G = pd.read_excel('c:\\in2.xlsx')
nodes = features['node'].unique()
print("Data loaded successfully.")

G = nx.from_pandas_edgelist(G, source='from', target='to')

# تعریف مدل node2vec
node2vec = Node2Vec(G, dimensions=6, walk_length=30, num_walks=200, workers=4)

# آموزش مدل
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# تعریف پارامترهای PSO
n_particles = 10  # تعداد ذرات
dimensions = 3  # تعداد ابعاد (تعداد پارامترها)
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}  # پارامترهای PSO
bounds = ((0, 0, 0), (1, 1, 1))  # محدوده مجاز برای پارامترها

# PSO optimizer
optimizer = GlobalBestPSO(n_particles=n_particles, dimensions=dimensions, options=options, bounds=bounds)

# تابع Fitness
def objective_function(x):
    embeddings = model.wv  # Use model.wv instead of model.embedding
    distance_matrix = []
    for i in range(len(embeddings)):
        row = []
        for j in range(len(embeddings)):
            row.append(cosine_similarity(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))[0][0])
        distance_matrix.append(row)
    distance_matrix = np.array(distance_matrix)
    fitness = np.sum(distance_matrix)
    return fitness

# Optimize using PSO
cost, pos = optimizer.optimize(objective_function, iters=2)

# Use best parameters to train Node2Vec
dimensions = int(round(pos[0]))
walk_length = int(round(pos[1]))
num_walks = int(round(pos[2]))

# ساخت گراف
G = pd.read_excel('c:\\in2.xlsx')
G = nx.from_pandas_edgelist(G, source='from', target='to')



# آموزش مدل Node2Vec
node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=4)
model = node2vec.fit(window=10, min_count=1, batch_words=4)
    # Get node embeddings
embeddings = model.wv.vectors 

# استخراج embeddings گره‌ها
#node_embeddings = [model.embedding[node] for node in nodes]

# تعریف مدل رگرسیون لجستیک
classifier = LogisticRegression()

# آموزش مدل رگرسیون لجستیک
classifier.fit(embeddings, features['label'])

# پیش‌بینی برچسب‌ها
predicted_labels = classifier.predict(embeddings)

# ارزیابی نتایج
accuracy = accuracy_score(features['label'], predicted_labels)
f1 = f1_score(features['label'], predicted_labels, average='weighted')
precision = precision_score(features['label'], predicted_labels, average='weighted')
recall = recall_score(features['label'], predicted_labels, average='weighted')

# ذخیره خروجی
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write(f'Best parameters: {pos}\n')
    f.write(f'Accuracy: {accuracy}\n')
    f.write(f'F1 Score: {f1}\n')
    f.write(f'Precision: {precision}\n')
    f.write(f'Recall: {recall}\n')
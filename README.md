The codes presented on this page pertain to the implementation of an optimization method for the Node2Vec algorithm using the Grey Wolf Optimizer.
We provide several source codes and implementations, including our proposed method, comparative sources between different methods, and sources related to the implementation of the Grey Wolf Optimizer with Genetic Algorithms, Firefly Algorithm, and PSO.
Reproducibility Instructions
To ensure the reproducibility of our results, we provide detailed instructions and guidelines for replicating the experiments described in this manuscript. The code repository, available at https://github.com/mahdi2006kal/Node2Vec GWO, includes all necessary scripts and datasets. Below, we outline the steps required to reproduce the experiments:
1. Prerequisites:
   - Install Python 3.8 or higher.
   - Install the following Python libraries:
     - numpy (version 1.21.0)
     - networkx (version 2.5),pandas,gensim
     - matplotlib (version 3.4.2)
     - scikit-learn (version 0.24.2)
pip install numpy pandas matplotlib networkx gensim scikit-learn
   - Ensure the system supports multiprocessing for computationally intensive tasks.
2. Data Preparation:
   - Input datasets used in this study, including Karate Club, Dolphin Network, and Facebook Ego Network, are included in the repository under the `datasets/` directory .
3. Execution Steps:
   - Clone the repository using the command:
          - git clone https://github.com/example/Node2Vec-GWO.git
- cd Node2Vec-GWO
      - Run the main script for experiments:
 -     python chandalgorembandingver2.py--dataset KarateClub --alpha 7 --beta 3 --iterations 40
4. Output Evaluation:
   - Results, including modularity, average distance, accuracy, and F1 score, are saved in the `results/` directory. Graphical outputs are saved as `.png` files for visualization.
By following these steps, researchers can replicate our experiments and validate the results presented in this study.

Cite this article
Rabiei, M., Fartash, M. & Nazari, S. Correction: A hybrid optimization approach for graph embedding: leveraging Node2Vec and grey wolf optimization. J Supercomput 81, 650 (2025). https://doi.org/10.1007/s11227-025-07168-z

Download citation

Published
24 March 2025

DOI
https://doi.org/10.1007/s11227-025-07168-z

Share this article


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde

# Function to load genotype file and convert it to a numpy array
def load_geno_file(filepath):
    with open(filepath, 'r') as file:
        data = [list(line.strip()) for line in file.readlines()]
    geno_matrix = np.array(data, dtype=int)
    return geno_matrix

# Example usage
geno_file = "data\Q2\mixture1.geno"
geno = load_geno_file(geno_file)
print(geno.shape)

# Initialize parameters for the model
def initialize_parameters(K, num_sps):
    # Initialize frequencies with random values between 0.01 and 0.99 to avoid extreme values
    freq = np.random.uniform(0.01, 0.99, (num_sps, K))
    # Initialize mixing proportions (pi) using exponential distribution and normalize
    pi = np.random.exponential(1, K)
    pi /= np.sum(pi)
    return pi, freq

# E-Step: Compute responsibilities (posterior probabilities)
def e_step(geno, pi, freq, K):
    num_individuals = geno.shape[1]
    log_likelihood = np.zeros((num_individuals, K))
    
    # Ensure freq values are clipped to prevent issues with log(0) or log(1)
    freq = np.clip(freq, 1e-10, 1 - 1e-10)
    
    for k in range(K):
        # Calculate the log-probabilities to avoid numerical underflow
        log_prob = np.sum(geno.T * np.log(freq[:, k]) + (1 - geno.T) * np.log(1 - freq[:, k]), axis=1)
        log_likelihood[:, k] = log_prob + np.log(pi[k])  # Add log of mixing proportions
    
    # Compute the log-sum-exp trick for numerical stability
    max_log_likelihood = np.max(log_likelihood, axis=1, keepdims=True)
    log_sum_exp = max_log_likelihood + np.log(np.sum(np.exp(log_likelihood - max_log_likelihood), axis=1, keepdims=True))
    
    # Compute log responsibilities
    log_responsibilities = log_likelihood - log_sum_exp
    
    # Normalize in log space and convert back from log-space to normal space (responsibilities)
    responsibilities = np.exp(log_responsibilities)
    
    return responsibilities

# M-Step: Update the parameters (pi and freq)
def m_step(geno, responsibilities, K):
    Nk = np.sum(responsibilities, axis=0) + 1e-10  # Add small value to avoid divide by zero
    pi = Nk / np.sum(Nk)  # Update pi (mixing proportions)
    freq = (geno @ responsibilities) / Nk  # Update freq (feature probabilities)
    
    return pi, freq

# Log-likelihood computation
def compute_log_likelihood(geno, pi, freq, K):
    num_individuals = geno.shape[1]
    log_likelihood = np.zeros((num_individuals, K))
    
    # Clip freq values to prevent log(0)
    freq = np.clip(freq, 1e-10, 1 - 1e-10)
    
    for k in range(K):
        # Calculate log-probabilities to avoid underflow
        log_prob = np.sum(geno.T * np.log(freq[:, k]) + (1 - geno.T) * np.log(1 - freq[:, k]), axis=1)
        log_likelihood[:, k] = log_prob + np.log(pi[k])  # Add log(pi[k])
    
    # Compute log-sum-exp for numerical stability
    max_log_likelihood = np.max(log_likelihood, axis=1, keepdims=True)
    log_sum_exp = max_log_likelihood + np.log(np.sum(np.exp(log_likelihood - max_log_likelihood), axis=1, keepdims=True))
    
    # Return the sum of the log-sum-exp
    return np.sum(log_sum_exp)

# EM Algorithm: Run the Expectation-Maximization
def em_algorithm(geno, K, max_iter=100, tol=1e-8):
    pi, freq = initialize_parameters(K, geno.shape[0])
    
    log_likelihoods = []
    
    for i in range(max_iter):
        # E-step: Compute responsibilities
        responsibilities = e_step(geno, pi, freq, K)
        
        # M-step: Update parameters
        pi, freq = m_step(geno, responsibilities, K)
        
        # Compute log-likelihood
        log_likelihood = compute_log_likelihood(geno, pi, freq, K)
        log_likelihoods.append(log_likelihood)
        
        # Check for convergence (if log-likelihood change is below tolerance)
        if i > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            print("Converges")
            break
    
    return pi, freq, log_likelihoods, responsibilities

# Running EM Algorithm with 3 random restarts
import pandas as pd

def em_algorithm_with_restarts(geno, K, max_iter=100, tol=1e-8, restarts=3):
    results = []  # List to store results of each restart
    all_log_likelihoods = []  # List to store log-likelihoods for all restarts
    best_result = None
    best_log_likelihood = -np.inf

    for restart in range(restarts):
        print(f"Restart {restart + 1}")

        # Run EM algorithm
        pi, freq, log_likelihoods, responsibilities = em_algorithm(geno, K, max_iter, tol)

        # Get the last log-likelihood of this restart
        last_log_likelihood = log_likelihoods[-1]

        # Save results for this restart
        restart_data = {
            'restart': restart + 1,
            'log_likelihood': last_log_likelihood,
            'pi': pi,
            'freq': freq,
            'responsibilities': responsibilities,
            'log_likelihoods': log_likelihoods  # Store all log-likelihoods for plotting
        }

        results.append(restart_data)
        all_log_likelihoods.append(log_likelihoods)

        # Update best result if this one has the highest log-likelihood
        if last_log_likelihood > best_log_likelihood:
            best_log_likelihood = last_log_likelihood
            best_result = restart_data

    # Convert the results into a DataFrame for easier comparison
    results_df = pd.DataFrame(results)

    # Plot log-likelihoods for each restart
    plt.figure(figsize=(10, 6))

    for i, log_likelihoods in enumerate(all_log_likelihoods):
        plt.plot(log_likelihoods, label=f'Restart {i + 1}')

    plt.title('Log-Likelihoods for Different Restarts')
    plt.xlabel('Iterations')
    plt.ylabel('Log-Likelihood')
    plt.legend()
    plt.show()

    return results_df, best_result


# Function to load true frequencies from the .ganc file
def load_true_label_from_ganc(filepath, K):
    # Assuming the .ganc file is in a space-separated or tab-separated format
    true_label = np.loadtxt(filepath)  # Change this if the .ganc format is different
    return true_label

# Function to make predictions based on the posterior probabilities
def make_predictions(responsibilities):
    # For each individual, pick the population (cluster) with the maximum posterior probability
    predictions = np.argmax(responsibilities, axis=1)  # Max probability for each individual (row)
    return predictions

def calculate_accuracy(predicted_labels, true_labels):
    # Calculate accuracy for original and flipped labels
    accuracy_original = np.sum(predicted_labels == true_labels) / len(true_labels)
    accuracy_flipped = np.sum(predicted_labels == 1 - true_labels) / len(true_labels)  # Flip the labels
    
    # Return the maximum accuracy
    return max(accuracy_original, accuracy_flipped)

def evaluate_snp_subsets(geno, true_labels, snp_counts, K, max_iter=100, tol=1e-8, restarts=3):
    accuracies = []

    for num_snps in snp_counts:
        print(f"Running EM for {num_snps} SNPs")
        geno_subset = geno[:num_snps, :]
        results_df, best_result = em_algorithm_with_restarts(geno_subset, K, max_iter, tol, restarts)

        # Predict labels based on max posterior probability
        predicted_labels = make_predictions(best_result['responsibilities'])
        
        accuracies.append(calculate_accuracy(predicted_labels, true_labels))

    # Plot accuracy as a function of number of SNPs
    plt.figure()
    plt.plot(snp_counts, accuracies, marker='o')
    plt.xlabel('Number of SNPs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Number of SNPs')
    plt.show()

    return accuracies

def em_algorithm_varying_K(geno, max_K=4, max_iter=100, tol=1e-8):
    results = []

    for K in range(1, max_K + 1):
        print(f"Running EM for K={K}")

        # Run EM algorithm
        pi, freq, log_likelihoods, responsibilities = em_algorithm(geno, K, max_iter, tol)

        # Get the last log-likelihood for this K
        last_log_likelihood = log_likelihoods[-1]

        # Save results for this K
        results.append({'K': K, 'log_likelihood': last_log_likelihood})

    # Convert the results into a DataFrame for easier comparison
    results_df = pd.DataFrame(results)

    # Plot log-likelihoods for different values of K
    plt.figure(figsize=(8, 6))
    plt.plot(results_df['K'], results_df['log_likelihood'], marker='o')
    plt.title('Log-Likelihood vs. K')
    plt.xlabel('K (Number of Clusters)')
    plt.ylabel('Final Log-Likelihood')
    plt.xticks(results_df['K'])
    plt.grid(True)
    plt.show()

    return results_df


# Load genotype and ground truth labels
geno_file = "data/Q2/mixture1.geno"
geno = load_geno_file(geno_file)

ganc_file = "data/Q2/mixture1.ganc"
ground_truth_labels = np.loadtxt(ganc_file)

K = 2  # Number of clusters

# Run EM Algorithm with 3 restarts and accuracy calculation
results_df, best_result = em_algorithm_with_restarts(
    geno, K, max_iter=100, tol=1e-8, restarts=3)


# Print the results DataFrame (for inspection)
print(calculate_accuracy(make_predictions(results_df.loc[0, 'responsibilities']), ground_truth_labels[:,0]))
print(results_df.loc[0, 'pi'])
print(results_df.loc[0, 'log_likelihoods'][-1])

print(calculate_accuracy(make_predictions(results_df.loc[1, 'responsibilities']), ground_truth_labels[:,0]))
print(results_df.loc[1, 'pi'])
print(results_df.loc[1, 'log_likelihoods'][-1])

print(calculate_accuracy(make_predictions(results_df.loc[2, 'responsibilities']), ground_truth_labels[:,0]))
print(results_df.loc[2, 'pi'])
print(results_df.loc[2, 'log_likelihoods'][-1])

snp_counts = [10, 100, 1000, 5000]
accuracies = evaluate_snp_subsets(geno, ground_truth_labels[:,0], snp_counts, K=2)
print(accuracies)

geno_file = "data/Q2/mixture2.geno"
geno = load_geno_file(geno_file)

# Run EM Algorithm with 3 restarts and accuracy calculation
results_df, best_result = em_algorithm_with_restarts(
    geno, K, max_iter=100, tol=1e-8, restarts=3)


# Print the results DataFrame (for inspection)
print(results_df.loc[0, 'pi'])
print(results_df.loc[0, 'log_likelihoods'][-1])

print(results_df.loc[1, 'pi'])
print(results_df.loc[1, 'log_likelihoods'][-1])

print(results_df.loc[2, 'pi'])
print(results_df.loc[2, 'log_likelihoods'][-1])

em_algorithm_varying_K(geno)

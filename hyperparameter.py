import argparse
from simulate import simulate



def run_hyperparameter_tuning(
        model, 
        dataset,
        algorithm,
        similarity,
        times,
        dp,
        sigma_gaussian,
        num_glob_iters,
        time,
        number,
        dim_pca, 
        nb_users,
        nb_samples,
        sample_ratio,
        local_updates,
        user_ratio,
        weight_decay,
        local_learning_rate,
        max_norm,
        dim_input,
        dim_output, 
):

    # Initialize parameters
    femnist_dict = {"dataset": "Femnist",
                    "model": model,
                    "dim_input": 784,
                    "dim_pca": dim_pca,
                    "dim_output": 47,
                    "nb_users": nb_users,
                    "nb_samples": nb_samples,
                    "sample_ratio": sample_ratio,
                    "local_updates": local_updates,
                    "user_ratio": user_ratio,
                    "weight_decay": weight_decay,
                    "local_learning_rate": local_learning_rate,
                    "max_norm": max_norm}

    # MNIST DATA
    # Potential models : mclr, NN1, NN1_PCA

    mnist_dict = {"dataset": "Mnist",
                  "model": model,
                  "dim_input": 784,
                  "dim_pca": dim_pca,
                  "dim_output": 10,
                  "nb_users": nb_users,
                  "nb_samples": nb_samples,
                  "sample_ratio": sample_ratio,
                  "local_updates": local_updates,
                  "user_ratio": user_ratio,
                  "weight_decay": weight_decay,
                  "local_learning_rate": local_learning_rate,
                  "max_norm": max_norm}

    # CIFAR-10 DATA
    # Potential models : CNN

    cifar10_dict = {"dataset": "CIFAR_10",
                    "model": "CNN",
                    "dim_input": 1024,
                    "dim_pca": None,
                    "dim_output": 10,
                    "nb_users": nb_users,
                    "nb_samples": nb_samples,
                    "sample_ratio": sample_ratio,
                    "local_updates": local_updates,
                    "user_ratio": user_ratio,
                    "weight_decay": weight_decay,
                    "local_learning_rate": local_learning_rate,
                    "max_norm": max_norm}

    # SYNTHETIC DATA
    # only one model : mclr

    logistic_dict = {"dataset": "Logistic",
                     "model": "mclr",
                     "dim_input": dim_input,
                     "dim_pca": None,
                     "dim_output": dim_output,
                     "nb_users": nb_users,
                     "nb_samples": nb_samples,
                     "sample_ratio": sample_ratio,
                     "local_updates": local_updates,
                     "user_ratio": user_ratio,
                     "weight_decay": weight_decay,
                     "local_learning_rate": local_learning_rate,
                     "max_norm": max_norm}

    input_dict = {}

    if dataset == 'Femnist':
        input_dict = femnist_dict
    elif dataset == 'Mnist':
        input_dict = mnist_dict
    elif dataset == 'Logistic':
        input_dict = logistic_dict
    elif dataset == 'CIFAR_10':
        input_dict = cifar10_dict
    
    candidate_learning_rates = [0.01, 0.001, 0.0001]

    for learning_rate in candidate_learning_rates:
        print(f"Running hyperparameter tuning with learning rate: {learning_rate}")
        

        # Simulate Training 
        simulate(**input_dict, algorithm=algorithm, similarity=similarity, noise=False,
                 times=times, dp=dp, sigma_gaussian=sigma_gaussian,
                 num_glob_iters=num_glob_iters, time=time)

        
        print(f"Training completed with learning rate: {learning_rate}")

        # Append the results to a file or database

    # Read the file 

    # Here you would read the results file and print the best hyperparameters





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--times", type=int, default=3, help="Number of random runs for each setting")
    parser.add_argument("--time", type=int, default=None, choices=[None, 0, 1, 2],
                        help="(<times) : used to process the run chosen independently from the others. If None, every run is performed")
    parser.add_argument("--num_glob_iters", type=int, default=250, help="T: Number of communication rounds")

    parser.add_argument("--dataset", type=str, default="Logistic", choices=["Femnist", "Logistic", "CIFAR_10", "Mnist"])
    parser.add_argument("--algo", type=str, default="FedAvg", choices=["FedSGD", "FedAvg", "SCAFFOLD-warm", "SCAFFOLD"])
    parser.add_argument("--model", type=str, default="mclr", choices=["mclr", "NN1", "NN1_PCA", "CNN"],
                        help="Chosen model. If using PCA on data, add '_PCA' at the end of the name.")
    parser.add_argument("--similarity", type=float, default=0.1,
                        help="Level of similarity between user data (for Femnist, Mnist, CIFAR_10 datasets)")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="Level of heterogeneity between user model (for Logistic dataset), -1 for iid models")
    parser.add_argument("--beta", type=float, default=0.0,
                        help="Level of heterogeneity between user data (for Logistic dataset), -1 for iid data")
    parser.add_argument("--number", type=int, default=0,
                        help="Id of dataset (used to avoid overwriting if same similarity parameters are used)")

    parser.add_argument("--nb_users", type=int, default=100, help="M: Number of all users for FL")
    # In the paper: FEMNIST : 40 users / Logistic : 100 users
    parser.add_argument("--user_ratio", type=float, default=0.1,
                        help="l: Subsampling ratio for users at each communication round")
    parser.add_argument("--nb_samples", type=int, default=5000,
                        help="R: Number of all data points by user (conditionally to same_sample_size)")
    # In the paper: FEMNIST : 2500 samples / Logistic : 5000 samples
    parser.add_argument("--sample_ratio", type=float, default=0.2,
                        help="s: Subsampling ratio for data points at each local update")
    parser.add_argument("--local_updates", type=int, default=10,
                        help="K: Number of local updates per selected user (local_epochs=local_updates*sample_ratio)")

    # For Logistic dataset generation
    parser.add_argument("--dim_input", type=int, default=40, help="For synthetic data : size of data points")
    parser.add_argument("--dim_output", type=int, default=10, help="For synthetic data : nb of labels")
    parser.add_argument("--same_sample_size", type=int, default=1,
                        help="For synthetic data (generation): same sample size for all users?")
    # For both datasets generation
    parser.add_argument("--normalise", type=int, default=1,
                        help="If 1: Normalise every input at the generation of the data")
    parser.add_argument("--standardize", type=int, default=1,
                        help="If 1: Standardize the features by user at the generation of the data")

    parser.add_argument("--weight_decay", type=float, default=5e-3, help="Regularization term")
    parser.add_argument("--local_learning_rate", type=float, default=1.0,
                        help="Multiplicative factor in the learning rate for local updates (TO TUNE)")
    parser.add_argument("--max_norm", type=float, default=1.0,
                        help="Gradient clipping value (not used with the heuristic implemented by default)")

    parser.add_argument("--dp", type=str, default="None", choices=["None", "Gaussian"],
                        help="Differential Privacy or not")
    parser.add_argument("--sigma_gaussian", type=float, default=10.0, help="Gaussian standard deviation for DP noise")

    parser.add_argument("--dim_pca", type=int, default=60,
                        help="Nb of components for generate_pca (for MNIST and FEMNIST data)")

    args = parser.parse_args()

    run_hyperparameter_tuning(
        args.model, 
        args.dataset,
        args.algo,
        args.similarity,
        args.times,
        args.dp,
        args.sigma_gaussian,
        args.num_glob_iters,
        args.time,
        args.number,
        args.dim_pca,
        args.nb_users,
        args.nb_samples,
        args.sample_ratio,
        args.local_updates,
        args.user_ratio,
        args.weight_decay,
        args.local_learning_rate,
        args.max_norm,
        args.dim_input,
        args.dim_output, 
        )


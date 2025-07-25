python hyperparameter.py \
--times 1 \
--num_glob_iters 500 \
--dataset Femnist \
--model NN1_PCA \
--nb_users 40 \
--user_ratio 0.2 \
--nb_samples 2500 \
--sample_ratio 0.2 \
--dim_pca 60 \
--algo FedAvg \
--dp Gaussian \
--sigma_gaussian 50. \
--local_updates 10 \



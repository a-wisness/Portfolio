# Hyperparameter Sweep — ml-100k

NeuMF, 8 epochs per run, leave-one-out evaluation. Best NDCG@10 marked ★.

| gmf_dim | num_negatives | learning_rate | HR@10 | NDCG@10 |
|---|---|---|---:|---:|
| 32 | 8 | 0.001 ★ | 0.7282 | 0.4692 |
| 16 | 8 | 0.001 | 0.7304 | 0.4569 |
| 32 | 4 | 0.001 | 0.7261 | 0.4511 |
| 16 | 4 | 0.001 | 0.7134 | 0.4349 |

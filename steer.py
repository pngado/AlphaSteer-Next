from src.learn_subspace import OTSubspaceLearner
from src.calc_steering_matrix import SafetyEnhancedSteering
import torch

torch.manual_seed(42)

def main():
    # -------------------
    # 1. Prepare data
    # -------------------
    N, d, p, K = 600, 10, 3, 3
    # synthetic benign clusters
    centers = [torch.randn(d) * 5 for _ in range(K)]
    D_b = torch.cat([c + 0.3 * torch.randn(N // (2*K), d) for c in centers], dim=0)
    # synthetic malicious prompts scattered
    D_m = torch.randn(N // 2, d) + 2.0

    # -------------------
    # 2. Train OT-subspace learner
    # -------------------
    learner = OTSubspaceLearner(d, p, K, epsilon=0.1, lr=1e-2)
    learner.fit(torch.cat([D_b, D_m], dim=0), epochs=30)

    # Cluster assignments
    cluster_ids_b = learner.assign_clusters(D_b)
    cluster_ids_m = learner.assign_clusters(D_m)

    # -------------------
    # 3. Fit principal/residual subspaces
    # -------------------
    U, V = learner.fit_principal_subspaces(torch.cat([D_b, D_m], dim=0),
                                           learner.assign_clusters(torch.cat([D_b, D_m], dim=0)))

    # -------------------
    # 4. Learn Delta_k
    # -------------------
    # Partition benign/malicious by cluster
    D_bk = [D_b[cluster_ids_b == k] for k in range(K)]
    D_mk = [D_m[cluster_ids_m == k] for k in range(K)]

    steering = SafetyEnhancedSteering(U, V)
    steering.learn_delta(D_bk, D_mk, gamma=1e-2)

    # -------------------
    # 5. Steering examples
    # -------------------
    h_benign = D_b[0]
    h_mal = D_m[0]

    h_benign_steered = steering.steer(h_benign, lam=1.0)
    h_mal_steered = steering.steer(h_mal, lam=1.0)

    print("Original benign norm:", h_benign.norm().item(),
          "Steered benign norm:", h_benign_steered.norm().item())
    print("Original malicious norm:", h_mal.norm().item(),
          "Steered malicious norm:", h_mal_steered.norm().item())

if __name__ == "__main__":
    main()
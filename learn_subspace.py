import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# -----------------------------
# Utilities
# -----------------------------
def random_stiefel(d: int, p: int, device="cpu"):
    """
    Generate a random matrix on the Stiefel manifold St(p, n).
    
    Args:
        n (int): Number of rows.
        p (int): Number of columns (p <= n).
        device (str): Torch device.
    
    Returns:
        torch.Tensor: Orthonormal matrix of shape (n, p).
    """
    # Step 1: Generate a random Gaussian matrix
    A = torch.randn(d, p, device=device)

    # Step 2: Orthonormalize its columns via QR decomposition.
    Q, _ = torch.linalg.qr(A)  

    return Q

def projection_distance(X, W):
    """
    Compute squared distance ||x - W W^T x||^2
    X: (N, d) where N=number of samples in the batch, d=dimensionality of each sample
    W: (d, p)
    Returns: (N,)
    """
    proj = X @ W @ W.T  # (N,d)
    return ((X - proj) ** 2).sum(dim=1) # sum across the feature dimension to get a scalar error per sample

# -----------------------------
# OT-based Subspace Learner
# -----------------------------
class OTSubspaceLearner:
    def __init__(self, d, p, K, epsilon=0.1, lr=1e-2, device="cpu"):
        self.d, self.p, self.K = d, p, K
        self.eps = epsilon
        self.device = device

        # Initialize subspaces
        self.W = [random_stiefel(d, p, device) for _ in range(K)]
        # Dual variables g_k
        self.g = torch.zeros(K, device=device, requires_grad=True)

        # Optimizer for g
        self.opt_g = optim.Adam([self.g], lr=lr)
        self.lr = lr

        self.U, self.V = None, None

    def loss_dual(self, X):
        """
        Eq. (2) from the paper:
        (1/K) * sum_k g_k
        - (eps/N) * sum_n log( (1/K) * sum_k exp((-c(x_n, W_k) + g_k)/eps) )
        """
        N, d = X.shape

        # compute costs c(x, W) = ||x - WW^T x||^2
        costs = []
        for W in self.W:
            proj = X @ W @ W.T
            residuals = X - proj
            cost = (residuals ** 2).sum(dim=1)  # (N,)
            costs.append(cost)
        costs = torch.stack(costs, dim=1)  # (N, K)

        # first term: (1/K) sum_k g_k
        term1 = self.g.mean()

        # second term: -(eps/N) * sum_n log( (1/K) sum_k exp(...) )
        logits = (-costs + self.g) / self.eps
        log_mean_exp = torch.logsumexp(logits, dim=1) - torch.log(
            torch.tensor(self.K, dtype=X.dtype, device=X.device)
        )
        term2 = -(self.eps / N) * torch.sum(log_mean_exp)

        return term1 + term2

    def step(self, X, g_steps=5):
        # === A) Update dual variables g (maximize) ===
        for _ in range(g_steps):
            self.opt_g.zero_grad()
            obj = self.loss_dual(X)
            loss = -obj  # maximize obj wrt g
            loss.backward()
            self.opt_g.step()

        # === B) Update subspaces W (minimize) ===
        new_Ws = []
        for k in range(self.K):
            Wk = self.W[k].clone().detach().requires_grad_(True)
            optimizer = optim.SGD([Wk], lr=self.lr)

            optimizer.zero_grad()
            obj = self.loss_dual(X)  # use global OT loss
            obj.backward()

            optimizer.step()

            # Retract to Stiefel manifold via QR
            with torch.no_grad():
                Q, _ = torch.linalg.qr(Wk)
            new_Ws.append(Q)

        self.W = new_Ws

    def fit(self, X, epochs=50):
        for epoch in range(epochs):
            self.step(X)
            if (epoch + 1) % 10 == 0:
                obj = self.loss_dual(X)
                print(f"[Epoch {epoch+1}] OT objective: {obj.item():.4f}")

    def assign_clusters(self, X):
        """Assign each data point to closest subspace by cost."""
        costs = torch.stack([projection_distance(X, W) for W in self.W], dim=1)
        assignment = costs.argmin(dim=1)
        return assignment
    
    def refine_subspaces(self, X, alpha=0.1):
        """
        Perform subspace refinement as in Sec 4.1
        Compute principal (U_k) and residual (V_k) subspaces.
        """
        cluster_ids = self.assign_clusters(X)
        U_list, V_list = [], []
        for k in range(self.K):
            Xk = X[cluster_ids == k]
            if Xk.shape[0] == 0:
                U_list.append(torch.eye(self.d, device=self.device))
                V_list.append(torch.zeros(self.d, self.d, device=self.device))
                continue

            # covariance
            C = Xk.T @ Xk
            eigvals, eigvecs = torch.linalg.eigh(C)
            idx = eigvals.argsort(descending=True)
            eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

            # keep enough to preserve (1-alpha) variance
            total = eigvals.sum()
            cutoff = (torch.cumsum(eigvals, 0) / total >= 1 - alpha).nonzero()[0].item() + 1

            U_list.append(eigvecs[:, :cutoff])
            V_list.append(eigvecs[:, cutoff:])

        self.U, self.V = U_list, V_list
        return U_list, V_list
    
if __name__ == "__main__":
    X = random_stiefel(5, 3)  # A random 5x3 matrix with orthonormal columns
    print(X)
    print("Check orthonormality:", X.T @ X)  # should be close to I
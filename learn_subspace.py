import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

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

def projection_distance_centered(X, W, m):
    """
    Compute squared distance ||x - m - W W^T (x - m)||^2 for centered subspaces
    X: (N, d) where N=number of samples in the batch, d=dimensionality of each sample
    W: (d, p) subspace basis
    m: (d,) mean/origin of the subspace
    Returns: (N,)
    """
    X_centered = X - m  # (N, d)
    proj = X_centered @ W @ W.T  # (N, d)
    return ((X_centered - proj) ** 2).sum(dim=1) # sum across the feature dimension

# -----------------------------
# OT-based Subspace Learner
# -----------------------------
class OTSubspaceLearner:
    def __init__(self, d, p, K, epsilon=0.1, lr_g=1e-2, lr_W=1e-3, device="cpu", use_wandb=False, layer=None):
        self.d, self.p, self.K = d, p, K
        self.eps = epsilon
        self.device = device
        self.lr_g = lr_g
        self.lr_W = lr_W
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.layer = layer

        # Initialize subspaces
        self.W = [random_stiefel(d, p, device).requires_grad_(True) for _ in range(K)]
        # DEBUG
        self._prev_W = [Wk.detach().cpu().clone() for Wk in self.W]  # snapshot for relative-change
        self.history = {
            'obj': [], 'gradW_mean': [], 'gradg_norm': [], 'relW_mean': [],
            'ortho_mean': [], 'entropy': [], 'hard_change': []
        }
        # Convergence tracking
        self.g_grad_history = []
        # Initialize learnable means/origins for each subspace
        self.m = [torch.zeros(d, device=device, requires_grad=True) for _ in range(K)]
        # Dual variables g_k
        self.g = torch.zeros(K, device=device, requires_grad=True)

        # Optimizer for g
        self.opt_g = optim.Adam([self.g], lr=lr_g)
        # Optimizers for each (W_k, m_k) pair
        self.opt_Wm = [optim.Adam([self.m[k]], lr=lr_g) for k in range(K)]

        self.U, self.V = None, None

        self._prev_m = [mi.detach().cpu().clone() for mi in self.m]
        self.history_m_changes = [[] for _ in range(self.K)]

    def loss_dual(self, X):
        """
        Eq. (2) from the paper, modified for centered subspaces:
        (1/K) * sum_k g_k
        - (eps/N) * sum_n log( (1/K) * sum_k exp((-c(x_n, W_k, m_k) + g_k)/eps) )
        where c(x, W, m) = ||x - m - WW^T (x - m)||^2
        """
        N, d = X.shape

        # compute costs c(x, W, m) = ||x - m - WW^T (x - m)||^2
        costs = []
        for W, m in zip(self.W, self.m):
            X_centered = X - m  # (N, d)
            proj = X_centered @ W @ W.T
            residuals = X_centered - proj
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
    
    def primal_objective(self, X):
        """
        Compute the primal objective: optimal assignment cost.
        For each point, assign it to the best subspace and sum the costs.
        This should decrease during training.
        """
        N = X.shape[0] # number of data samples
        
        # Compute costs for each point to each subspace
        costs = []
        for W, m in zip(self.W, self.m):
            cost = projection_distance_centered(X, W, m)  # (N,)
            costs.append(cost)
        costs = torch.stack(costs, dim=1)  # (N, K)
        
        # For each point, take the minimum cost across all subspaces
        min_costs = costs.min(dim=1)[0]  # (N,)
        
        # Return average cost
        return min_costs.mean()
    
    def primal_objective_regularized(self, X):
        """
        Compute the entropic regularized primal cost:
        sum_{n,k} pi_{nk} * c(x_n, W_k, m_k) + eps * sum_{n,k} pi_{nk} (log pi_{nk} - log(1/K))
        """
        N, d = X.shape

        # costs (N, K)
        costs = []
        for W, m in zip(self.W, self.m):
            X_centered = X - m
            proj = X_centered @ W @ W.T
            residuals = X_centered - proj
            cost = (residuals ** 2).sum(dim=1)  # (N,)
            costs.append(cost)
        costs = torch.stack(costs, dim=1)

        # soft assignments pi (N, K)
        logits = (-costs + self.g) / self.eps
        log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        pi = torch.exp(log_probs)  # softmax over k

        # expected cost
        transport_cost = (pi * costs).sum() / N

        # entropy term (relative to uniform 1/K)
        entropy = -(pi * log_probs).sum() / N
        reg_primal = transport_cost + self.eps * entropy

        return reg_primal

    def converged_g(self, tolerance=1e-4, min_steps=5):
        """Check if inner g optimization has converged"""
        if len(self.g_grad_history) < min_steps:
            return False

        # Check last few gradient norms
        recent_grads = self.g_grad_history[-5:]
        if len(recent_grads) < 5:
            return False

        # Relative change in gradient norm
        grad_changes = []
        for i in range(1, len(recent_grads)):
            prev_grad, curr_grad = recent_grads[i-1], recent_grads[i]
            if prev_grad > 1e-12:  # Avoid division by zero
                rel_change = abs(curr_grad - prev_grad) / prev_grad
                grad_changes.append(rel_change)

        return all(change < tolerance for change in grad_changes)

    def converged_outer(self, tolerance=1e-4, window=5):
        """Check convergence based on multiple criteria"""
        if len(self.history['obj']) < window:
            return False

        recent_window = self.history['obj'][-window:]

        # 1. Objective change
        obj_changes = []
        for i in range(1, len(recent_window)):
            if abs(recent_window[i-1]) > 1e-12:
                rel_change = abs(recent_window[i] - recent_window[i-1]) / abs(recent_window[i-1])
                obj_changes.append(rel_change)

        obj_converged = all(change < tolerance for change in obj_changes)
        return obj_converged

    def step(self, X, max_g_steps=1000, g_tolerance=1e-4, print_every=1):
        # === A) Inner loop with adaptive stopping ===
        inner_converged = False

        for g_step in range(max_g_steps):
            self.opt_g.zero_grad()
            obj = self.loss_dual(X)
            loss = -obj  # maximize obj wrt g
            loss.backward()

            # Track gradient norm
            g_grad_norm = float(self.g.grad.norm().item())
            self.g_grad_history.append(g_grad_norm)

            self.opt_g.step()

            # Check inner convergence after minimum steps
            if g_step >= 10 and self.converged_g(g_tolerance):
                if print_every:
                    print(f"  Inner loop converged at step {g_step+1}")
                inner_converged = True
                break

        # Final gradient norm for diagnostics
        gradg_norm = self.g_grad_history[-1] if self.g_grad_history else None

        # === B) Update subspaces W and means m (minimize) ===
        # 1) zero grads
        for opt in self.opt_Wm:
            opt.zero_grad()

        # 2) Single forward/backward to compute gradients w.r.t W and m
        obj = self.loss_dual(X)
        obj.backward()

        # Diagnostics before parameter updates
        gradW_norms = []
        for Wk in self.W:
            if Wk.grad is None:
                gradW_norms.append(0.0)
            else:
                gradW_norms.append(float(Wk.grad.norm().cpu().item()))
        gradW_mean = float(np.mean(gradW_norms))

        # 3) update m_k (via persistent optimizers)
        for opt in self.opt_Wm:
            opt.step()

        # 4) update W_k manually + retraction
        with torch.no_grad():
            for k in range(self.K):
                # Check if gradient exists
                if self.W[k].grad is not None:
                    # gradient step
                    self.W[k] -= self.lr_W * self.W[k].grad
                    # retract to Stiefel manifold
                    Q, _ = torch.linalg.qr(self.W[k])
                    self.W[k].copy_(Q)   # in-place overwrite, keeps Parameter object
                    # clear gradient after manual update to avoid accumulating stale grads
                    self.W[k].grad.zero_()
                else:
                    print(f"Warning: W[{k}] has no gradient")


        # ----- Diagnostics after update -----
        # Recompute obj with updated W and m for accurate diagnostics (to match primal objectives)
        with torch.no_grad():
            obj_after_update = self.loss_dual(X)

        # relative change in W (Frobenius) using CPU snapshots
        rel_changes = []
        for k in range(self.K):
            cur = self.W[k].detach().cpu()
            prev = self._prev_W[k]
            denom = (prev.norm().item() + 1e-12)
            rel = float((cur - prev).norm().item() / denom)
            rel_changes.append(rel)
            # update snapshot
            self._prev_W[k] = cur.clone()

        relW_mean = float(np.mean(rel_changes))

        # orthogonality check: ||W^T W - I||_F per W_k
        ortho_errs = []
        for Wk in self.W:
            WT_W = Wk.T @ Wk
            I = torch.eye(WT_W.shape[0], device=WT_W.device)
            ortho_errs.append(float(torch.norm(WT_W - I).cpu().item()))
        ortho_mean = float(np.mean(ortho_errs))

        # Save diagnostics to history (using obj after update for consistency with primal objectives)
        self.history['obj'].append(float(obj_after_update.detach().cpu().item()))
        self.history['gradW_mean'].append(gradW_mean)
        self.history['gradg_norm'].append(gradg_norm if gradg_norm is not None else float('nan'))
        self.history['relW_mean'].append(relW_mean)
        self.history['ortho_mean'].append(ortho_mean)

        diagnostics = {
            'obj': self.history['obj'][-1],
            'gradW_mean': gradW_mean,
            'gradg_norm': gradg_norm,
            'relW_mean': relW_mean,
            'ortho_mean': ortho_mean,
        }

        # print concise diagnostics (if desired)
        if print_every and (print_every <= 1):
            print(f"[step] obj={diagnostics['obj']:.6e}, gradW_mean={diagnostics['gradW_mean']:.3e}, "
                f"relW={diagnostics['relW_mean']:.3e}, ortho={diagnostics['ortho_mean']:.3e}, ")

        return diagnostics, inner_converged



    def fit(self, X, max_epochs=300, outer_tolerance=1e-4, patience=5):
        consecutive_converged = 0

        for epoch in range(max_epochs):
            diagnostics, inner_converged = self.step(X)

            # Check outer convergence
            if self.converged_outer(outer_tolerance):
                consecutive_converged += 1
                if consecutive_converged >= patience:
                    print(f"Training converged at epoch {epoch+1}")
                    break
            else:
                consecutive_converged = 0

            # Compute primal objectives for logging (dual already computed in step)
            primal_reg = self.primal_objective_regularized(X)
            primal_unreg = self.primal_objective(X)

            # Log to wandb every epoch
            if self.use_wandb:
                wandb.log({
                    f'layer_{self.layer}/dual_obj': diagnostics['obj'],
                    f'layer_{self.layer}/gradW_mean': diagnostics['gradW_mean'],
                    f'layer_{self.layer}/gradg_norm': diagnostics['gradg_norm'],
                    f'layer_{self.layer}/relW_mean': diagnostics['relW_mean'],
                    f'layer_{self.layer}/ortho_mean': diagnostics['ortho_mean'],
                    f'layer_{self.layer}/primal_reg': primal_reg.item(),
                    f'layer_{self.layer}/primal_unreg': primal_unreg.item(),
                    f'layer_{self.layer}/converged_count': consecutive_converged,
                }, step=epoch)

            # Print every 5 epochs
            if (epoch + 1) % 5 == 0:
                print(f"[Epoch {epoch+1}] Dual: {diagnostics['obj']:.6f}, "
                      f"Primal_reg: {primal_reg.item():.6f}, "
                      f"Primal_unreg: {primal_unreg.item():.6f}, "
                      f"Converged_count: {consecutive_converged}/{patience}")

        return self

    def assign_clusters(self, X):
        """Assign each data point to closest subspace by cost."""
        costs = torch.stack([projection_distance_centered(X, W, m) for W, m in zip(self.W, self.m)], dim=1)
        assignment = costs.argmin(dim=1)
        return assignment
    
    def refine_subspaces(self, X, alpha=0.1):
        """
        Perform subspace refinement as in Sec 4.1
        Compute principal (U_k) and residual (V_k) subspaces.
        Now uses centered data for each cluster.
        Skips empty clusters entirely.
        """
        cluster_ids = self.assign_clusters(X)
        U_list, V_list = [], []
        valid_cluster_indices = []  # Track which clusters actually have data

        for k in range(self.K):
            Xk = X[cluster_ids == k]
            print(f"Cluster {k}: {Xk.shape[0]} samples assigned")

            if Xk.shape[0] == 0:
                print(f"Cluster {k}: Empty cluster, skipping PCA")
                continue  # Skip empty clusters entirely

            # Record this as a valid cluster
            valid_cluster_indices.append(k)

            # Center data around the learned mean for this subspace
            Xk_centered = Xk - self.m[k]

            # covariance of centered data
            C = Xk_centered.T @ Xk_centered
            eigvals, eigvecs = torch.linalg.eigh(C)
            idx = eigvals.argsort(descending=True)
            eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

            # Debug eigenvalue information
            total = eigvals.sum()
            cumsum_ratios = torch.cumsum(eigvals, 0) / total

            # print(f"Cluster {k} eigenvalue analysis:")
            # print(f"  Total variance: {total.item():.6f}")
            # print(f"  Num eigenvalues: {len(eigvals)}")
            # print(f"  Top 10 eigenvalues: {eigvals[:10].tolist()}")
            # print(f"  Bottom 5 eigenvalues: {eigvals[-5:].tolist()}")
            # print(f"  Top 10 cumsum ratios: {cumsum_ratios[:10].tolist()}")
            # print(f"  Target threshold (1-alpha): {1-alpha}")

            # Check which eigenvalues meet the threshold
            valid_indices = (cumsum_ratios >= 1 - alpha).nonzero()
            # print(f"  Valid indices meeting threshold: {valid_indices.flatten().tolist()}")

            if len(valid_indices) == 0:
                # print(f"  ERROR: No eigenvalues meet threshold {1-alpha}")
                # print(f"  Max cumsum ratio achieved: {cumsum_ratios.max().item()}")
                # Temporary fallback - we'll fix this after debugging
                cutoff = len(eigvals) - 1  # Leave at least 1 for residual
                # print(f"  Using fallback cutoff: {cutoff}")
            else:
                cutoff = valid_indices[0].item() + 1
                # print(f"  Using cutoff: {cutoff}")

            # print(f"  Principal dim: {cutoff}, Residual dim: {len(eigvals) - cutoff}")
            # print()

            U_list.append(eigvecs[:, :cutoff])
            V_list.append(eigvecs[:, cutoff:])

        # Update the learner to only store valid subspaces
        self.U, self.V = U_list, V_list
        self.valid_cluster_indices = valid_cluster_indices
        self.K_effective = len(valid_cluster_indices)  # Actual number of subspaces with data

        print(f"Subspace refinement completed:")
        print(f"  Original K: {self.K}")
        print(f"  Effective K: {self.K_effective}")
        print(f"  Valid cluster indices: {valid_cluster_indices}")

        # Log refinement results to wandb
        if self.use_wandb:
            cluster_sizes = [int((cluster_ids == k).sum().item()) for k in valid_cluster_indices]
            principal_dims = [U.shape[1] for U in U_list]
            residual_dims = [V.shape[1] for V in V_list]

            wandb.log({
                f'layer_{self.layer}/K_effective': self.K_effective,
                f'layer_{self.layer}/K_original': self.K,
                f'layer_{self.layer}/cluster_sizes': cluster_sizes,
                f'layer_{self.layer}/principal_dims_mean': np.mean(principal_dims),
                f'layer_{self.layer}/residual_dims_mean': np.mean(residual_dims),
            })

            # Log individual cluster statistics
            for i, k in enumerate(valid_cluster_indices):
                wandb.log({
                    f'layer_{self.layer}/cluster_{k}_size': cluster_sizes[i],
                    f'layer_{self.layer}/cluster_{k}_principal_dim': principal_dims[i],
                    f'layer_{self.layer}/cluster_{k}_residual_dim': residual_dims[i],
                })

        return U_list, V_list
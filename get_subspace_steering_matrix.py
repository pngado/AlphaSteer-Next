import torch

class SafetyEnhancedSteering:
    def __init__(self, U, V, m, device="cpu"):
        """
        U: list of principal subspace bases for each cluster
        V: list of residual subspace bases for each cluster
        m: list of subspace origins/means for each cluster
        """
        self.U = U
        self.V = V
        self.m = m
        self.K = len(U)
        self.d = U[0].shape[0]
        self.device = device

        # Initialize Delta_k as identity first
        self.Delta = [torch.eye(self.d, device=device) for _ in range(self.K)]

    def learn_delta(self, D_b, D_m, gamma=1e-3):
        """
        Learn Delta_k for each cluster from benign + malicious prompts
        D_b, D_m: lists of tensors, each entry corresponds to one cluster
          - D_b[k]: benign samples assigned to cluster k, shape (N_bk, d)
          - D_m[k]: malicious samples assigned to cluster k, shape (N_mk, d)
        gamma: regularization weight
        """
        for k in range(self.K):
            if D_m[k].shape[0] == 0: # note: not sure about the logic here
                continue  # no malicious data in this cluster

            Vk = self.V[k]
            Pk = Vk @ Vk.T
            mk = self.m[k]  # subspace origin

            # Center data around learned subspace origin
            D_b_centered = D_b[k] - mk  # benign data centered
            D_m_centered = D_m[k] - mk  # malicious data centered

            # Convert centered malicious set into matrix (d, N_mk)
            Dm_k = D_m_centered.T  # transpose to (d, N_mk)

            # Define refusal vector r_k in centered coordinates
            if D_b[k].shape[0] > 0:
                r_k = D_b_centered.mean(dim=0) - D_m_centered.mean(dim=0)  # (d,)
            # edge case: cluster has no benign samples
            else:
                r_k = -Dm_k.mean(dim=1)  # fallback: push malicious away from subspace origin

            # Stack r_k to get R_k (d, N_mk)
            Rk = r_k.unsqueeze(1).repeat(1, Dm_k.shape[1]) # Rk shape: (d, N_mk)

            # Eq. (6): closed form solution with centered data
            A = Pk @ Dm_k          # (d, N_mk)
            G = A @ A.T + gamma * (Pk @ Pk.T)  # (d, d)
            G_pinv = torch.linalg.pinv(G) # compute pseudo-inverse

            Delta_k = Rk @ Dm_k.T @ Pk.T @ G_pinv
            self.Delta[k] = Delta_k

    def closest_subspace(self, h):
        """
        Find the subspace closest to a prompt activation h.
        Args:
        h: (d,) input embdding vector
        """
        dists = []
        for k in range(self.K):
            Vk = self.V[k]
            mk = self.m[k]
            h_centered = h - mk  # center around subspace origin
            proj_res = Vk @ (Vk.T @ h_centered) # compute projection onto residual subspace
            dists.append((proj_res**2).sum().item())
        return int(torch.tensor(dists).argmin().item())

    def steer(self, h, lam=1.0):
        # Apply steering using learned Delta_k with centered projections
        k = self.closest_subspace(h)
        Vk = self.V[k]
        mk = self.m[k]
        Pk = Vk @ Vk.T
        Delta_k = self.Delta[k]
        h_centered = h - mk  # center around subspace origin
        return h + lam * (Delta_k @ (Pk @ h_centered)) # Equation (3) with centering
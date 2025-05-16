import torch

__all__ = [
    "null_space_projection_l", "cal_P", 
    "cal_tilde_delta_l", "cal_tilde_delta", 
    "cal_tilde_delta_with_regularization_l", "cal_tilde_delta_with_regularization", 
    "cal_steering_matrix_l", "cal_steering_matrix"]

def null_space_l(A, min_null_space_ratio=0.1, abs_nullspace_ratio=0.0):
    """
    Calculate the null space of matrix A.
    
    Parameters:
    - A: Input matrix
    - min_null_space_ratio: Minimum ratio of null space dimension to matrix dimension
    - abs_nullspace_ratio: Absolute ratio of null space dimension to use if > 0
    
    Returns:
    - Q: Orthonormal basis for the null space
    """
    _, S, Vh = torch.linalg.svd(A.T @ A)
    M, N = A.shape[0], A.shape[1]
    
    if abs_nullspace_ratio > 0:
        num = int(N * abs_nullspace_ratio)
    
    else:    
        S_ = torch.sqrt(S)
        rcond = torch.finfo(S.dtype).eps * max(M, N)
        tol = torch.amax(S_) * rcond
        num = torch.sum(S_ < tol)
    
        if num / N < min_null_space_ratio:
            num = int(N * min_null_space_ratio)
    
    print(f"final null space ratio: {num / N}")
    Q = Vh[-num:,:].T.conj()
    return Q

def null_space_projection_l(A, min_null_space_ratio=0.1, abs_nullspace_ratio=0.0):
    """
    Calculate the projection matrix onto the null space of A.
    
    Parameters:
    - A: Input matrix
    - min_null_space_ratio: Minimum ratio of null space dimension to matrix dimension
    - abs_nullspace_ratio: Absolute ratio of null space dimension to use if > 0
    
    Returns:
    - P: Projection matrix onto the null space
    """
    Q = null_space_l(A, min_null_space_ratio, abs_nullspace_ratio)
    P = Q @ Q.T
    return P


def cal_P(H_b, layers, min_nullspace_ratio=0.1, abs_nullspace_ratio=0.0, device="cuda:0"):
    """
    Calculate projection matrices for multiple layers.
    
    Parameters:
    - H_b: Feature matrix of benign behaviors [batch_size, num_layers, hidden_dim]
    - layers: List of layer indices to process
    - min_nullspace_ratio: Minimum ratio of null space dimension to matrix dimension
    - abs_nullspace_ratio: Absolute ratio of null space dimension to use if > 0
    - device: Device to perform computation on
    
    Returns:
    - P: Stack of projection matrices for each layer [num_layers, hidden_dim, hidden_dim]
    """
    H_b = H_b.to(device)
    P = []
    for layer in layers:
        P_layer = null_space_projection_l(
            H_b[:, layer, :], 
            min_null_space_ratio=min_nullspace_ratio, 
            abs_nullspace_ratio=abs_nullspace_ratio)
        P.append(P_layer)
    P = torch.stack(P, dim=0)
    return P


def cal_tilde_delta_l(H_h_layer, P_layer, refusal_vector, device="cuda:0"):
    '''
    Calculate tilde_delta for a single layer.
    
    Parameters:
    - H_h_layer: [batch_size, hidden_dim] - Feature matrix of harmful behaviors
    - P_layer: [hidden_dim, hidden_dim] - Projection matrix
    - refusal_vector: [hidden_dim] - Target refusal vector
    - device: Device to perform computation on
    
    Goal: Solve H_h_layer @ P_layer @ tilde_delta_layer = refusal_vector using pseudoinverse
    
    Returns:
    - tilde_delta_layer: Solution vector
    '''
    H_h_layer = H_h_layer.to(device)
    P_layer = P_layer.to(device)
    refusal_vector = refusal_vector.to(device)
    
    # tilde_delta_layer = torch.linalg.solve(H_h_layer @ P_layer, refusal_vector)
    tilde_delta_layer = torch.linalg.pinv(H_h_layer @ P_layer) @ refusal_vector
    result = H_h_layer @ P_layer @ tilde_delta_layer
    avg_reconstruction_error = torch.norm(result - refusal_vector) / H_h_layer.shape[0]
    print(f"avg_reconstruction_error: {avg_reconstruction_error}", end="\t")
    print(f"refusal_vector norm: {torch.norm(refusal_vector)}")
    
    return tilde_delta_layer

def cal_tilde_delta(H_h, P, refusal_vectors, layers, device="cuda:0"):
    '''
    Calculate the steering vector using pseudoinverse method.
    
    Parameters:
    - H_h: [b, l, t] - Feature matrix of harmful behaviors, where b is batch size, l is number of layers, t is feature dimension
    - P: [l, t, t] - Projection matrices for each layer
    - refusal_vectors: [l, t] - Target refusal vectors
    - layers: List of layer indices to process
    
    Optimization objective:
    We want to find tilde_delta such that H_h @ P @ tilde_delta ≈ refusal_vectors
    This is solved using the Moore-Penrose pseudoinverse: tilde_delta = pinv(H_h @ P) @ refusal_vectors
    '''
    H_h = H_h.to(device)
    P = P.to(device)
    refusal_vectors = refusal_vectors.to(device)

    tilde_delta_list = []
    for layer in layers:
        print(f"layer {layer}:", end="\t")
        tilde_delta_layer = cal_tilde_delta_l(
            H_h[:, layer, :], 
            P[layer], 
            refusal_vectors[layer], 
            device=device)
        
        tilde_delta_list.append(tilde_delta_layer)
    tilde_delta = torch.stack(tilde_delta_list, dim=0)
    return tilde_delta


def cal_tilde_delta_with_regularization_l(
    H_h_layer, P_layer, refusal_vector, lambda_reg, device="cuda:0"):
    '''
    Calculate tilde_delta for a single layer with regularization.
    
    Parameters:
    - H_h_layer: [batch_size, hidden_dim] - Feature matrix of harmful behaviors
    - P_layer: [hidden_dim, hidden_dim] - Projection matrix
    - refusal_vector: [hidden_dim] - Target refusal vector
    - lambda_reg: float - Regularization parameter
    - device: Device to perform computation on
    
    Returns:
    - tilde_delta_layer: Regularized solution vector
    '''
    X = H_h_layer @ P_layer
    A = X.T @ X + lambda_reg * (P_layer.T @ P_layer)
    b = X.T @ refusal_vector.repeat(X.shape[0], 1)
    tilde_delta_layer = torch.linalg.pinv(A) @ b
    result = X @ tilde_delta_layer
    avg_reconstruction_error = torch.norm(result - refusal_vector) / X.shape[0]
    print(f"avg_reconstruction_error: {avg_reconstruction_error}", end="\t")
    print(f"refusal_vector norm: {torch.norm(refusal_vector)}")
    return tilde_delta_layer


def cal_tilde_delta_with_regularization(H_h, P, refusal_vectors, layers, lambda_reg=1e-5, device="cuda:0"):
    """
    Calculate the steering vector using regularized pseudoinverse method for multiple layers.
    
    Parameters:
    - H_h: [batch_size, num_layers, hidden_dim] - Feature matrix of harmful behaviors
    - P: [num_layers, hidden_dim, hidden_dim] - Projection matrices for each layer
    - refusal_vectors: [num_layers, hidden_dim] - Target refusal vectors
    - layers: List of layer indices to process
    - lambda_reg: Regularization parameter
    - device: Device to perform computation on
    
    Returns:
    - tilde_delta: Stack of regularized solution vectors for each layer
    """
    H_h = H_h.to(device)
    P = P.to(device)
    refusal_vectors = refusal_vectors.to(device)

    tilde_delta_list = []
    for layer in layers:
        print(f"layer {layer}:", end="\t")
        tilde_delta_layer = cal_tilde_delta_with_regularization_l(
            H_h[:, layer, :], 
            P[layer], 
            refusal_vectors[layer], 
            lambda_reg, 
            device=device)
        tilde_delta_list.append(tilde_delta_layer)
    tilde_delta = torch.stack(tilde_delta_list, dim=0)
    return tilde_delta

def cal_steering_matrix_l(P_layer, tilde_delta_layer, device="cuda:0"):
    """
    Calculate the steering matrix for a single layer.
    
    Parameters:
    - P_layer: [hidden_dim, hidden_dim] - Projection matrix
    - tilde_delta_layer: [hidden_dim] - Solution vector
    - device: Device to perform computation on
    
    Returns:
    - steering_matrix_layer: Steering matrix for the layer
    """
    P_layer = P_layer.to(device)
    tilde_delta_layer = tilde_delta_layer.to(device)
    steering_matrix_layer = P_layer @ tilde_delta_layer
    return steering_matrix_layer

def cal_steering_matrix(P, tilde_delta, layers, device="cuda:0"):
    """
    Calculate the steering matrix for multiple layers.
    
    Parameters:
    - P: [num_layers, hidden_dim, hidden_dim] - Projection matrices
    - tilde_delta: [num_layers, hidden_dim] - Solution vectors
    - layers: List of layer indices to process
    - device: Device to perform computation on
    
    Returns:
    - steering_matrix: Stack of steering matrices for each layer
    """
    P = P.to(device)
    tilde_delta = tilde_delta.to(device)
    
    steering_matrix_list = []
    for layer in layers:
        steering_matrix_layer = cal_steering_matrix_l(
            P[layer], tilde_delta[layer], device=device)
        steering_matrix_list.append(steering_matrix_layer)
    steering_matrix = torch.stack(steering_matrix_list, dim=0)
    return steering_matrix
from numpy import ndarray, minimum, zeros, empty, float64, ones, asarray, full, inf, array, int32, repeat, arange, tile
from numpy import max, abs
from scipy.sparse import eye, csr_matrix, vstack
from scipy.sparse.linalg import spsolve
from Const import Const
from utils import compute_DH_probability, compute_YV_probability, DH_space

def solution(C: Const, N_value_it: int = 10) -> tuple[ndarray, ndarray]:

    ## Settings
    value_funct_init = -50.0
    tol = 1e-6

    ## Precompute P and Q
    P_list = compute_transition_probability(C)
    P_stack = vstack(P_list)
    Q = asarray([-1.0, C.lam_weak - 1.0, C.lam_strong - 1.0], dtype=float64)

    ## Preallocated memory space / init
    state_size = P_list[0].shape[0]
    J_opt = full(state_size, value_funct_init, dtype=float64)
    J_prev = empty(state_size, dtype=float64)
    cand = empty(state_size, dtype=float64)
    u_opt = zeros(state_size, dtype=int32)   

    # Initiate Loop Variables
    iter = 0
    policy_evaluation_counter = 0
    L_inf_norm = inf

    ### Generalized policy iteration
    # start_time = time.time()
    while L_inf_norm > tol:
        # VALUE ITERATION
        J_prev, J_opt = J_opt, J_prev
        J_opt = P_list[0] @ J_prev + Q[0]
        for u in range(1,3):
            cand = P_list[u] @ J_prev + Q[u]
            minimum(J_opt, cand, out=J_opt)
        L_inf_norm = max(abs(J_opt - J_prev))
        iter += 1

        if iter % N_value_it != 0:
            continue
        
        policy_evaluation_counter += 1
        # POLICY ITERATION
        J_prev, J_opt = J_opt, J_prev
        u_opt[:] = 0
        J_opt = P_list[0] @ J_prev + Q[0]
        for u in range(1,3):
            cand = P_list[u] @ J_prev + Q[u]
            mask = cand < J_opt
            u_opt[mask] = u
            J_opt[mask] = cand[mask]
        # POLICY EVALUATION
        row_idx = u_opt * state_size + arange(state_size)
        P_mu = P_stack[row_idx, :]
        A = eye(state_size, format="csr") - P_mu
        J_opt = spsolve(A, Q[u_opt])
        

    # POLICY ITERATION
    J_prev, J_opt = J_opt, J_prev
    u_opt[:] = 0
    J_opt = P_list[0] @ J_prev + Q[0]
    for u in range(1,3):
        cand = P_list[u] @ J_prev + Q[u]
        mask = cand < J_opt
        u_opt[mask] = u
        J_opt[mask] = cand[mask]
    u_opt = array(C.input_space)[u_opt]
    
    return J_opt, u_opt

def compute_transition_probability(C:Const) -> array:
        """Computes the transition probability matrix P.

        Args:
                C (Const): The constants describing the problem instance.

        Returns:
                array: Transition probability matrix of shape (K,K,L), where:
                - K is the size of the state space;
                - L is the size of the input space.
                - P[i,j,l] corresponds to the probability of transitioning
                from the state i to the state j when input l is applied.
        """     
        
        ## Compute P
        # Compute P_DH (CSR sparse)
        P_DH, DH = compute_DH_probability(C)
        DH_size = P_DH.shape[1]
        # Compute P_YV (CSR sparse)
        P_YV_list = compute_YV_probability(C)
        YV_size = P_YV_list[0].shape[1]
        # Compute P_termination (dense)
        P_termination = compute_termination_probability(C, DH, YV_size, DH_size)
        
        ## Compute problem parameters
        input_size = 3
        total_size = DH_size * YV_size
        
        # Compute P
        P_list = []
        
        # Extract P_DH structure (control-independent)
        P_DH_rows, P_DH_cols = P_DH.nonzero()
        P_DH_data = P_DH.data
        n_DH_nonzero = len(P_DH_data)
        
        for u in range(input_size):
                
                P_YV = P_YV_list[u]
                
                # Extract P_YV structure (control dependent)
                P_YV_rows, P_YV_cols = P_YV.nonzero()
                P_YV_data = P_YV.data
                n_YV_nonzero = len(P_YV_data)
                
                # Preallocate arrays (note: each (i_DH, j_DH) pair combines with each (i_YV, j_YV) pair)
                preallocated_len = n_DH_nonzero * n_YV_nonzero
                rows = empty(preallocated_len, dtype=int32)
                cols = empty(preallocated_len, dtype=int32)
                data = empty(preallocated_len, dtype=float64)
                
                # Construct paring
                idx_DH = arange(n_DH_nonzero)
                idx_YV = arange(n_YV_nonzero)
                mesh_DH = repeat(idx_DH, n_YV_nonzero)
                mesh_YV = tile(idx_YV, n_DH_nonzero)
                
                # Indices computation based on on mesh
                rows[:] = P_YV_rows[mesh_YV] * DH_size + P_DH_rows[mesh_DH]
                cols[:] = P_YV_cols[mesh_YV] * DH_size + P_DH_cols[mesh_DH]
                
                # P value computation according to P_termination(i_YV, i_DH) * P_DH * P_YV
                data[:] = (P_termination[P_YV_rows[mesh_YV], P_DH_rows[mesh_DH]] * 
                        P_DH_data[mesh_DH] * 
                        P_YV_data[mesh_YV])
                
                # Append current-control CSR matrix
                P_list.append(csr_matrix((data, (rows, cols)), shape=(total_size, total_size)))
        return P_list

def compute_termination_probability(C : Const, DH : DH_space = None, YV_size : int = None, DH_size : int = None):
        
        if DH is None:
                DH = DH_space(C)
        if YV_size is None:
                YV_size = C.Y * (2 * C.V_max + 1)
        # determine whether to compute the state space or not
        if hasattr(C, "_state_space"):
                DH.build_DH_space_from_Const(C)
                state_space = asarray(C.state_space[:DH.N])[:, 2:]
        else:
                DH.build_DH_space_indipendently(C)
                state_space = array(DH.states)[:, 2:]
        if DH_size is None:
                DH_size = DH.N
                
        # precompute
        d0 = state_space[:, 0]
        h0 = state_space[:, C.M]
        mask = (d0 == 0)
        V_size = 2 * C.V_max + 1
        y = arange(YV_size) // V_size
        P_termination = ones((YV_size, DH_size), dtype=float64)
        half = (C.G - 1) // 2
        # compute P_term
        P_termination[:, mask] = ((y[:, None] <= h0[None, mask] + half) &
                                  (y[:, None] >= h0[None, mask] - half)).astype(float64)
        return P_termination

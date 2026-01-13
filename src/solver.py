from numpy import ndarray, minimum, zeros, empty, float64, asarray, full, inf, array, int32, arange
from numpy import max, abs
from scipy.sparse import eye, vstack
from scipy.sparse.linalg import spsolve
from Const import Const
from kernel import transition_kernel

def bellman_solver(C: Const, N_value_it: int = 10) -> tuple[ndarray, ndarray]:

    ## Settings
    v_init =    -50.0
    tol =       1e-6

    ## Precompute P and Q
    P_list =    transition_kernel(C) 
    P_stack =   vstack(P_list)
    Q =         asarray([-1.0, C.lam_weak - 1.0, C.lam_strong - 1.0], dtype=float64)

    ## Preallocated memory space / init
    state_sz =  P_list[0].shape[0]
    J_opt =     full(state_sz, v_init, dtype=float64)
    J_prev =    empty(state_sz, dtype=float64)
    cand =      empty(state_sz, dtype=float64)
    u_opt =     zeros(state_sz, dtype=int32)   

    ## Initiate Loop Variables
    iter =      0
    err_norm =  inf

    while err_norm > tol:
        ## VALUE ITERATION
        J_prev, J_opt = J_opt, J_prev
        J_opt = P_list[0] @ J_prev + Q[0]
        for u in range(1,3):
            cand = P_list[u] @ J_prev + Q[u]
            minimum(J_opt, cand, out=J_opt)
        err_norm = max(abs(J_opt - J_prev))
        iter += 1

        if iter % N_value_it != 0:
            continue
        
        ## POLICY ITERATION
        ### GREEDY POLICY
        J_prev, J_opt = J_opt, J_prev
        u_opt[:] = 0
        J_opt = P_list[0] @ J_prev + Q[0]
        for u in range(1,3):
            cand = P_list[u] @ J_prev + Q[u]
            mask = cand < J_opt
            u_opt[mask] = u
            J_opt[mask] = cand[mask]
            
        ### POLICY EVALUATION
        row_idx = u_opt * state_sz + arange(state_sz)
        P_mu = P_stack[row_idx, :]
        A = eye(state_sz, format="csr") - P_mu
        J_opt = spsolve(A, Q[u_opt])
        
    ## GREEDY POLICY
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
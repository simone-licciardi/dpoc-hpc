from numpy import ndarray, empty, float64, ones, asarray, full, array, int32, repeat, arange, tile
from scipy.sparse import csr_matrix
from Const import Const
from utils import DH_space
from Const import Const
from numpy import hstack, arange, tile, ones, float64, full, clip, ndarray, empty
from scipy.sparse import csr_matrix

def transition_kernel(C:Const) -> array:
        
        # Compute P
        ## Compute P_DH (CSR sparse)
        P_DH, DH =      environment_kernel(C)
        DH_size =       P_DH.shape[1]
        
        ## Compute P_YV (CSR sparse)
        P_YV_list =     kinematics_kernel(C)
        YV_size =       P_YV_list[0].shape[1]
        
        # Compute P_end (dense)
        P_end =         termination_kernel(C, DH, YV_size, DH_size)
        
        ## Compute problem parameters
        input_size =    3
        total_size =    DH_size * YV_size
        P_list =        []
        
        # Extract P_DH structure (control-independent)
        P_DH_rows, P_DH_cols = P_DH.nonzero()
        P_DH_data =     P_DH.data
        n_DH_nonzero =  len(P_DH_data)
        
        for u in range(input_size):
                
                P_YV =          P_YV_list[u]
                
                # Extract P_YV structure (control dependent)
                P_YV_rows, P_YV_cols = P_YV.nonzero()
                P_YV_data =     P_YV.data
                n_YV_nonzero =  len(P_YV_data)
                
                # Preallocate arrays (note: each (i_DH, j_DH) pair combines with each (i_YV, j_YV) pair)
                preallocated_len = n_DH_nonzero * n_YV_nonzero
                rows =          empty(preallocated_len, dtype=int32)
                cols =          empty(preallocated_len, dtype=int32)
                data =          empty(preallocated_len, dtype=float64)
                
                # Construct paring
                idx_DH =        arange(n_DH_nonzero)
                idx_YV =        arange(n_YV_nonzero)
                mesh_DH =       repeat(idx_DH, n_YV_nonzero)
                mesh_YV =       tile(idx_YV, n_DH_nonzero)
                
                # Indices computation based on on mesh
                rows[:] =       P_YV_rows[mesh_YV] * DH_size + P_DH_rows[mesh_DH]
                cols[:] =       P_YV_cols[mesh_YV] * DH_size + P_DH_cols[mesh_DH]
                
                # P value computation according to P_end(i_YV, i_DH) * P_DH * P_YV
                data[:] =       (P_end[P_YV_rows[mesh_YV], P_DH_rows[mesh_DH]] * 
                                P_DH_data[mesh_DH] * 
                                P_YV_data[mesh_YV])
                
                # Append current-control CSR matrix
                P_list.append(csr_matrix((data, (rows, cols)), shape=(total_size, total_size)))
        return P_list

def termination_kernel(C : Const, DH : DH_space = None, YV_size : int = None, DH_size : int = None):
        
        if DH is None:
                DH =            DH_space(C)
                
        if YV_size is None:
                YV_size =       C.Y * (2 * C.V_max + 1)
                
        # determine whether to compute the state space or not
        if hasattr(C, "_state_space"):
                DH.build_DH_space_from_Const(C)
                state_space =   asarray(C.state_space[:DH.N])[:, 2:]
        else:
                DH.build_DH_space_indipendently(C)
                state_space =   array(DH.states)[:, 2:]
        if DH_size is None:
                DH_size =       DH.N
                
        # precomputations
        d0 =            state_space[:, 0]
        h0 =            state_space[:, C.M]
        mask =          (d0 == 0)
        V_size =        2 * C.V_max + 1
        y =             arange(YV_size) // V_size
        P_end =         ones((YV_size, DH_size), dtype=float64)
        half =          (C.G - 1) // 2
        
        # compute P_term
        P_end[:, mask] = ((y[:, None] <= h0[None, mask] + half) &
                          (y[:, None] >= h0[None, mask] - half)).astype(float64)
        
        return P_end

def environment_kernel(C : Const):
    # get model parameters
    DH =        DH_space(C)
    X =         C.X
    S_h =       C.S_h
    H =         len(S_h)
    D_min =     C.D_min
    M =         C.M
    pmf_hole =  1/H
    
    def pmf_spawn(s, X, D_min):
        if s < D_min:
                return 0
        if s >= X:
                return 1
        return (s-D_min+1)/(X-D_min)

    
    # get state space & its indexing
    if hasattr(C, '_state_space'):
        DH.build_DH_space_from_Const(C)
        state_space = C.state_space[:DH.N]
    else:
        DH.build_DH_space_indipendently(C)
        state_space = DH.states

    # pre allocation of result arrays
    P =         empty((H+1)*DH.N, dtype=float64)
    row =       empty((H+1)*DH.N, dtype=int32)
    col =       empty((H+1)*DH.N, dtype=int32)

    # pre allocation of loop arrays
    next =      empty((2*M, 1 + H), dtype=int32)
    next_ids =  empty(H + 1, dtype=int32)
    d_hat =     empty(M, dtype=int32)
    h_hat =     empty(M, dtype=int32)

    m = 0
    for k in range(DH.N):
        curr_state =            (state_space[k])[2:]
        if curr_state[0] != 0:
            d_hat[0] =          curr_state[0] - 1
            d_hat[1:] =         curr_state[1:M]
            h_hat[:] =          curr_state[M:]
        else:
            d_hat[0] =          curr_state[1] - 1
            d_hat[1:-1] =       curr_state[2:M]
            d_hat[-1] = 0
            h_hat[:-1] =        curr_state[M+1:]
            h_hat[-1] =         S_h[0]

        s =             (X - 1) - int(d_hat.sum())
        pr_spawn =      pmf_spawn(s, X, D_min)

        # compute the (2M)x(1+H) entry of possible next states, and their probability
        # compute m_min
        m_min = 1
        while m_min < M - 1 and d_hat[m_min] != 0:
            m_min += 1

        # compute all H+1 possible new states, 1 per spawn possibility
        next[:M, :] =           d_hat[:, None]
        next[m_min, 1:] =       s
        next[M:, :] =           h_hat[:, None]
        next[M+m_min, 1:] =     S_h
        
        # print(curr_state,next,pr_spawn)
        
        # save the next states by computing their indices and the associated probability
        if pr_spawn == 0:
            col[m] = DH.find_index(next[:, 0])
            row[m] = k
            P[m] = 1
            m += 1
        elif pr_spawn == 1:
            for i in range(H):
                next_ids[i] =   DH.find_index(next[:, i+1])
            col[m:m+H] =        next_ids[:-1]
            row[m:m+H] =        k
            P[m:m+H] =          pmf_hole
            m +=                H
        else:
            for i in range(H+1):
                next_ids[i] =   DH.find_index(next[:, i])
            col[m:m+H+1] =      next_ids
            row[m:m+H+1] =      k
            P[m] =              1-pr_spawn
            P[m+1:m+H+1] =      pr_spawn*pmf_hole
            m +=                H+1

    # return
    return csr_matrix((P[:m], (row[:m], col[:m])), shape=(DH.N, DH.N)), DH

def kinematics_kernel(C: Const):

    def y_dynamics(Y: int, V: int):
        # column vector of all admissible Y states
        Y_column =  arange(0, Y, dtype=int).reshape((-1, 1))
        V_row =     arange(-V, V+1, dtype=int)  # row vector of all admissible V states
        return clip(Y_column + V_row, 0, Y-1)  # broadcasting and clipping


    def v_dynamics(V: int, U: ndarray):
        V_column =  arange(-V, V+1, dtype=int).reshape((-1, 1))
        U_row = U
        return clip(V_column + U_row, -V, V)


    def YV_find_index(y, v, V_dim : int, V_max : int): # vectorized
                return y * V_dim + v + V_max
    

    # extract model parameters
    Y =         C.Y
    V =         C.V_max
    S_u =       [C.U_no_flap, 
                 C.U_weak, 
                 C.U_strong]
    V_dev =     C.V_dev

    # extract size of each state component
    Y_size =            Y
    V_size =            2 * V + 1
    total_states =      Y_size * V_size
    dev_on_V_size =     2 * V_dev + 1

    # we precompute the support of state update U as a nparray,
    # both under deterministic effect (no_flap, weak),
    # and under process noise (strong)
    U = hstack([S_u[0],  # deterministic control (no_flap)
                S_u[1],  # deterministic control (weak)
                # support of the control (strong), with process noise
                arange(-V_dev, V_dev+1) + S_u[2]], dtype=int)
    U = U - C.g  # this includes the effect of gravity, as part of the state update

    # we compute these handler functions
    Y_new =     y_dynamics(Y, V)  # (Y_new)_ij = y_dynamics(y_i,v_j)
    V_new =     v_dynamics(V, U)  # (V_new)_jk = v_dynamics(v_j, u_k)

    # we compute the associated
    # this is (y_0,v_0), (y_0,v_1), ..., (y_1,v_0), ..., (y_Y_size, v_V_size)
    Y_new_flat =        Y_new.reshape(-1)
    # this is (v_0,u_0), (v_1,u_0), ..., (v_V_size,u_0)
    V_no_flap_flat =    (V_new[:, 0]).reshape(-1)
    # this is (v_0,u_1), (v_1,u_1), ..., (v_V_size,u_1)
    V_weak_flat =       (V_new[:, 1]).reshape(-1)
    # this is (v_0,u_2), ..., (v_V_size,u_2), (v_0,u_3), ..., (v_V_size,u_3), ...
    V_strong_flap =     (V_new[:, 2:]).reshape(-1)

    # we compute the indices associated to the support of X_next provided the control choice and process disturbance
    I_new_no_flap =     YV_find_index(Y_new_flat,
                                        tile(V_no_flap_flat, Y_size),
                                        V_size, V)
    I_new_weak =        YV_find_index(Y_new_flat,
                                        tile(V_weak_flat, Y_size),
                                        V_size, V)
    I_new_strong =      YV_find_index(tile(Y_new_flat, (dev_on_V_size, 1)).reshape(-1, order="F"),  # this is (y_0,v_0) * dev_on_size,  (y_0,v_1) * dev_on_size, ...
                                        tile(V_strong_flap, Y_size),
                                        V_size, V)

    # we compute the row, col and P matrices in such a way that the size of each array is (total_states)*(dev_on_states + 2),
    # Namely, row describes the 3*total_states rows of the P matrix (each block corresponds to a control)
    #       i % total_states input state
    #       i // total_states control choice.
    # col describes the indices between 0 and total_states of each state in the support for x_next
    # P describes determinism for the first two inputs, and uniform stochasticity for the rest

    row_no_flap =       arange(total_states, dtype=int)
    col_no_flap =       I_new_no_flap
    P_no_flap =         ones(total_states, dtype=float64)

    row_weak =          arange(total_states, dtype=int)
    col_weak =          I_new_weak
    P_weak =            ones(total_states, dtype=float64)

    row_strong =        tile(arange(total_states, dtype=int),
                             (dev_on_V_size, 1)).reshape(-1, order="F")
    col_strong =        I_new_strong
    P_strong =          full(total_states*dev_on_V_size, 1 /
                             (dev_on_V_size), dtype=float64)

    # we choose a csr matrix as we will need to take matmult later in the code
    P_shape =   (total_states, total_states)
    return [csr_matrix((P_no_flap,  (row_no_flap,   col_no_flap)),  shape=P_shape),
            csr_matrix((P_weak,     (row_weak,      col_weak)),     shape=P_shape),
            csr_matrix((P_strong,   (row_strong,    col_strong)),   shape=P_shape)]
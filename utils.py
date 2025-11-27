"""utils.py

Python script containg utility functions. Modify if needed,
but be careful as these functions are used, e.g., in simulation.py.

Dynamic Programming and Optimal Control
Fall 2025
Programming Exercise

Contact: Antonio Terpin aterpin@ethz.ch

Authors: Marius Baumann, Antonio Terpin

--
ETH Zurich
Institute for Dynamic Systems and Control
--
"""

from Const import Const
from numpy import hstack, arange, tile, ones, float64, full, clip, ndarray, empty, int64
from scipy.sparse import csr_matrix
from itertools import product

def spawn_probability(C: Const, s: int) -> float:
    """Distance-dependent spawn probability p_spawn(s).
    
    Args:
        C (Const): The constants describing the problem instance.
        s (int): Free distance, as defined in the assignment.

    Returns:
        float: The spawn probability p_spawn(s).
    """
    return max(min((s - C.D_min + 1) / (C.X - C.D_min), 1.0), 0.0)

def is_in_gap(C: Const, y: int, h1: int) -> bool:
    """Returns true if bird in gap.
    
    Args:
        C (Const): The constants describing the problem instance.
        y (int): Vertical position of the bird.
        h1 (int): Center of the gap of the first obstacle.

    Returns:
        bool: True if bird is in the gap, False otherwise.
    """
    half = (C.G - 1) // 2
    return abs(y - h1) <= half

def is_passing(C: Const, y: int, d1: int, h1: int) -> bool:
    """Return true if bird is currently passing the gap without colliding.
    
    Args:
        C (Const): The constants describing the problem instance.
        y (int): Vertical position of the bird.
        d1 (int): Distance to the first obstacle.
        h1 (int): Center of the gap of the first obstacle.

    Returns:
        bool: True if bird is passing the gap, False otherwise.
    """
    return (d1 == 0) and is_in_gap(C, y, h1)

def is_collision(C: Const, y: int, d1: int, h1: int) -> bool:
    """Return true if bird is colliding with obstacle.
    
    Args:
        C (Const): The constants describing the problem instance.
        y (int): Vertical position of the bird.
        d1 (int): Distance to the first obstacle.
        h1 (int): Center of the gap of the first obstacle.

    Returns:
        bool: True if bird is colliding with obstacle, False otherwise.
    """
    return (d1 == 0) and not is_in_gap(C, y, h1)

# YV_space_handler.py

def YV_find_index(y, v, V_dim : int, V_max : int): # vectorized
        return y * V_dim + v + V_max
    
# P_YV_builder.py
# depends on YV_space_handler.py and Const.py

def compute_YV_probability(C: Const):

    # extract model parameters
    Y = C.Y
    V = C.V_max
    S_u = [C.U_no_flap, C.U_weak, C.U_strong]
    V_dev = C.V_dev

    # extract size of each state component
    Y_size = Y
    V_size = 2 * V + 1
    total_states = Y_size * V_size
    dev_on_V_size = 2 * V_dev + 1

    # we precompute the support of state update U as a nparray,
    # both under deterministic effect (no_flap, weak),
    # and under process noise (strong)
    U = hstack([S_u[0],  # deterministic control (no_flap)
                S_u[1],  # deterministic control (weak)
                # support of the control (strong), with process noise
                arange(-V_dev, V_dev+1) + S_u[2]], dtype=int)
    U = U - C.g  # this includes the effect of gravity, as part of the state update

    # we compute these handler functions
    Y_new = new_y_due_to(Y, V)  # (Y_new)_ij = new_y_due_to(y_i,v_j)
    V_new = new_v_due_to(V, U)  # (V_new)_jk = new_v_due_to(v_j, u_k)

    # we compute the associated
    # this is (y_0,v_0), (y_0,v_1), ..., (y_1,v_0), ..., (y_Y_size, v_V_size)
    Y_new_flat = Y_new.reshape(-1)
    # this is (v_0,u_0), (v_1,u_0), ..., (v_V_size,u_0)
    V_no_flap_flat = (V_new[:, 0]).reshape(-1)
    # this is (v_0,u_1), (v_1,u_1), ..., (v_V_size,u_1)
    V_weak_flat = (V_new[:, 1]).reshape(-1)
    # this is (v_0,u_2), ..., (v_V_size,u_2), (v_0,u_3), ..., (v_V_size,u_3), ...
    V_strong_flap = (V_new[:, 2:]).reshape(-1)

    # we compute the indices associated to the support of X_next provided the control choice and process disturbance
    I_new_no_flap = YV_find_index(Y_new_flat,
                                  tile(V_no_flap_flat, Y_size),
                                  V_size, V)
    I_new_weak = YV_find_index(Y_new_flat,
                               tile(V_weak_flat, Y_size),
                               V_size, V)
    I_new_strong = YV_find_index(tile(Y_new_flat, (dev_on_V_size, 1)).reshape(-1, order="F"),  # this is (y_0,v_0) * dev_on_size,  (y_0,v_1) * dev_on_size, ...
                                 tile(V_strong_flap, Y_size),
                                 V_size, V)

    # we compute the row, col and P matrices in such a way that the size of each array is (total_states)*(dev_on_states + 2),
    # Namely, row describes the 3*total_states rows of the P matrix (each block corresponds to a control)
    #       i % total_states input state
    #       i // total_states control choice.
    # col describes the indices between 0 and total_states of each state in the support for x_next
    # P describes determinism for the first two inputs, and uniform stochasticity for the rest

    row_no_flap = arange(total_states, dtype=int)
    col_no_flap = I_new_no_flap
    P_no_flap = ones(total_states, dtype=float64)

    row_weak = arange(total_states, dtype=int)
    col_weak = I_new_weak
    P_weak = ones(total_states, dtype=float64)

    row_strong = tile(arange(total_states, dtype=int),
                      (dev_on_V_size, 1)).reshape(-1, order="F")
    col_strong = I_new_strong
    P_strong = full(total_states*dev_on_V_size, 1 /
                    (dev_on_V_size), dtype=float64)

    # we choose a csr matrix as we will need to take matmult later in the code
    P_shape = (total_states, total_states)
    return [csr_matrix((P_no_flap, (row_no_flap,   col_no_flap)),  shape=P_shape),
            csr_matrix((P_weak,     (row_weak,      col_weak)),
                       shape=P_shape),
            csr_matrix((P_strong,   (row_strong,    col_strong)),   shape=P_shape)]


def new_y_due_to(Y: int, V: int):
    # column vector of all admissible Y states
    Y_column = arange(0, Y, dtype=int).reshape((-1, 1))
    V_row = arange(-V, V+1, dtype=int)  # row vector of all admissible V states
    return clip(Y_column + V_row, 0, Y-1)  # broadcasting and clipping


def new_v_due_to(V: int, U: ndarray):
    V_column = arange(-V, V+1, dtype=int).reshape((-1, 1))
    U_row = U
    return clip(V_column + U_row, -V, V)

# DH_space_handler.py
# depends on Const.py

class DH_space:
    __slots__ = ("D_min","X","M","S_h","S_d1","S_d","V_max","states","index_of","N","has_states")

    def __init__(self, C: Const):
        # parameters
        self.D_min = C.D_min
        self.X = C.X
        self.M = C.M
        self.S_h = C.S_h
        self.S_d1 = tuple(range(self.X))
        self.S_d  = (0,) + tuple(range(self.D_min, self.X))
        self.V_max = C.V_max
        self.has_states = False
        
    def build_DH_space_from_Const(self, C : Const):
        
        if self.has_states:
            return
        
        self.index_of = {}
    
        tot = C.K
        Y_size = C.Y
        V_size = 2 * self.V_max + 1
        total_states = Y_size * V_size
    
        self.N = tot // total_states
        self.index_of = {st[2:]: i for i, st in enumerate(C.state_space[:self.N])}
        
        self.has_states = True
        
    def build_DH_space_indipendently(self, C: Const):
        
        if self.has_states:
            return
        
        # containers to fill
        self.states = []
        self.index_of = {}

        Sd1 = C.S_d1                # d1 ∈ {0, ..., X-1}
        Sd  = C.S_d                 # d_i ∈ {0} ∪ {D_min, ..., X-1}  for i ≥ 2
        Sh  = C.S_h                 # admissible gap centers
        M   = C.M
        X   = C.X
        hd  = C.S_h[0]              # default gap height when d_i == 0 for i ≥ 2
        
        self.has_states = True

        # helper: check zero-tail constraint on (d2,...,dM)
        def valid_tail(d_tail):
            seen_zero = False
            for di in d_tail:
                if seen_zero and di != 0:
                    return False
                if di == 0:
                    seen_zero = True
            return True

        # Enumerate in the required lexicographic order: d1, d2..dM, then h1..hM
        for d1 in Sd1:
            for d_tail in product(Sd, repeat=M-1):
                # enforce: if d1 == 0 then d2 > 0
                if d1 == 0 and d_tail[0] == 0:
                    continue
                # enforce zero-tail structure on d2..dM
                if not valid_tail(d_tail):
                    continue
                D = (d1,) + d_tail
                # enforce total span fits on screen
                if sum(D) > X - 1:
                    continue

                # Build H iterables: h1 ∈ Sh; for i≥2, if d_i==0 then h_i=hd else h_i ∈ Sh
                H_iters = [Sh]
                for i in range(1, M):            # indices 1..M-1 correspond to d2..dM
                    if d_tail[i - 1] == 0:
                        H_iters.append([hd])
                    else:
                        H_iters.append(Sh)

                for H in product(*H_iters):
                    st = (0, -C.V_max) + D + H                    # (d1,...,dM,h1,...,hM)
                    self.states.append(st)

        # forward pass to index the state_space
        self.index_of = {st[2:]: i for i, st in enumerate(self.states)}
        # optional: store size if your class uses it elsewhere
        try:
            self.N = len(self.states)
        except Exception:
            pass

    def find_index(self, DH_state):
        return self.index_of[tuple(DH_state)]
    

# P_DH_builder.py 
# depends on DH_space_handler.py, Const.py

def compute_DH_probability(C : Const):
    # get model parameters
    DH = DH_space(C)
    X = C.X
    S_h = C.S_h
    H = len(S_h)
    D_min = C.D_min
    M = C.M
    pr_obstacle = 1/H
    
    # get state space & its indexing
    if hasattr(C, '_state_space'):
        DH.build_DH_space_from_Const(C)
        state_space = C.state_space[:DH.N]
    else:
        DH.build_DH_space_indipendently(C)
        state_space = DH.states

    # pre allocation of result arrays
    P = empty((H+1)*DH.N, dtype=float64)
    row = empty((H+1)*DH.N, dtype=int64)
    col = empty((H+1)*DH.N, dtype=int64)

    # pre allocation of loop arrays
    next_states = empty((2*M, 1 + H), dtype=int64)
    next_states_indices = empty(H + 1, dtype=int64)
    d_hat = empty(M, dtype=int64)
    h_hat = empty(M, dtype=int64)

    m = 0
    for k in range(DH.N):
        curr_state = (state_space[k])[2:]
        if curr_state[0] != 0:
            d_hat[0] = curr_state[0] - 1
            d_hat[1:] = curr_state[1:M]
            h_hat[:] = curr_state[M:]
        else:
            d_hat[0] = curr_state[1] - 1
            d_hat[1:-1] = curr_state[2:M]
            d_hat[-1] = 0
            h_hat[:-1] = curr_state[M+1:]
            h_hat[-1] = S_h[0]

        s = (X - 1) - int(d_hat.sum())
        pr_spawn = pmf_spawn(s, X, D_min)

        # compute the (2M)x(1+H) entry of possible next states, and their probability
        # compute m_min
        m_min = 1
        while m_min < M - 1 and d_hat[m_min] != 0:
            m_min += 1

        # compute all H+1 possible new states, 1 per spawn possibility
        next_states[:M, :] = d_hat[:, None]
        next_states[m_min, 1:] = s
        next_states[M:, :] = h_hat[:, None]
        next_states[M+m_min, 1:] = S_h
        
        # print(curr_state,next_states,pr_spawn)
        
        # save the next states by computing their indices and the associated probability
        if pr_spawn == 0:
            col[m] = DH.find_index(next_states[:, 0])
            row[m] = k
            P[m] = 1
            m += 1
        elif pr_spawn == 1:
            for i in range(H):
                next_states_indices[i] = DH.find_index(next_states[:, i+1])
            col[m:m+H] = next_states_indices[:-1]
            row[m:m+H] = k
            P[m:m+H] = pr_obstacle
            m += H
        else:
            for i in range(H+1):
                next_states_indices[i] = DH.find_index(next_states[:, i])
            col[m:m+H+1] = next_states_indices
            row[m:m+H+1] = k
            P[m] = 1-pr_spawn
            P[m+1:m+H+1] = pr_spawn*pr_obstacle
            m += H+1

    # return
    return csr_matrix((P[:m], (row[:m], col[:m])), shape=(DH.N, DH.N)), DH


def pmf_spawn(s, X, D_min):
    if s < D_min:
        return 0
    if s >= X:
        return 1
    return (s-D_min+1)/(X-D_min)

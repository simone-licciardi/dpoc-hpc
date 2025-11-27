"""ComputeTransitionProbabilities.py

Template to compute the transition probability matrix.

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

import numpy as np
from scipy.sparse import csr_matrix
from Const import Const
# from my_algo.P_YV_builder import compute_YV_probability
# from my_algo.P_DH_builder import compute_DH_probability
from utils import compute_DH_probability, compute_YV_probability

def compute_transition_probabilities(C:Const) -> np.array:
        """Computes the transition probability matrix P.

        Args:
                C (Const): The constants describing the problem instance.

        Returns:
                np.array: Transition probability matrix of shape (K,K,L), where:
                - K is the size of the state space;
                - L is the size of the input space.
                - P[i,j,l] corresponds to the probability of transitioning
                from the state i to the state j when input l is applied.
        """     
        P_DH, DH = compute_DH_probability(C)
        P_YV_list = compute_YV_probability(C)
        YV_size = P_YV_list[0].shape[1]
        DH_size = P_DH.shape[1]
        V_size = 2 * C.V_max + 1

        # Get P_term
        if hasattr(C, "_state_space"):
                DH.build_DH_space_from_Const(C)
                state_space = np.array(C.state_space[:DH.N])[:, 2:]
        else:
                DH.build_DH_space_indipendently(C)
                state_space = np.array(DH.states)[:, 2:]

        d0 = state_space[:, 0]
        h0 = state_space[:, C.M]
        mask = (d0 == 0)

        # Precompute termination probability matrix
        y = np.arange(YV_size) // V_size
        P_termination = np.ones((YV_size, DH_size), dtype=np.float64)
        half = (C.G - 1) // 2
        P_termination[:, mask] = ((y[:, None] <= h0[None, mask] + half) &
                              (y[:, None] >= h0[None, mask] - half)).astype(np.float64)

        n_actions = len(P_YV_list)
        total_size = DH_size * YV_size
        
        # Convert P_DH to CSR if not already
        if not isinstance(P_DH, csr_matrix):
                P_DH = csr_matrix(P_DH)
        
        P_list = []
        
        # Pre-extract P_DH structure once
        P_DH_rows, P_DH_cols = P_DH.nonzero()
        P_DH_data = P_DH.data
        n_DH_nonzero = len(P_DH_data)
        
        for u in range(n_actions):
                # Convert P_YV to CSR if needed
                P_YV = P_YV_list[u]
                
                # Extract P_YV structure
                P_YV_rows, P_YV_cols = P_YV.nonzero()
                P_YV_data = P_YV.data
                n_YV_nonzero = len(P_YV_data)
                
                # Preallocate arrays for the Kronecker-like product
                # Each (i_DH, j_DH) pair combines with each (i_YV, j_YV) pair
                n_total = n_DH_nonzero * n_YV_nonzero
                
                rows = np.empty(n_total, dtype=np.int32)
                cols = np.empty(n_total, dtype=np.int32)
                data = np.empty(n_total, dtype=np.float64)
                
                # Vectorized construction using broadcasting
                # Create index arrays
                idx_DH = np.arange(n_DH_nonzero)
                idx_YV = np.arange(n_YV_nonzero)
                
                # Use meshgrid for efficient pairing
                mesh_DH, mesh_YV = np.meshgrid(idx_DH, idx_YV, indexing='ij')
                mesh_DH = mesh_DH.ravel()
                mesh_YV = mesh_YV.ravel()
                
                # Compute row indices: i_YV * DH_size + i_DH
                rows[:] = P_YV_rows[mesh_YV] * DH_size + P_DH_rows[mesh_DH]
                
                # Compute column indices: j_YV * DH_size + j_DH
                cols[:] = P_YV_cols[mesh_YV] * DH_size + P_DH_cols[mesh_DH]
                
                # Compute data values: P_termination(i_YV, i_DH) * P_DH * P_YV
                # Use P_termination directly (NOT transposed)
                data[:] = (P_termination[P_YV_rows[mesh_YV], P_DH_rows[mesh_DH]] * 
                        P_DH_data[mesh_DH] * 
                        P_YV_data[mesh_YV])
                
                # Create CSR matrix
                P_list.append(csr_matrix((data, (rows, cols)), shape=(total_size, total_size)))
                
        P_list_dense = []
        for P in P_list:
                P_list_dense.append(P.toarray())
                
        return np.stack(P_list_dense, axis=-1)
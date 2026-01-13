from Const import Const
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

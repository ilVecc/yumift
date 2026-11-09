from abc import ABCMeta, abstractmethod
from typing import List, Tuple
from typing_extensions import override

import numpy as np
from quadprog import solve_qp


class HQPTaskError(RuntimeError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class Task(object, metaclass=ABCMeta):
    """ A generic task for the HQP solver.
        Directly subclassing this class is discouraged, please subclass one of 
        `LowerTask`, `EqualTask` or `UpperTask`, which come with all of the 
        required inner logic implemented.
        
        Based on https://github.com/ritalaezza/sot-myo/blob/akos_re/src/Task.py
    """
    
    def __init__(self, n: int, m: int, slack_ratio: float = 1e4):
        """ Prepares all the internal caches for the subclasses to use.
            :param n: degrees of freedom of the solution (i.e. number of free variables solving the task)
            :param m: degrees of freedom of the task (i.e. number of constraint equations defining the task)
            :param slack_ratio: ratio between the solution cost and the slack variables cost (i.e. how important slack is, higher values mean less importance)
        """
        # NON-SLACKED PROBLEM
        #   constr_mat * x = constr_vec
        # where 
        #   constr_mat is m x n
        #            x is n x 1
        #   constr_vec is m x 1
        #
        # SLACKED PROBLEM  (add m slack variables)
        #   slack_contrs_mat * y = constr_vec
        # where 
        #   slack_contrs_mat is m x (n + m)
        #                  y is (n + m) x 1
        #         constr_vec is m x 1
        self.n : int = n
        self.m : int = m
        self.slack_ratio: float = slack_ratio
        self.constr_mat = np.zeros((self.m,self.n))
        self.constr_vec = np.zeros((self.m,))
        # handy slack vector to store previously calculated slack for this task
        self.slack_vec = np.zeros((self.m,))
        # cache for the QP cost (i.e. 1/2 x^T P x + q^T x )
        self._cost_mat = np.eye(self.n + self.m)
        self._cost_mat[-self.m:, -self.m:] *= self.slack_ratio
        self._cost_vec = np.zeros((self.n + self.m,))
        # cache for the slack matrices
        self._with_slack_locked_mat : np.ndarray = None  # (m, n+?)         # cache matrix for A or G in .with_slack_locked()
        self._with_slack_constr_mat = np.zeros((self.m, self.n + self.m))   # cache matrix for A or G in .with_slack_ready()
        self._with_slack_zeros_mat = np.zeros((1, self.n + self.m))     # either for A or G
        self._with_slack_zeros_vec = np.zeros((1,))                     # either for b or h

    def preallocate_locked_slack_matrix(self, m_alloc):
        self._with_slack_locked_mat = np.zeros((self.m, self.n + m_alloc))

    @abstractmethod
    def with_slack_ready(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Returns `constr_mat` and `constr_vec` with `m` added slack variables, 
            i.e. one for each row of the task. Solving this linear problem returns
            its solution and the slack vector associated to it, which defines the 
            "allowed workspace" for the next tasks.
        """
        # Based on constraint type, either `A` and `b` or `G` and `h` must be 
        # set to a zero vector and a unit zero, respectively (trivial constraint), 
        # so to render adding previous tasks easier (also, the solver needs it in 
        # the case of 1 task).
        raise NotImplementedError("Return matrices A, b, G, h")
    
    @abstractmethod
    def with_slack_locked(self, m_other: int, w) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Returns `constr_mat` and `constr_vec` with slack variables locked 
            to values `w` and `m_other` added slack variables for the next task.
            Essentially, this function returns a relaxed version of the task in 
            which some slack is allowed to the solution, so to allow the next task
            to be solved in the "remaning space" (i.e. task null-space).
            
            ATTENTION: we expect `w` to contain the slack values previously obtained 
            by solving the problem of `with_slack_ready()`, and so to be the optimal slack 
            variables. When solving `with_slack_ready()`, `slack_vec` can be used to store 
            the optimal slack values.
            
            TODO could read directly from `self.slack_vec` instead of allowing `w`
        """
        # Based on constraint type, either `A` and `b` or `G` and `h` must be set 
        # to `None`, accordingly to `with_slack_ready()`
        raise NotImplementedError("Return matrices A, b, G, h")

    @abstractmethod
    def compute(self, *args) -> "Task":
        """ Calculate `constr_mat` and `constr_vec`.
        """
        return self


class LowerTask(Task):
    """ Task describing the problem `Gx <= h`
    """
    
    def __init__(self, n: int, m: int, slack_ratio: float = 1e4):
        super().__init__(n, m, slack_ratio)
        # cache
        self._with_slack_constr_mat[:, self.n:] = -np.eye(self.m)

    @override
    def with_slack_locked(self, m_other: int, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Returns `constr_mat` and `constr_vec` for previously solved tasks 
            including optimal slack variables, thus defining their nullspaces.
        """
        if self._with_slack_locked_mat is None:
            self.preallocate_locked_slack_matrix(m_other)
        
        self._with_slack_locked_mat[:, :self.n] = self.constr_mat
        G = self._with_slack_locked_mat[:, :self.n+m_other]  # slice only the required columns
        h = self.constr_vec + np.maximum(w, 0)
        return None, None, G, h

    @override
    def with_slack_ready(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Returns `constr_mat` and `constr_vec` with `m` added slack variables, 
            i.e. one for each row of the task.
        """
        self._with_slack_constr_mat[:, :self.n] = self.constr_mat
        return self._with_slack_zeros_mat, self._with_slack_zeros_vec, self._with_slack_constr_mat, self.constr_vec

class EqualTask(Task):
    """ Task describing the problem `Gx == h`
    """
    
    def __init__(self, n: int, m: int, slack_ratio: float = 1e4):
        super().__init__(n, m, slack_ratio)
        # cache
        self._with_slack_constr_mat[:, self.n:] = -np.eye(self.m)

    @override
    def with_slack_locked(self, m_other: int, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Returns `constr_mat` and `constr_vec` for previously solved tasks 
            including optimal slack variables, thus defining their nullspaces.
        """
        
        if self._with_slack_locked_mat is None:
            self.preallocate_locked_slack_matrix(m_other)
        
        self._with_slack_locked_mat[:, :self.n] = self.constr_mat
        A = self._with_slack_locked_mat[:, :self.n+m_other]  # slice only the required columns
        b = self.constr_vec + w
        return A, b, None, None

    @override
    def with_slack_ready(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Returns `constr_mat` and `constr_vec` with `m` added slack variables, 
            i.e. one for each row of the task.
        """
        self._with_slack_constr_mat[:, :self.n] = self.constr_mat
        return self._with_slack_constr_mat, self.constr_vec, self._with_slack_zeros_mat, self._with_slack_zeros_vec

class UpperTask(Task):
    """ Task describing the problem `Gx >= h`.
    """
    
    def __init__(self, n: int, m: int, slack_ratio: float = 1e4):
        super().__init__(n, m, slack_ratio)
        # cache
        self._with_slack_constr_mat[:, self.n:] = -np.eye(self.m)

    @override
    def with_slack_locked(self, m_other: int, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Returns `constr_mat` and `constr_vec` for previously solved tasks 
            including optimal slack variables, thus defining their nullspaces.
        """
        if self._with_slack_locked_mat is None:
            self.preallocate_locked_slack_matrix(m_other)
        
        self._with_slack_locked_mat[:, :self.n] = self.constr_mat
        G = self._with_slack_locked_mat[:, :self.n+m_other]  # slice only the required columns
        h = self.constr_vec + np.maximum(w, 0)
        return None, None, -G, -h

    @override
    def with_slack_ready(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Returns `constr_mat` and `constr_vec` with `m` added slack variables, 
            i.e. one for each row of the task.
        """
        self._with_slack_constr_mat[:, :self.n] = self.constr_mat
        return self._with_slack_zeros_mat, self._with_slack_zeros_vec, -self._with_slack_constr_mat, -self.constr_vec



class HQPSolver(object):
    """ Solver for HQP problems.
        Based on https://github.com/ritalaezza/sot-myo/blob/akos_re/src/HQPSolver.py
    """
    
    def __init__(self, SoT: List[Task] = None):
        # self.slack_boundary = 1e-5  # currently unused
        self._cache_SoT = SoT
        if SoT is not None:
            self._preallocate_tasks(SoT)
    
    def _preallocate_tasks(self, SoT: List[Task]):
        # pre-allocate matrices to be as big as the biggest m_i of the next 
        # task w.r.t. j, then simply slice by m_i (each j will be used by 
        # each i, but i is always larger than j)
        m_alloc = 0
        for i in range(len(SoT)-1, -1, -1):
            SoT[i].preallocate_locked_slack_matrix(m_alloc)
            m_alloc = max(m_alloc, SoT[i].m)
    
    def solve(self, SoT: List[Task] = None):
        """ Solves the stack of tasks and returns the optimal solution for the variables.
            Requires the task constraints to have been updated via `update_constraints()`
            beforehand to work. Requires the tasks to have the same value for `n`.
        """
        # use cache if no explicit SoT is given
        if len(SoT) is None:
            SoT = self._cache_SoT
        else:
            self._preallocate_tasks(SoT)
        
        # create matrix big enough to allocate the largest task requirements
        # Each task requires it's m_i rows and always the same n_i cols.
        # By preallocating a sum(m_i) rows we can progressevely populate and slice 
        # the matrix as we keep adding constrains, while the n_i + max(m_i) cols are
        # enough for each i,j combination and need to be properly sliced every time.
        # We also incorporate the vectors in the same matrix, so that it will be 
        # internally divided as [A b; G h], and thus we add a +1 in the cols.
        # Finally, we add a +1 in the rows as well, so to account for the extra row
        # that either (A,b) or (G,h) adds when is not in use when calling .with_slack_ready()
        # (e.g. if the task is an equality, A and b will have an appropriate size, 
        # while G and h will be a zero vector and a unit zero).
        ms = [task.m for task in SoT]
        required_rows = 1 + sum(ms)
        required_cols = 1 + max(ms) + SoT[0].n  # WARNING n must be the same in all tasks
        prealloc_mat = np.zeros((required_rows, required_cols))
        
        # loop through each task in the stack, descending in the hierarchy
        for i in range(len(SoT)):
            n_i, m_i = SoT[i].n, SoT[i].m
            
            # these are views of the cache matrix above, sliced so to fix task i
            # and all the previous tasks j in on big constraint matrix
            # A and b will load from the top, G and h from the bottom, and will 
            # naturally meet at the center.
            rows = (1 + m_i) + sum(ms[:i])  # e.g. for TASK 2 -> (1 + m_2) + sum([m_0, m_1])
            AG, bh = prealloc_mat[:rows, :n_i+m_i], prealloc_mat[:rows, n_i+m_i]  # notice, only last column!
            A, b = AG[:], bh[:]
            G, h = AG[::-1, :], bh[::-1]  # rows are flipped but it's fine
                        
            # set up task to solve
            # this returns only one between (A,b) or (G,h) of size m_i x (n_i + m_i), 
            # the other is a vector of zeros 1 x (n_i + m_i) and a single zero element
            Ai, bi, Gi, hi = SoT[i].with_slack_ready()
            A_idx = Ai.shape[0]  # one has m_i rows ...
            G_idx = Gi.shape[0]  # ... the other only 1
            A[:A_idx, :], b[:A_idx] = Ai, bi
            G[:G_idx, :], h[:G_idx] = Gi, hi
            
            
            # set up tasks over previously solved task in stack
            for j in range(i):
                m_j = SoT[j].m
                
                # return the previous tasks with some slack allowed.
                # this returns either (A,b) or (G,h), of size m_j x (n_j + m_i)
                # assuming n_j = n_i always (obviously, otherwise the solution
                # has inconsistent dimensions across the various tasks)
                Aj, bj, Gj, hj = SoT[j].with_slack_locked(m_i, SoT[j].slack_vec)
                
                # add previously solved tasks with optimal slack variables, 
                # s.t. task i is solved within their null-space
                if Aj is not None:
                    # task is an equality constraint
                    A[A_idx:A_idx+m_j, :] = Aj
                    b[A_idx:A_idx+m_j] = bj
                    A_idx += m_j
                if Gj is not None:
                    # task is an inequality constraint
                    G[G_idx:G_idx+m_j, :] = Gj
                    h[G_idx:G_idx+m_j] = hj
                    G_idx += m_j
            
            # set cost matrix and solve current hierarchy level
            P, q = SoT[i]._cost_mat, SoT[i]._cost_vec
            try:
                # tasks are given as <= inequalities, but `solve_qp` uses >=, so negate everything
                x = solve_qp(P, -q, -AG.T, -bh, A_idx, False)[0]
                # `sol` is return only at the very end, while slack values are stored in the current
                # task to be used for the following tasks
                sol, SoT[i].slack_vec = x[:n_i], x[n_i:]
            except Exception as ex:
                raise HQPTaskError(f"Error in task {i} ({type(SoT[i])}) : {ex}")

        # the last task solution is the optimal solution for the entire problem
        return sol

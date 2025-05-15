from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import numpy as np
from quadprog import solve_qp


class HQPTaskError(RuntimeError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class Task(object, metaclass=ABCMeta):
    """ Base abstract class for all tasks.
        Based on https://github.com/ritalaezza/sot-myo/blob/akos_re/src/Task.py
    """
    
    # EQUAL =  0,  #  Gx =  h
    # UPPER =  1,  #  Gx <  h
    # LOWER = -1,  # -Gx < -h
    
    def __init__(self, n: int, m: int, slack_ratio: float = 1e4):
        # constr_mat * x = constr_vec
        # where constr_mat is m x n
        #                x is n x 1
        #       constr_vec is m x 1
        #
        # slack_contrs_mat * y = constr_vec
        # where slack_contrs_mat is m x (n + m)
        #                      y is (n + m) x 1
        #             constr_vec is m x 1
        self.n: int = n  # number of free variables
        self.m: int = m  # number of constraint equations defining the task
        self.slack_ratio: float = slack_ratio  # between solution and w cost
        self.constr_mat = np.zeros((self.m,self.n))
        self.constr_vec = np.zeros((self.m,))
        self.slack_vec = np.zeros((self.m,))
        # cache
        self._cost_matrix = np.eye(self.n + self.m)
        self._cost_matrix[-self.m:, -self.m:] *= self.slack_ratio
        self._zeros = np.zeros((self.n + self.m,))
        # cache for the slack matrices
        self._with_slack_locked_mat: np.ndarray = None
        self._with_slack_contr_mat = np.zeros((self.m, self.n + self.m))
        self._with_slack_zeros_mat = np.zeros((1, self.n + self.m))
        self._with_slack_zeros_vec = np.zeros((1,))

    def preallocate_locked_slack_matrix(self, m_alloc):
        self._with_slack_locked_mat = np.zeros((self.m, self.n + m_alloc))

    @abstractmethod
    def with_slack_locked(self, m: int, w) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Returns `constr_mat` and `constr_vec` for previously solved tasks 
            including optimal slack variables, thus defining their nullspaces.
        """
        raise NotImplementedError("Return matrices A, b, G, h")

    @abstractmethod
    def with_slack(self, m) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Returns `constr_mat` and `constr_vec` with `m` added slack variables, 
            i.e. one for each row of the task.
        """
        # Based on constraint type, either `A` or `G` must be set to zeros (trivial
        # constraint) so to render adding previous stages easier (also, the 
        # solver needs it in the case of 1 task); also, either `b` or `h` must be 0 
        # as well, otherwise the solver won't find a solution.
        raise NotImplementedError("Return matrices A, b, G, h")
    
    @abstractmethod
    def compute(self, *args) -> "Task":
        """ Calculate `constr_mat` and `constr_vec`.
        """
        return self


class UpperTask(Task):
    
    def __init__(self, n: int, m: int, slack_ratio: float = 1e4):
        super().__init__(n, m, slack_ratio)
        # cache
        self._with_slack_contr_mat[:, self.n:] = -np.eye(self.m)

    def with_slack_locked(self, m_other: int, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Returns `constr_mat` and `constr_vec` for previously solved tasks 
            including optimal slack variables, thus defining their nullspaces.
        """
        if self._with_slack_locked_mat is None:
            self.preallocate_locked_slack_matrix(m_other)
        
        self._with_slack_locked_mat[:, :self.n] = self.constr_mat
        G = self._with_slack_locked_mat[:, :self.n+m_other]
        h = self.constr_vec + np.maximum(w, 0)
        return None, None, G, h

    def with_slack(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Returns `constr_mat` and `constr_vec` with `m` added slack variables, 
            i.e. one for each row of the task.
        """
        self._with_slack_contr_mat[:, :self.n] = self.constr_mat
        return self._with_slack_zeros_mat, self._with_slack_zeros_vec, self._with_slack_contr_mat, self.constr_vec

class EqualTask(Task):
    
    def __init__(self, n: int, m: int, slack_ratio: float = 1e4):
        super().__init__(n, m, slack_ratio)
        # cache
        self._with_slack_contr_mat[:, self.n:] = -np.eye(self.m)

    def with_slack_locked(self, m_other: int, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Returns `constr_mat` and `constr_vec` for previously solved tasks 
            including optimal slack variables, thus defining their nullspaces.
        """
        
        if self._with_slack_locked_mat is None:
            self.preallocate_locked_slack_matrix(m_other)
        
        self._with_slack_locked_mat[:, :self.n] = self.constr_mat
        A = self._with_slack_locked_mat[:, :self.n+m_other]
        b = self.constr_vec + w
        return A, b, None, None

    def with_slack(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Returns `constr_mat` and `constr_vec` with `m` added slack variables, 
            i.e. one for each row of the task.
        """
        self._with_slack_contr_mat[:, :self.n] = self.constr_mat
        return self._with_slack_contr_mat, self.constr_vec, self._with_slack_zeros_mat, self._with_slack_zeros_vec

class LowerTask(Task):
    
    def __init__(self, n: int, m: int, slack_ratio: float = 1e4):
        super().__init__(n, m, slack_ratio)
        # cache
        self._with_slack_contr_mat[:, self.n:] = np.eye(self.m)

    def with_slack_locked(self, m_other: int, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Returns `constr_mat` and `constr_vec` for previously solved tasks 
            including optimal slack variables, thus defining their nullspaces.
        """
        if self._with_slack_locked_mat is None:
            self.preallocate_locked_slack_matrix(m_other)
        
        self._with_slack_locked_mat[:, :self.n] = self.constr_mat
        G = self._with_slack_locked_mat[:, :self.n+m_other]
        h = self.constr_vec - np.maximum(w, 0)
        return None, None, G, h

    def with_slack(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Returns `constr_mat` and `constr_vec` with `m` added slack variables, 
            i.e. one for each row of the task.
        """
        self._with_slack_contr_mat[:, :self.n] = self.constr_mat
        return self._with_slack_zeros_mat, self._with_slack_zeros_vec, self._with_slack_contr_mat, self.constr_vec



# TODO do extensive tests, this MGITH BE BROKEN
class HQPSolver(object):
    """ Solver for HQP problems.
        Based on https://github.com/ritalaezza/sot-myo/blob/akos_re/src/HQPSolver.py
    """
    def __init__(self, SoT: List[Task] = None):
        self._cache_SoT = SoT
        if SoT is not None:
            self._preallocate_tasks(SoT)
        #self.slack_boundary = 1e-5      # Currently unused.
        #self.slack_ratio = 5e4          # Between solution and w cost
    
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

        # create matrix big enough to alloce the largest task requirements
        ms = [task.m for task in SoT]
        required_rows = 1 + sum(ms)
        required_cols = 1 + max(ms) + SoT[0].n  # n must be same across tasks
        cache_zeros = np.zeros((required_rows, required_cols))

        # loop through each task in the stack, descending in the hierachy
        for i in range(len(SoT)):
            n_i, m_i = SoT[i].n, SoT[i].m
            
            # these are views of the cache matrix above.
            # A and b will load from the top, G and h from the bottom, and will 
            # naturally meet at the center. given this definition, AB and bh are 
            # simply the same as A and b
            end = (m_i + 1) + sum([SoT[j].m for j in range(i)])
            A, b = cache_zeros[:end, :n_i+m_i], cache_zeros[:end, n_i+m_i]
            G, h = A[::-1, :], b[::-1]
            AG, bh = A[:], b[:]
                        
            # set up task to solve
            # this returns only one between (A,b) or (G,h) of size m_i x (n_i + m_i), 
            # the other is a vector of zeros 1 x (n_i + m_i) and a single zero element
            Ai, bi, Gi, hi = SoT[i].with_slack()
            A[:Ai.shape[0], :], b[:Ai.shape[0]] = Ai, bi
            G[:Gi.shape[0], :], h[:Gi.shape[0]] = Gi, hi
            A_idx = Ai.shape[0]
            G_idx = Gi.shape[0]
            
            # set up tasks over previously solved task in stack
            for j in range(i):
                
                # this returns either (A,b) or (G,h), of size m_j x (n_j + m_i)
                Aj, bj, Gj, hj = SoT[j].with_slack_locked(m_i, SoT[j].slack_vec)
                
                # add previously solved tasks with optimal slack variables, 
                # s.t. task i is solved within their null-space
                if Aj is not None:
                    # task has equality constraint
                    A[A_idx:A_idx+SoT[j].m, :] = Aj
                    b[A_idx:A_idx+SoT[j].m] = bj
                    A_idx += SoT[j].m
                if Gj is not None:
                    # task has inequality constraint
                    G[G_idx:G_idx+SoT[j].m, :] = Gj
                    h[G_idx:G_idx+SoT[j].m] = hj
                    G_idx += SoT[j].m

            # set cost matrix and solve level
            G, a = SoT[i]._cost_matrix, SoT[i]._zeros
            try:
                x = solve_qp(G, -a, -AG.T, -bh, A_idx, False)[0]
                sol, SoT[i].slack_vec = x[:n_i], x[n_i:]
            except Exception as ex:
                raise HQPTaskError(f"Error in task number {i} ({type(SoT[i])}) : {ex}")

        # the last task solution is the optimal solution for the entire problem
        return sol

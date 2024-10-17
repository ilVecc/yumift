from typing import List
from enum import Enum
import numpy as np
from quadprog import solve_qp


class HQPTaskError(RuntimeError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class Task(object):
    """ Base abstract class for all tasks.
        Based on https://github.com/ritalaezza/sot-myo/blob/akos_re/src/Task.py
    """
    
    class ConstraintType(Enum):
        EQUAL =  0,  #  Gx ==  h
        UPPER =  1,  #  Gx <=  h
        LOWER = -1,  # -Gx >= -h
    
    def __init__(self, dof: int, slack_ratio: float = 1e4):
        self.dof: int = dof
        self.constr_mat: np.ndarray = np.array([])
        self.constr_vec: np.ndarray = np.array([])
        self.constr_type: Task.ConstraintType = None
        self.slack_ratio: float = slack_ratio         # Between solution and w cost

    @property
    def ndim(self):
        """ Returns number of free variables.
        """
        return self.dof

    @property
    def mdim(self) -> int:
        """ Returns the number of constraint equations defining the task.
        """
        return self.constr_vec.size

    def with_slack_locked(self, m: int, w):
        """ Returns `constr_mat` and `constr_vec` for previously solved tasks 
            including optimal slack variables, thus defining their nullspaces.
        """
        if self.constr_type == Task.ConstraintType.EQUAL:
            A = np.hstack((self.constr_mat, np.zeros((self.mdim, m))))
            b = self.constr_vec + w
            G = None
            h = None
            
        elif self.constr_type == Task.ConstraintType.UPPER:
            G = np.hstack((self.constr_mat, np.zeros((self.mdim, m))))
            h = self.constr_vec + np.maximum(w, 0)
            A = None
            b = None
            
        elif self.constr_type == Task.ConstraintType.LOWER:
            G = np.hstack((self.constr_mat, np.zeros((self.mdim, m))))
            h = self.constr_vec - np.maximum(w, 0)
            A = None
            b = None

        return A, b, G, h

    def with_slack(self, m):
        """ Returns `constr_mat` and `constr_vec` with `m` added slack variables, 
            i.e. one for each row of the task.
        """
        # Based on constraint type, either `A` or `G` must be set to zeros (trivial
        # constraint) so to render adding previous stages easier (also, the 
        # solver needs it in the case of 1 task); also, either `b` or `h` must be 0 
        # as well, otherwise the solver won't find a solution.
        if self.constr_type == Task.ConstraintType.EQUAL:
            A = np.hstack((self.constr_mat, -np.eye(m)))
            b = self.constr_vec
            G = np.zeros((1, A.shape[1]))       # see comment above
            h = np.zeros((1, ))                 # see comment above

        elif self.constr_type == Task.ConstraintType.UPPER:
            G = np.hstack((self.constr_mat, -np.eye(m)))
            h = self.constr_vec
            A = np.zeros((1, G.shape[1]))       # see comment above
            b = np.zeros((1, ))                 # see comment above

        elif self.constr_type == Task.ConstraintType.LOWER:
            # BUG think this might have the wrong sign on the relaxation variable
            G = np.hstack((self.constr_mat, np.eye(m)))
            h = self.constr_vec
            A = np.zeros((1, G.shape[1]))       # see comment above
            b = np.zeros((1, ))                 # see comment above

        return A, b, G, h
    
    def compute(self, *args):
        return self


class HQPSolver(object):
    """ Solver for HQP problems.
        Based on https://github.com/ritalaezza/sot-myo/blob/akos_re/src/HQPSolver.py
    """
    def __init__(self):
        #self.SoT = SoT                  # List of Task objects
        #self.slack_boundary = 1e-5      # Currently unused.
        #self.slack_ratio = 5e4          # Between solution and w cost
        self.slack: List[np.ndarray] = []
        
    def solve(self, SoT: List[Task] = []):
        """ Solves the stack of tasks and returns the optimal solution for the variables.
            Requires the task constraints to have been updated via `update_constraints()`
            beforehand to work.
        """

        self.slack = [np.zeros((task.mdim,)) for task in SoT]

        # loop through each task in the stack, descending in the hierachy
        for i in range(len(SoT)):
            n_i = SoT[i].ndim
            m_i = SoT[i].mdim
            
            # set up task to solve
            A, b, G, h = SoT[i].with_slack(m_i)
                
            # set up tasks over currently solved task in stack:
            if i > 0:                
                # loop through all previously solved tasks:
                for j in range(i):
  
                    Aj, bj, Gj, hj = SoT[j].with_slack_locked(m_i, self.slack[j])
                    
                    # add previously solved tasks with optimal slack variables, s.t. task i is solved within their null-space.
                    if Aj is not None:
                        A = np.vstack((A, Aj))
                        b = np.concatenate((b, bj), axis=0)
                    if Gj is not None:
                        G = np.vstack((G, Gj))
                        h = np.concatenate((h, hj), axis=0)

            # set cost matrix and solve level:
            P = np.eye(n_i + m_i)
            P[-m_i:, -m_i:] = SoT[i].slack_ratio * np.eye(m_i)

            try:
                x = self._quadprog_solve_qp(P, np.zeros((n_i + m_i, )), G, h, A, b)
                self.slack[i] = x[n_i:]
                sol = x[:n_i]
            except Exception as ex:
                raise HQPTaskError(f"Error in task number {i} ({type(SoT[i])}) : {ex}")

        # the last task solution is the optimal solution for the entire problem
        return sol

    @staticmethod
    def _quadprog_solve_qp(
        P: np.ndarray, q: np.ndarray = None, 
        G: np.ndarray = None, h: np.ndarray = None, 
        A: np.ndarray = None, b: np.ndarray = None
    ):
        """ Wrapper for solver https://pypi.org/project/quadprog/
        """
        qp_G = 0.5 * (P + P.T)   # make sure P is symmetric
        if q is not None:
            qp_a = -q
        else:
            qp_a = np.zeros(qp_G.shape[0])
        if A is not None and G is not None:     # Mixed equality and inequality constraints
            qp_C = -np.vstack([A, G]).T
            qp_b = -np.hstack([b, h])
            meq = A.shape[0]
        elif A is not None and G is None:       # Only equality constraint (x = a reformed as -a <= x <= a) 
            qp_C = -A.T
            qp_b = -b
            meq = A.shape[0]
        else:                                   # Only ineqality constraint
            qp_C = -G.T
            qp_b = -h
            meq = 0

        return solve_qp(qp_G, qp_a, qp_C, qp_b, meq, False)[0]

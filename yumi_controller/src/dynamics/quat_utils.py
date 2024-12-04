from typing import List

import numpy as np
import quaternion as quat

#
# Here we represent quaternions with
#     q = w + xi + yj + zk  ∈ H
#       = (w, v)
#       = [cos(θ/2) sin(θ/2)*e]
# where (θ,e) is the axis-angle representation.
# 

def to_axis_angle(Q):
    Q = unit(Q)
    th = 2*np.arccos(Q[0])
    if th == 0:
        return 0., np.array([0., 0., 0.])
    r = Q[1:] / np.sin(th/2)
    return th, r

def to_rotation_vector(Q):
    a, k = to_axis_angle(Q)
    return a * k

def from_axis_angle(th, r):
    w = np.cos(th/2)
    v = np.sin(th/2) * r
    return np.concatenate([[w], v])

def from_rotation_vector(v):
    a = np.linalg.norm(v)
    k = v / (a or 1)
    return from_axis_angle(a, k)

def unit(Q):
    return Q / (np.linalg.norm(Q) or 1)
    
def conj(Q):
    return np.concatenate([[Q[0]], -Q[1:]])

def inv(Q):
    return conj(Q) / (np.linalg.norm(Q) or 1)

def mult(Q1, Q2):
    w1, v1 = Q1[0], Q1[1:]
    w2, v2 = Q2[0], Q2[1:]
    return np.concatenate([[w1*w2 - v1 @ v2], 
                           v1*w2 + w1*v2 + np.cross(v1, v2)])

def rot(v, Q):
    V = np.concatenate([[0], v])
    Q = unit(Q)
    return mult(mult(Q, V), inv(Q))[1:]
    
def log(Q):
    """ Logarithmic map for quaternions `log : H -> H`.
        When `Q` is a unit vector, the scalar component is zero.
    """
    w, v = (Q[0], Q[1:]) if len(Q) == 4 else (0, Q)
    Qnorm = np.linalg.norm(Q)
    return np.concatenate([[np.log(Qnorm)], 
                           np.arccos(w / (Qnorm or 1)) * v / (np.linalg.norm(v) or 1)])

def exp(Q):
    """ Exponential map for quaternions `exp : H -> H`.
        If `Q` has scalar component zero, the output is a unit vector.
    """
    w, v = (Q[0], Q[1:]) if len(Q) == 4 else (0, Q)
    vnorm = np.linalg.norm(v)
    return np.exp(w) * np.concatenate([[np.cos(vnorm)], 
                                       np.sin(vnorm) * v / (vnorm or 1)])

def jac_Q(Q):
    """ Creates the 3x4 map `dq = JQ @ dQ` where 
        - `Q` is a quaternion (as a 4-array)
        - `q = loq(Q)` (as a 3-array),
        - `@` is the matrix multiplication operation
        Also, `dQ = W * Q.conj()` where `W = [0 w]` is the angular velocity 
        quaternion (as a 4-array).
    """
    th, r = to_axis_angle(Q)
    if th == 0:
        return np.hstack([np.zeros((3,1)), np.eye(3)])
    a = th / 2
    ca, sa = np.cos(a), np.sin(a)
    JQ = 2 * np.hstack([ (a*ca - sa)/(sa**2) * r[:, np.newaxis], a/sa * np.eye(3) ])
    return JQ

def jac_q(q):
    """ Creates the 4x3 map `dQ = Jq @ dq` where 
        - `Q` is a quaternion (as a 4-array)
        - `q = loq(Q)` (as a 3-array),
        - `@` is the matrix multiplication operation
        Also, `dQ = W * Q.conj()` where `W = [0 w]` is the angular velocity 
        quaternion (as a 4-array).
    """
    assert len(q) == 3 or q[0] == 0, "Can accept only 3-array or vector quaternion"
    a = np.linalg.norm(q)
    if a == 0:
        return np.vstack([np.zeros((1,3)), np.eye(3)])
    ca, sa = np.cos(a), np.sin(a)
    r = q[-3:, np.newaxis] / a
    Jq = 0.5 * np.vstack([-sa * r.T,
                          sa/a * np.eye(3) + (ca - sa/a) * (r*r.T)])
    return Jq


# inspired by https://github.com/christophhagen/averaging-quaternions/blob/master/averageQuaternions.py
def quat_avg(*Q: List[np.ndarray]):
    """
    Calculates the average quaternion of N quaternions.
    :param Q: np.ndarray of np.quaternion with shape(N) or tuple of np.quaternion
    """
    # handle varargs
    if len(Q) == 1 and isinstance(Q[0], np.ndarray):
        Q = Q[0]
    else:
        Q = np.stack(Q)

    n = len(Q)
    Q = quat.as_float_array(Q)
    for i in range(1, n):
        # use minimum distance pairwise
        if Q[i-1, :] @ Q[i, :] < 0:
            Q[i, :] = -Q[i, :]
    A = np.zeros(shape=(4, 4))
    for i in range(n):
        q = Q[i, :]
        A += np.outer(q, q)  #  === q.T @ q
    A /= n
    eigvals, eigvecs = np.linalg.eig(A)
    # the vector will be type(complex) with only real part, so casting to real is safe
    avgQ = np.real(eigvecs[:, np.argmax(eigvals)])
    avgQ = quat.from_float_array(avgQ)
    return avgQ


def quat_diff(qi: np.quaternion, qf: np.quaternion, shortest: bool = True) -> np.quaternion:
    """ Returns the quaternion that achieves `qr * qi = qf` that using the 
        shorthest path (by default). 
        See more https://en.wikipedia.org/wiki/Quaternion#Geodesic_norm
        :param qi: initial quaternion
        :param qf: final quaternion
        :param shortest: whether to force using the shorthest path on the great circle or not
    """
    # this seemingly random equation comes directly from the geodesic distance 
    # between two quaternions, which we want to be less than 180 deg, meaning:
    #    np.arccos(2 * (p @ q) ** 2 - 1) < np.pi
    # can be simplified to 
    #    p @ q < 0
    # where @ is the quaternion dot product
    if shortest and quat.as_float_array(qi) @ quat.as_float_array(qf) < 0:
        qf = -qf
    return qf * qi.conj()

import numpy as np
import quaternion as quat

#
# Here we represent quaternions with
#     q = w + xi + yj + zk 
#       = (w, v)
#       = [cos(θ/2) sin(θ/2)*e]
# where (θ,e) is the axis-angle representation.
# 

def to_axis_angle(Q):
    a = 2*np.arccos(Q[0])
    if a == 0:
        return 0., np.array([0., 0., 0.])
    k = Q[1:] / np.sin(a/2)
    return a, k

def to_rotation_vector(Q):
    a, k = to_axis_angle(Q)
    return a * k

def from_axis_angle(a, k):
    w = np.cos(a/2)
    v = np.sin(a/2) * k
    Q = np.concatenate([[w], v])
    return Q

def from_rotation_vector(v):
    a = np.linalg.norm(v)
    if a != 0:
        k = v / a
    else:
        k = np.array([0., 0., 0.])
    return from_axis_angle(a, k)

def unit(Q):
    if Q[0] != 1:
        Q /= np.linalg.norm(Q)
    return Q

def conj(Q):
    Q[1:] *= -1
    return Q

def inv(Q):
    return conj(Q) / np.linalg.norm(Q)

def rot(v, Q):
    V = np.concatenate([[0], v])
    return mult(mult(Q, V), inv(Q))[1:]
    
def mult(Q1, Q2):
    w1 = Q1[0]
    v1 = Q1[1:]
    w2 = Q2[0]
    v2 = Q2[1:]
    Q = np.concatenate([
        [w1*w2 - v1 @ v2],
        v1*w2 + w1*v2 + np.cross(v1, v2)
    ])
    return Q

def log(Q):
    """ Logarithmic map for quaternions. 
        Implements the mapping log : S^3 -> R^3 """
    w = Q[0]
    v = Q[1:]
    if np.abs(w) == 1:
        return np.array([0., 0., 0.])
    return 2*np.arccos(w) * v / np.linalg.norm(v)

def exp(q):
    """ Exponential map for quaternions. 
        Implements the mapping exp : R^3 -> S^3 """
    a = np.linalg.norm(q)
    if a == 0:
        return np.array([1., 0., 0., 0.])
    w = np.cos(a/2)
    v = np.sin(a/2) * (q / a)
    return np.concatenate([[w], v])

def jac_Q(Q):
    a, k = to_axis_angle(Q)
    if a == 0:
        return np.hstack([np.zeros((3,1)), np.eye(3)])
    th = a / 2
    JQ = 2 * np.hstack([
        (th*np.cos(th) - np.sin(th))/(np.sin(th)**2) * k[:, np.newaxis],
        th/np.sin(th) * np.eye(3)
    ])
    return JQ

def jac_q(q):
    a = np.linalg.norm(q)
    if a == 0:
        return np.vstack([np.zeros((1,3)), np.eye(3)])
    k = q / a
    th = a / 2
    k = k[:, np.newaxis]
    Jq = 0.5 * np.vstack([
        -np.sin(th) * k.T,
        np.sin(th)/th * (np.eye(3) - k*k.T) + np.cos(th)*k*k.T
    ])
    return Jq


# inspired by https://github.com/christophhagen/averaging-quaternions/blob/master/averageQuaternions.py
def quat_avg(*Q):
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

def quat_min_diff(qi: np.quaternion, qf: np.quaternion) -> np.quaternion:
    """ Returns the representation of `qf` that is nearest to `qi` for interpolation purposes.
        :param qi: initial quaternion
        :param qf: final quaternion
    """
    if quat.as_float_array(qi) @ quat.as_float_array(qf) < 0:
        return -qf
    return qf

# TODO is this useful?
# def quat_consistent_diff(qi: np.quaternion, qf: np.quaternion, qr: np.quaternion = None):
#     qf = quat_min_diff(qi, qf)


if __name__ == "__main__":
    Q1 = np.array([1., 0., 0., 0.])
    Q2 = from_axis_angle(np.pi/4, np.array([0., 0., 1.]))
    
    a, k = to_axis_angle(mult(Q2, Q2))
    assert np.allclose( a, np.pi/2 ) and np.allclose( k, np.array([0., 0., 1.]) )
    assert np.allclose( Q1, exp(log(Q1)) )
    assert np.allclose( Q2, exp(log(Q2)) )
    assert np.allclose( rot(np.array([1., 1., 0.]), Q2), np.array([0., np.sqrt(2), 0.]) )
    assert np.allclose( jac_Q(Q2) @ jac_q(log(Q2)), np.eye(3) )

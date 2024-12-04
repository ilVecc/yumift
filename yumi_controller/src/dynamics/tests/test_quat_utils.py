import numpy as np

from .. import *

def main():
    Q1 = np.array([1., 0., 0., 0.])
    Q2 = from_axis_angle(np.pi/4, np.array([0., 0., 1.]))
    
    a, k = to_axis_angle(mult(Q2, Q2))
    assert np.allclose( a, np.pi/2 ) and np.allclose( k, np.array([0., 0., 1.]) )
    assert np.allclose( Q1, exp(log(Q1)) )
    assert np.allclose( Q2, exp(log(Q2)) )
    assert np.allclose( rot(np.array([1., 1., 0.]), Q2), np.array([0., np.sqrt(2), 0.]) )
    
    JQ, Jq = jac_Q(Q2), jac_q(log(Q2))
    assert np.allclose( JQ @ Jq, np.eye(3) )
    assert np.allclose( jac_q(log(Q2)) @ jac_Q(Q2), np.eye(4) )


if __name__ == "__main__":
    main()

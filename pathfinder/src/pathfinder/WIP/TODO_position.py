import numpy as np


class vector(np.ndarray):

    def __new__(cls, x: np.float64, y: np.float64, z: np.float64):
        obj = np.array([x, y, z]).view(cls)
        return obj
    
    @property
    def x(self):
        return self[0]
    
    @property
    def y(self):
        return self[1]
    
    @property
    def z(self):
        return self[2]
    
    def __repr__(self) -> str:
        # TODO not working
        return f"{self.__class__.__name__}({self.x}, {self.y}, {self.z})"

np.vector = vector
np.sctypeDict["vector"] = np.dtype(vector)


v = np.vector(2., 3, 4)
print(v)               # [2. 3. 4.]
print(type(v))         # <class '__main__.vector'>
print(v.dtype)         # float64
print(type(v.dtype))   # <class 'numpy.dtype'>



import quaternion as quat

q = np.quaternion(1, 2, 3, 4)
print(q)               # quaternion(1, 2, 3, 4)
print(type(q))         # <class 'quaternion.quaternion'>
print(q.dtype)         # quaternion
print(type(q.dtype))   # <class 'numpy.dtype'>

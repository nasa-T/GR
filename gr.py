import numpy as np
from inspect import signature
import sympy as sp

class Metric:
    """
    x0 is the time component and x1, 2, and 3 are the spatial components:
        x0, 1, 2, and 3 can be array-like with strings denoting expressions with the number of dimensions or a single string for the diagonal components of each coordinate
    coords are the names of the coordinates in a string separated by a single space
    """
    def __init__(self, x0, x1, x2, x3=None, coords='t x y'):
        assert len(coords.split(' ')) in [3, 4]
        if len(coords.split(' ')) == 3:
            pre_met = [x0, x1, x2]
        else:
            try:
                assert x3 != None
            except:
                raise Exception('x3 must also be specified if you have a fourth coordinate')
            pre_met = [x0, x1, x2, x3]
        symb_list = sp.symbols(coords)
        self.coord_names = coords.split(' ')
        self._sym_dict = {}
        self._idx_dict = {}
        for i, name in enumerate(self.coord_names):
            self._sym_dict[name] = symb_list[i]
            self._idx_dict[name] = i

        self.dims = len(pre_met)
        metric = []
        for i, x in enumerate(pre_met):
            if type(x) in [list, tuple]:
                assert len(x) == self.dims
                metric += [x]
                # print(metric)
            else:
                x_array = np.zeros(self.dims, dtype=type(sp.sympify('0')))
                for j in range(self.dims):
                    x_array[j] = sp.sympify('0')
                x_array[i] = sp.sympify(x)
                metric += [x_array]
        self.metric = sp.Matrix(metric)
        self.inverse_metric = self.metric.inv()
        # 1/determinant(self.metric) * self.metric.adjoint()

        self.christoffels = {}
        
    def get_element(self, a, b):
        if type(a) == str:
            try:
                assert a in self.coord_names
                a_idx = self._idx_dict[a]
            except:
                raise Exception('Choose an appropriate coordinate name: {}'.format(self.coord_names))
        elif type(a) == int:
            a_idx = a
        else:
            raise Exception('Make sure parameter a is either the name (string) of the coordinate or the index (int) of it.')
        if type(b) == str:
            try:
                assert b in self.coord_names
                b_idx = self._idx_dict[b]
            except:
                raise Exception('Choose an appropriate coordinate name: {}'.format(self.coord_names))
        elif type(b) == int:
            b_idx = b
        else:
            raise Exception('Make sure parameter b is either the name (string) of the coordinate or the index (int) of it.')
        return self.metric[a_idx,b_idx]

    def get_inv_element(self, a, b):
        if type(a) == str:
            try:
                assert a in self.coord_names
                a_idx = self._idx_dict[a]
            except:
                raise Exception('Choose an appropriate coordinate name: {}'.format(self.coord_names))
        elif type(a) == int:
            assert a < self.dims
            a_idx = a
        else:
            raise Exception('Make sure parameter a is either the name (string) of the coordinate or the index (int) of it.')
        if type(b) == str:
            try:
                assert b in self.coord_names
                b_idx = self._idx_dict[b]
            except:
                raise Exception('Choose an appropriate coordinate name: {}'.format(self.coord_names))
        elif type(b) == int:
            assert b < self.dims
            b_idx = b
        else:
            raise Exception('Make sure parameter b is either the name (string) of the coordinate or the index (int) of it.')
        return self.inverse_metric[a_idx,b_idx]

    def christoffel(self, a, b, c, **kwargs):
        if (a, b, c) in self.christoffels:
            return self.christoffels[(a,b,c)]
        tot = 0
        if type(a) == int:
            assert a < self.dims
            a_str = self.coord_names[a]
        else:
            assert type(a) == str
            a_str = a
        if type(b) == int:
            assert b < self.dims
            b_str = self.coord_names[b]
        else:
            assert type(b) == str
            b_str = b
        if type(c) == int:
            assert c < self.dims
            c_str = self.coord_names[c]
        else:
            assert type(c) == str
            c_str = c
        for i in range(self.dims):
            coeff = 1/2*self.get_inv_element(i,a) 
            if coeff != 0:
                ders = sp.diff(self.get_element(i, b), c_str) + sp.diff(self.get_element(i, c), b_str) - sp.diff(self.get_element(b, c), self.coord_names[i])
                tot += coeff * ders
        self.christoffels[(a,b,c)] = tot
        if kwargs:
            for key in kwargs.keys():
                try:
                    assert key in self.coord_names
                except:
                    raise Exception('The keywords must be valid coordinate names: {}'.format(self.coord_names))
            tot = tot.subs(kwargs)
        
        return tot

    # def eval_christoffel(self, a, b, c, **kwargs):
    #     for key in kwargs.keys():
    #         try:
    #             assert key in self.coord_names
    #         except:
    #             raise Exception('The keywords must be a valid coordinate name: {}'.format(self.coord_names))
    #     return self.christoffel(a,b,c).subs(kwargs)
    
    def __str__(self):
        return str(self.metric)
    
    def __repr__(self):
        return repr(self.metric)

# class metric_function:
#     """
#     fun is a function with less than four arguments
#     """
#     def __init__(self, fun):
#         self.func = fun
#         sig = signature(self.func)
#         self.n_params = len(sig.parameters)
#         assert self.n_params <= 4
#         self.key_names = []
#         for param in sig.parameters:
#             self.key_names.append(param.name)

#     def __call__(self, **kwargs):
#         assert len(kwargs) == self.n_params
#         param_list = np.zeros(self.n_params)
#         for param in kwargs:
#             try:
#                 i = self.key_names.index(param)
#             except:
#                 raise Exception('{} is not a parameter in this function.'.format(param))
#             param_list[i] = kwargs[param]
#         if self.n_params == 0:
#             return self.func()
#         elif self.n_params == 1:
#             return self.func(param_list[0])
#         elif self.n_params == 2:
#             return self.func(param_list[0], param_list[1])
#         elif self.n_params == 3:
#             return self.func(param_list[0], param_list[1], param_list[2])
#         elif self.n_params == 4:
#             return self.func(param_list[0], param_list[1], param_list[2], param_list[3])

def determinant(matrix):
    if matrix.shape[0] < 2 or matrix.shape[1] < 2 or not matrix.is_square:
        raise Exception("Matrix must be square and at least 2x2")
        
    if matrix.shape[0] == matrix.shape[1] == 2:
        return matrix[0,0]*matrix[1,1] - matrix[0,1] * matrix[1,0]
    else:
        det = sp.sympify(0)
        for i in range(matrix.shape[0]):
            det += sp.sympify('(-1)**{}'.format(i)) * matrix[0,i] * determinant(matrix[1:,1:])
        return det

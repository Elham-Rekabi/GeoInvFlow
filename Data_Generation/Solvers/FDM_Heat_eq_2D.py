import torch 
import numpy as np
import matplotlib.pyplot as plt
import scipy

# Solves:  ∇·(xi_1 ∇u) = xi_2  on (0, nx*hx) x (0, ny*hy)
# Boundary Conditions: {'Left': ('BC_Neumann', val), 'Right': ('BC_Neumann', val), 'Bottom': ('BC_Dirichlet', val), 'Top': ('BC_Dirichlet', val)}

class Finite_Difference_Heat_eq_2D:

    def __init__(self, nx, ny, hx, hy, BCs, xi_1, xi_2 = None):
        self.ny = ny
        self.nx = nx
        self.hx = hx
        self.hy = hy
        self.BCs = BCs
        self.xi_1 = xi_1
        self.xi_2 = xi_2
        

    def Stiffness_Matrix_A(self):
        ny, nx, hx, hy = self.ny, self.nx, self.hx, self.hy
        N = nx*ny
        A = torch.zeros((N, N), dtype=torch.float32)
        for j in range(ny):
            for i in range(nx):
                is_boundary = False
                if i == 0:
                    is_boundary = True
                    Left_bc_type, Left_bc_val = self.BCs['Left']
                if i == nx-1:
                    is_boundary = True
                    Right_bc_type, Right_bc_val = self.BCs['Right']
                if j == 0:
                    is_boundary = True
                    Bottom_bc_type, Bottom_bc_val = self.BCs['Bottom']
                if j == ny-1:
                    is_boundary = True
                    Top_bc_type, Top_bc_val = self.BCs['Top']

                xi_1 = self.xi_1
                xi_2 = self.xi_2
                
                k = j*nx +i
                
                x_i = i*hx
                y_j = j*hy
                
                x_i_1 = x_i + 0.5*hx
                x_i_2 = x_i - 0.5*hx
                y_j_1 = y_j + 0.5*hy
                y_j_2 = y_j - 0.5*hy
                
                if is_boundary == False:
                
                    A[k, k+nx] = xi_1(x_i, y_j_1) / hy**2
                    A[k, k-nx] = xi_1(x_i, y_j_2) / hy**2
                    A[k, k+1] = xi_1(x_i_1, y_j) / hx**2
                    A[k, k-1] = xi_1(x_i_2, y_j) / hx**2
                    A[k, k] =  - (xi_1(x_i_1, y_j) + xi_1(x_i_2, y_j)) / hx**2 - (xi_1(x_i, y_j_1) + xi_1(x_i, y_j_2)) / hy**2
                    
                else:
                    # Boundary Conditions
                    if i == 0 and Left_bc_type == 'BC_Neumann':
                        KE = xi_1(x_i_1, y_j) / hx**2 
                        A[k, k+1] = KE
                        A[k,k] = - KE
                        KS = xi_1(x_i, y_j_2) / hy**2
                        if j > 0: A[k, k-nx] = KS
                        A[k,k] += - KS
                        KN = xi_1(x_i, y_j_1) / hy**2
                        if j < ny-1: A[k, k+nx] = KN
                        A[k,k] += - KN
                    if i == 0 and Left_bc_type == 'BC_Dirichlet':
                        A[k, :] = 0.0
                        A[k,k] = 1.0
                    
                        
                    if i == nx-1 and Right_bc_type == 'BC_Neumann':
                        KW = xi_1(x_i_2, y_j) / hx**2
                        A[k, k-1] = KW
                        A[k, k] = - KW
                        KS = xi_1(x_i, y_j_2) / hy**2
                        if j > 0: A[k, k-nx] = KS
                        A[k,k] += - KS
                        KN = xi_1(x_i, y_j_1) / hy**2
                        if j < ny-1: A[k, k+nx] = KN
                        A[k,k] += - KN   
                              
                    if i == nx-1 and Right_bc_type == 'BC_Dirichlet':
                        A[k, :] = 0.0
                        A[k,k] = 1.0
                        
                    if j == 0 and Bottom_bc_type == 'BC_Neumann':
                        KN = xi_1(x_i, y_j_1) / hy**2
                        A[k, k+nx] = KN
                        A[k,k] = - KN
                        KW = xi_1(x_i_2, y_j) / hx**2
                        if i > 0: A[k, k-1] = KW
                        A[k,k] += - KW   
                        KE = xi_1(x_i_1, y_j) / hx**2    
                        if i < nx-1: A[k, k+1] = KE
                        A[k,k] += - KE
                        
                    if j == 0 and Bottom_bc_type == 'BC_Dirichlet':
                        A[k, :] = 0.0
                        A[k,k] = 1.0
                        
                    if j == ny-1 and Top_bc_type == 'BC_Neumann':
                        KS = xi_1(x_i, y_j_2) / hy**2
                        A[k, k-nx] = KS
                        A[k,k] = - KS
                        KW = xi_1(x_i_2, y_j) / hx**2
                        if i > 0: A[k, k-1] = KW
                        A[k,k] += - KW       
                        KE = xi_1(x_i_1, y_j) / hx**2   
                        if i < nx-1: A[k, k+1] = KE
                        A[k,k] += - KE    
                            
                    if j == ny-1 and Top_bc_type == 'BC_Dirichlet':
                        A[k, :] = 0.0
                        A[k,k] = 1.0

        return A
    
    def Load_vector(self):
        ny, nx, hx, hy = self.ny, self.nx, self.hx, self.hy
        N = nx*ny
        f = torch.zeros((N,), dtype=torch.float32)
        for j in range(ny):
            for i in range(nx):
                is_boundary = False
                if i == 0:
                    is_boundary = True
                    Left_bc_type, Left_bc_val = self.BCs['Left']
                if i == nx-1:
                    is_boundary = True
                    Right_bc_type, Right_bc_val = self.BCs['Right']
                if j == 0:
                    is_boundary = True
                    Bottom_bc_type, Bottom_bc_val = self.BCs['Bottom']
                if j == ny-1:
                    is_boundary = True
                    Top_bc_type, Top_bc_val = self.BCs['Top']

                
                k = j*nx +i
                
                x_i = i*hx
                y_j = j*hy
                
                x_i_1 = x_i + 0.5*hx
                x_i_2 = x_i - 0.5*hx
                y_j_1 = y_j + 0.5*hy
                y_j_2 = y_j - 0.5*hy
                
                f[k] = self.xi_2(x_i, y_j).item()
                
                if i == 0 and Left_bc_type == 'BC_Neumann':
                    f[k] += -self.xi_1(x_i_2, y_j) * Left_bc_val(x_i, y_j) / hx
                if i == 0 and Left_bc_type == 'BC_Dirichlet':
                    f[k] = Left_bc_val(x_i, y_j)
                    
                if i == nx-1 and Right_bc_type == 'BC_Neumann':
                    f[k] += -self.xi_1(x_i_1, y_j) * Right_bc_val(x_i, y_j) / hx
                if i == nx-1 and Right_bc_type == 'BC_Dirichlet':
                    f[k] = Right_bc_val(x_i, y_j)

                if j == 0 and Bottom_bc_type == 'BC_Neumann':
                    f[k] += -self.xi_1(x_i, y_j_2) * Bottom_bc_val(x_i, y_j) / hy
                if j == 0 and Bottom_bc_type == 'BC_Dirichlet':
                    f[k] = Bottom_bc_val(x_i, y_j)

                if j == ny-1 and Top_bc_type == 'BC_Neumann':
                    f[k] += -self.xi_1(x_i, y_j_1) * Top_bc_val(x_i, y_j) / hy
                if j == ny-1 and Top_bc_type == 'BC_Dirichlet':
                    f[k] = Top_bc_val(x_i, y_j)

        return f
    
    def solve(self, A, b):

        u = torch.linalg.solve(A, b)
        u_mat = u.reshape((self.ny, self.nx)).detach().numpy()
        
        return u, u_mat
    
    




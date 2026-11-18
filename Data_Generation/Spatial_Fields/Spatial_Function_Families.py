import torch 
import numpy as np
import matplotlib.pyplot as plt
import scipy


class Function_Families:

    def __init__(self, hx, hy, family_type, params  = None):
        
        self.family_type = family_type
        self.hx = hx
        self.hy = hy
        
        if family_type in ['Gaussian', 'Composite']:
            
            if params is None:
                self.Gaussian_J = 4
                self.Gaussian_A = np.random.randn(self.Gaussian_J)
                self.Gaussian_B = np.random.randn(self.Gaussian_J)
                self.Gaussian_cx = np.random.rand(self.Gaussian_J)
                self.Gaussian_cy = np.random.rand(self.Gaussian_J)
                self.Gaussian_r = 0.2 + 0.25 * np.random.rand(self.Gaussian_J)
            else:
                self.Gaussian_J = params['J']
                self.Gaussian_A = params['A']
                self.Gaussian_B = params['B']
                self.Gaussian_cx = params['cx']
                self.Gaussian_cy = params['cy']
                self.Gaussian_r = params['r']
       

        if family_type in ['Sine_Cosine', 'Composite']:
            
            nx_max = 3
            ny_max = 3
            
            if params is None:
                self.Sine_Cosine_J = 5
                self.Sine_Cosine_A = np.abs(np.random.randn(self.Sine_Cosine_J))
                self.Sine_Cosine_A = np.sort(self.Sine_Cosine_A, axis=0)[::-1] # Sort amplitudes in descending order
                self.Sine_Cosine_t1 = np.random.uniform(0, 2*np.pi, self.Sine_Cosine_J) # Phase shift for sine: 0 to 2pi
                self.Sine_Cosine_t2 = np.random.uniform(0, 2*np.pi, self.Sine_Cosine_J) # Phase shift for cosine: 0 to 2pi
                self.Sine_Cosine_w1 = np.pi*np.random.uniform(0.1, nx_max, size=self.Sine_Cosine_J) # Frequency for sine
                self.Sine_Cosine_w1 = np.sort(self.Sine_Cosine_w1) # Sort frequencies 
                self.Sine_Cosine_w2 = np.pi*np.random.uniform(0.1, ny_max, size=self.Sine_Cosine_J) # Frequency for cosine
                self.Sine_Cosine_w2 = np.sort(self.Sine_Cosine_w2) # Sort frequencies

            else:
                self.Sine_Cosine_J = params['J']
                self.Sine_Cosine_A = params['A']
                self.Sine_Cosine_t1 = params['theta1']
                self.Sine_Cosine_t2 = params['theta2']
                self.Sine_Cosine_w1 = params['w1']
                self.Sine_Cosine_w2 = params['w2']

        if family_type in ['Polynomial', 'Composite']:
            N = int(5)
            if params is None:
                self.Polynomial_order = N
                self.Polynomial_coefficients = np.random.randn(N*(N-1)//2+2*N +1)
            
            else:
                self.Polynomial_order = params['order']
                self.Polynomial_coefficients = params['coefficients']
        
        if family_type not in ['Gaussian', 'Sine_Cosine','Polynomial', 'Composite']:
            raise ValueError("Unsupported family type. Choose from 'Gaussian', 'Sine_Cosine','Polynomial' or 'Composite'.") 
        
        self.shift()
        
    def shift(self):
        
        hx = self.hx
        hy = self.hy
        nx = int(1.0/hx) + 1
        ny = int(1.0/hy) + 1
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)
        Z = self.evaluate(X, Y)
        Z_min = np.min(Z)
        Z_max = np.max(Z)
        self.L = Z_min
        self.U = Z_max
        
        
    def evaluate(self, x, y):
        
        
        if self.family_type == 'Gaussian':
            z = 0.0
            for j in range(self.Gaussian_J):
                z += self.Gaussian_A[j] * np.exp(-((x-self.Gaussian_cx[j])**2 + (y-self.Gaussian_cy[j])**2)/(self.Gaussian_r[j]**2)) + self.Gaussian_B[j]
            return z
        
        elif self.family_type == 'Sine_Cosine':
            
            z = 0.0
            for j in range(self.Sine_Cosine_J):
                z += self.Sine_Cosine_A[j] * np.sin(self.Sine_Cosine_w1[j]*x+self.Sine_Cosine_t1[j]) * np.sin(self.Sine_Cosine_w2[j]*y+self.Sine_Cosine_t2[j])
            return z
        
        elif self.family_type == 'Polynomial':
            z = 0.0
            idx = 0
            for i in range(self.Polynomial_order+1):
                for j in range(self.Polynomial_order+1-i):
                    scale = 1.0 / (i + j + 1)
                    xc = 2 * x - 1
                    yc = 2 * y - 1
                    z += scale * self.Polynomial_coefficients[idx] * (xc**i) * (yc**j)
                    idx += 1
            return z
        
        elif self.family_type == 'Composite':

            z = 0.0
            for j in range(self.Gaussian_J):
                z += self.Gaussian_A[j] * np.exp(-((x-self.Gaussian_cx[j])**2 + (y-self.Gaussian_cy[j])**2)/(self.Gaussian_r[j]**2)) + self.Gaussian_B[j]

            for j in range(self.Sine_Cosine_J):
                z += self.Sine_Cosine_A[j] * np.sin(self.Sine_Cosine_w1[j]*x+self.Sine_Cosine_t1[j]) * np.sin(self.Sine_Cosine_w2[j]*y+self.Sine_Cosine_t2[j])

            idx = 0
            for i in range(self.Polynomial_order+1):
                for j in range(self.Polynomial_order+1-i):
                    z += self.Polynomial_coefficients[idx] * (x**i) * (y**j)
                    idx += 1
            return z

        else:
            raise ValueError("Unsupported family type. Choose from 'Gaussian', 'Sine_Cosine','Polynomial' or 'Composite'.")
        
        
    def evaluate_shifted(self, x, y):
        z = self.evaluate(x, y)
        z_normalized = z - self.L +0.1
        return z_normalized
    
    def evaluate_normalized(self, x, y):
        z = self.evaluate(x, y)
        z_normalized = (z - self.L)/ (self.U - self.L)
        return z_normalized 
        
        
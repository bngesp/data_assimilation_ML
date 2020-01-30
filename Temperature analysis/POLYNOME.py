#%%
import matplotlib.pyplot as plt
import numpy as np



#%%
def p(x):
    return x**4 - 4*x**2 + 3*x


#%%
class Polynomial:
    
    def __init__(self, *coefficients):
        """ input: coefficients are in the form a_n, ...a_1, a_0 
        """
        self.coefficients = list(coefficients) # tuple is turned into a list
        #self.coefficients.reverse()
     
    def __repr__(self):
        """
        method to return the canonical string representation 
        of a polynomial.
   
        """
        return "Polynomial" + str(self.coefficients)
            
    def __call__(self, x):    
        res = 0
        for index, coeff in enumerate(self.coefficients[::-1]):
            res += coeff * np.power(x, index)
        return res 
    


#%%
p = Polynomial(31.71049579, -1.49946772, -5.49572183, -10.97552262, 2.11497251, 26.5201382, 0.7778732, -14.94275691)
X = range(50)
F = p(X)
plt.plot(X, F, 'r')

plt.show()

#%%

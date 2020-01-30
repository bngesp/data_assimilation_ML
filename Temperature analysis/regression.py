import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# dataset
x, y = make_regression(n_samples=100, n_features=1, noise=10)
y = y.reshape(y.shape[0], 1)
X = np.hstack((x, np.ones(x.shape)))
teta = np.random.randn(2,1)

def model(X, teta):
    return X.dot(teta)


def cout(x, y, teta):
    m = len(y)
    return 1/(2*m) * np.sum((model(x,teta) - y)**2)

def grad(X, y, teta):
    m = len(y)
    return 1/m * X.T.dot(model(X, teta) - y)


def gradient_descent(X, y, teta, learning_rate, n_iterations):
    
    cost_history = np.zeros(n_iterations) # création d'un tableau de stockage pour enregistrer l'évolution du Cout du modele
    
    for i in range(0, n_iterations):
        teta = teta - learning_rate * grad(X, y, teta) # mise a jour du parametre teta (formule du gradient descent)
        cost_history[i] = cout(X, y, teta) # on enregistre la valeur du Cout au tour i dans cost_history[i]
        
    return teta, cost_history


n_iterations = 1000
learning_rate = 0.01


teta_final, cost_history = gradient_descent(X, y, teta, learning_rate, n_iterations)

# création d'un vecteur prédictions qui contient les prédictions de notre modele final
predictions = model(X, teta_final)

# Affiche les résultats de prédictions (en rouge) par rapport a notre Dataset (en bleu)
plt.scatter(x, y)
#plt.plot(range(n_iterations), cost_history)

plt.plot(x, predictions, c='r')

plt.show()
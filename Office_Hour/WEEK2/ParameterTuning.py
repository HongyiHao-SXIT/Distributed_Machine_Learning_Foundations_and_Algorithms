import numpy as np
import matplotlib.pyplot as plt

def polynomial_loss(theta, X ,y):
    m = len(y)
    h = np.dot(X ,theta)
    loss = np.sum((h - y)** 2) / m
    return loss

def stochastic_batch_gradient_descent(X, y, batch_size, learning_rate, num_epochs):
    m, n = X.shape
    theta = np.zeros(n)
    theta_history = [theta.copy()]
    num_batches = m // batch_size
    for epoch in range(num_epochs):
        indices = np.random.permutation(m)
        X = X[indices]
        y = y[indices]

        for batch in range(num_batches):
            start_index = batch * batch_size
            end_index = start_index + batch_size
            X_batch = X[start_index:end_index, :]
            y_batch = y[start_index:end_index]
            
            h_batch = np.dot(X_batch, theta)
            error_batch = h_batch - y_batch
            gradient = np.dot(X_batch.T, error_batch) / batch_size
            theta = theta - learning_rate * gradient
            theta_history.append(theta.copy())
    
    return theta, theta_history

def plot_loss_contour(X, y, theta_history, theta_range=(-1, 6)):
    theta0_vals = np.linspace(theta_range[0], theta_range[1], 100)
    theta1_vals = np.linspace(theta_range[0], theta_range[1], 100)
    theta0_mesh, theta1_mesh = np.meshgrid(theta0_vals, theta1_vals)
    
    loss_vals = np.zeros_like(theta0_mesh)
    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            theta = np.array([theta0_mesh[i, j], theta1_mesh[i, j]])
            loss_vals[i, j] = polynomial_loss(theta, X, y)
    
    plt.figure(figsize=(8, 6))
    plt.contour(theta0_mesh, theta1_mesh, loss_vals, levels=20, cmap='jet')
    
    theta_history = np.array(theta_history)
    plt.plot(theta_history[:, 0], theta_history[:, 1], 'bo-', alpha=0.4, linewidth=0.5, markersize=2)
    plt.plot(theta_history[-1, 0], theta_history[-1, 1], 'ro', alpha=0.8, markersize=5)
    
    plt.xlabel('theta0')
    plt.ylabel('theta1')
    plt.title('Contour of Loss Function and Gradient Descent Path')
    plt.colorbar(label="Loss")
    plt.grid(True)
    plt.show()


np.random.seed(0)
X = np.random.rand(100, 2)
#y = 2 * X[:, 0] + 3 * X[:, 1] + 0.01*np.random.randn(100)
y = 2 * X[:, 0] + 3 * X[:, 1] + 10*np.random.randn(100)

batch_size = 1
learning_rate = 0.3
num_epochs = 100

theta_final, theta_history = stochastic_batch_gradient_descent(
    X, y, batch_size, learning_rate, num_epochs
)

plot_loss_contour(X, y, theta_history)

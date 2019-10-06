import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io
from scipy.interpolate import griddata
import time
from plotting import newfig, savefig
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, t, x, u,
                 layers_u, layers_d):
        
        X = np.concatenate([t, x], 1)
        self.lb = X.min(0)
        self.ub = X.max(0)
        
        # data on velocity (inside the domain)
        self.t = t
        self.x = x
        self.u = u
        
        # layers
        self.layers_u = layers_u
        self.layers_d = layers_d
        
        # initialize NN
        self.weights_u, self.biases_u = self.initialize_NN(layers_u)
        self.weights_d, self.biases_d = self.initialize_NN(layers_d)
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        # placeholders for data on concentration (inside the domain)
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, 1])
        
        # physics informed neural networks (inside the domain)
        (self.u_pred,
         self.d_pred,
         self.eq_pred) = self.net_u(self.t_tf, self.x_tf)
        
        # loss
        self.loss = tf.reduce_sum(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_sum(tf.square(self.eq_pred))
        
        # optimizers
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.sin(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
            
    def net_u(self, t, x):
        u = self.neural_net(tf.concat([t,x], 1), self.weights_u, self.biases_u)
        d = self.neural_net(tf.concat([t,x], 1), self.weights_d, self.biases_d)
                        
        u_t = tf.gradients(u, t)[0]
        du_x = tf.gradients(d*u, x)[0]
        eq = u_t + du_x
        
        return u, d, eq
    
    def train(self, num_epochs, batch_size, learning_rate):

        for epoch in range(num_epochs):
            
            N = self.t.shape[0]
            perm = np.random.permutation(N)
            
            start_time = time.time()
            for it in range(0, N, batch_size):
                idx = perm[np.arange(it,it+batch_size)]
                (t_batch,
                 x_batch,
                 u_batch) = (self.t[idx,:],
                             self.x[idx,:],
                             self.u[idx,:])
                
                tf_dict = {self.t_tf: t_batch, self.x_tf: x_batch,
                           self.u_tf: u_batch, self.learning_rate: learning_rate}
                
                self.sess.run(self.train_op, tf_dict)
                
                # Print
                if it % (10*batch_size) == 0:
                    elapsed = time.time() - start_time
                    loss_value, learning_rate_value = self.sess.run([self.loss, self.learning_rate], tf_dict)
                    print('Epoch: %d, It: %d, Loss: %.3e, Time: %.2f, Learning Rate: %.3e' % 
                          (epoch, it, loss_value, elapsed, learning_rate_value))
#                    loss_value = self.sess.run(self.loss, tf_dict)
#                    print('Epoch: %d, It: %d, Loss: %.3e, Time: %.2f'
#                          %(epoch, it/batch_size, loss_value, elapsed))
                    start_time = time.time()
    
    def predict(self, t_star, x_star):
        
        tf_dict = {self.t_tf: t_star, self.x_tf: x_star}
        
        u_star = self.sess.run(self.u_pred, tf_dict)
        d_star = self.sess.run(self.d_pred, tf_dict)
        
        return u_star, d_star
    
def plot_solution(x_star, y_star, u_star, ax):
    
    nn = 200
    x = np.linspace(x_star.min(), x_star.max(), nn)
    y = np.linspace(y_star.min(), y_star.max(), nn)
    X, Y = np.meshgrid(x,y)
    
    X_star = np.concatenate((x_star, y_star), axis=1)
    
    U_star = griddata(X_star, u_star.flatten(), (X, Y), method='linear')
    
    # h = ax.pcolor(X,Y,U_star, cmap = 'jet')
    
    h = ax.imshow(U_star, interpolation='nearest', cmap='rainbow', 
                  extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
                  origin='lower', aspect='auto')
    
    return h

if __name__ == "__main__": 
    
    N_train = 20000

    layers_u = [2] + 10*[64] + [1]
    layers_d = [2] + 10*[64] + [1]
    
    # Load Data
    data = scipy.io.loadmat('./turbulence.mat')
        
    # Load Data
    T_star = data['T']
    X_star = data['X']
    U_star = data['P']
    D_star = data['D']
    
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    ax.plot_surface(T_star,X_star,U_star)
#    ax.set_xlabel('$t$')
#    ax.set_ylabel('$x$')
#    ax.set_zlabel('$P_1(t,x)$')
#    ax.axis('tight')
#    
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    ax.plot_surface(T_star,X_star,D_star)
#    ax.set_xlabel('$t$')
#    ax.set_ylabel('$x$')
#    ax.set_zlabel('$D(t,x)$')
#    ax.axis('tight')

    T = T_star.shape[1]
    N = T_star.shape[0]
    
    t = T_star.flatten()[:,None] # NT x 1    
    x = X_star.flatten()[:,None] # NT x 1
    u = U_star.flatten()[:,None] # NT x 1    
    d = D_star.flatten()[:,None] # NT x 1
        
    ######################################################################
    ######################## Noiseles Data ###############################
    ######################################################################
    # Training Data
    idx = np.random.choice(N*T, N_train, replace=False)
    t_train = t[idx,:]
    x_train = x[idx,:]
    u_train = u[idx,:]
    d_train = d[idx,:]
    
    # Training
    model = PhysicsInformedNN(t_train, x_train, u_train,
                              layers_u, layers_d)
    
    model.train(num_epochs = 2*10**4, batch_size = 20000, learning_rate=1e-3)
    model.train(num_epochs = 3*10**4, batch_size = 20000, learning_rate=1e-4)
    model.train(num_epochs = 3*10**4, batch_size = 20000, learning_rate=1e-5)
    model.train(num_epochs = 2*10**4, batch_size = 20000, learning_rate=1e-6)
    
    model.train(num_epochs = 2*10**4, batch_size = 20000, learning_rate=1e-6)
    model.train(num_epochs = 8*10**4, batch_size = 20000, learning_rate=1e-6)
        
    # Prediction
    u_pred, d_pred = model.predict(t, x)
    
    # Error
    error_u = np.linalg.norm(u-u_pred,2)/np.linalg.norm(u,2)
    error_d = np.linalg.norm(d-d_pred,2)/np.linalg.norm(d,2)
    
    print('Error p: %e' % (error_u))
    print('Error d: %e' % (error_d))
        
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################
    
    fig, ax = newfig(1.0, 1.2)
    ax.axis('off')

    gs = gridspec.GridSpec(2, 2)
    gs.update(top=0.9, bottom=0.1, left=0.1, right=0.9, wspace=0.5, hspace=0.7)
        
    ########      Exact p(t,x)     ###########     
    ax = plt.subplot(gs[0:1, 0])
    h = plot_solution(t,x,u,ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('Exact $P_1(t,x)$', fontsize = 10)
    
    ########     Learned p(t,x)     ###########
    ax = plt.subplot(gs[0:1, 1])
    h = plot_solution(t,x,u_pred,ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('Learned $P_1(t,x)$', fontsize = 10)
    
    ########      Exact d(t,x,y)     ###########     
    ax = plt.subplot(gs[1:2, 0])
    h = plot_solution(t,x,d,ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('Exact $D(t,x)$', fontsize = 10)

    ########     Learned d(t,x,y)     ###########
    ax = plt.subplot(gs[1:2, 1])
    h = plot_solution(t,x,d_pred,ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('Learned $D(t,x)$', fontsize = 10)
    
    savefig('./figures/Results_1D', crop = False)
    
    scipy.io.savemat('turbulence_results_%s.mat' %(time.strftime('%d_%m_%Y')),
                     {'t':t, 'x':x, 'u':u, 'd':d, 'u_pred':u_pred, 'd_pred':d_pred})
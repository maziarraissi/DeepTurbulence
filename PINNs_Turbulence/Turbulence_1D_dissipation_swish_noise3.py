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
    def __init__(self, t, x, u, e,
                 layers_ue):
        
        tx = np.concatenate([t, x], 1)
        self.tx_min = tx.min(0)
        self.tx_max = tx.max(0)
        
        ue = np.concatenate([u, e], 1)
        self.ue_mean = ue.mean(0)
        self.ue_std = ue.std(0)
        
        # data (inside the domain)
        self.t = t
        self.x = x
        self.u = u
        
        # layers
        self.layers_ue = layers_ue
        
        # initialize NN
        self.weights_ue, self.biases_ue = self.initialize_NN(layers_ue)
        
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
         self.e_pred,
         self.eq_pred) = self.net_u(self.t_tf, self.x_tf)
        
        # loss
        self.loss = tf.reduce_sum(tf.square((self.u_tf - self.u_pred)/self.ue_std[0])) + \
                    tf.reduce_sum(tf.square(self.eq_pred/self.ue_std[0]))
        
        self.loss_mean = tf.reduce_mean(tf.square((self.u_tf - self.u_pred)/self.ue_std[0])) + \
                         tf.reduce_mean(tf.square(self.eq_pred/self.ue_std[0]))
        
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
        xavier_stddev = np.sqrt(2.0/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, H, weights, biases):
        num_layers = len(weights) + 1
        
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.matmul(H, W) + b
            H = H*tf.sigmoid(H)
        W = weights[-1]
        b = biases[-1]
        H = tf.matmul(H, W) + b
        return H
            
    def net_u(self, t, x):
        
        tx = tf.concat([t,x], 1)
        tx = 2.0*(tx - self.tx_min)/(self.tx_max - self.tx_min) - 1.0
        
        ue = self.neural_net(tx, self.weights_ue, self.biases_ue)
        
        ue = self.ue_mean + ue*self.ue_std
        
        u = ue[:,0:1]
        e = ue[:,1:2]
        
        u_t = tf.gradients(u, t)[0]
        eu_x = tf.gradients(e*u, x)[0]
        eu_xx = tf.gradients(eu_x, x)[0]
        eq = u_t + eu_xx
        
        return u, e, eq
    
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
                    loss_value, learning_rate_value = self.sess.run([self.loss_mean, self.learning_rate], tf_dict)
                    print('Epoch: %d, It: %d, Loss: %.3e, Time: %.2f, Learning Rate: %.3e' % 
                          (epoch, it, loss_value, elapsed, learning_rate_value))
                    start_time = time.time()
    
    def predict(self, t_star, x_star):
        
        tf_dict = {self.t_tf: t_star, self.x_tf: x_star}
        
        u_star = self.sess.run(self.u_pred, tf_dict)
        e_star = self.sess.run(self.e_pred, tf_dict)
        
        return u_star, e_star
    
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
    
    noise = 0.1
    
    N_train = 20000

    layers_ue = [2] + 10*[2*50] + [2]
    
    # Load Data
    data = scipy.io.loadmat('./turbulence.mat')
    
    # Load Data
    T_star = data['T']
    X_star = data['X']
    U_star = data['P']
    E_star = data['E']
        
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
#    ax.plot_surface(T_star,X_star,E_star)
#    ax.set_xlabel('$t$')
#    ax.set_ylabel('$x$')
#    ax.set_zlabel('$E(t,x)$')
#    ax.axis('tight')

    T = T_star.shape[1]
    N = T_star.shape[0]
    
    t = T_star.flatten()[:,None] # NT x 1    
    x = X_star.flatten()[:,None] # NT x 1
    u = U_star.flatten()[:,None] # NT x 1    
    e = E_star.flatten()[:,None] # NT x 1
        
    ######################################################################
    ######################## Noiseles Data ###############################
    ######################################################################
    # Training Data
    idx = np.random.choice(N*T, N_train, replace=False)
    t_train = t[idx,:]
    x_train = x[idx,:]
    u_train = u[idx,:]
    e_train = e[idx,:]
    
    # add noise
    u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
    
    # Training
    model = PhysicsInformedNN(t_train, x_train, u_train, e_train,
                              layers_ue)
    
    model.train(num_epochs = 1*10**5, batch_size = N_train, learning_rate=1e-3)
    model.train(num_epochs = 2*10**5, batch_size = N_train, learning_rate=1e-4)
    model.train(num_epochs = 3*10**5, batch_size = N_train, learning_rate=1e-5)
    model.train(num_epochs = 4*10**5, batch_size = N_train, learning_rate=1e-6)

#    loss_reg = tf.reduce_sum(tf.square((model.u_tf - model.u_pred)/model.ue_std[0]))    
#    loss_eq = tf.reduce_sum(tf.square(model.eq_pred/model.ue_std[0]))
#    tf_dict = {model.t_tf: model.t, model.x_tf: model.x, model.u_tf: model.u}
#    print(model.sess.run(loss_reg, tf_dict))
#    print(model.sess.run(loss_eq, tf_dict))
    
    # Prediction
    u_pred, e_pred = model.predict(t, x)
    
    # Error
    error_u = np.linalg.norm(u-u_pred,2)/np.linalg.norm(u,2)
    error_e = np.linalg.norm(e-e_pred,2)/np.linalg.norm(e,2)
    
    print('Error p: %e' % (error_u))
    print('Error e: %e' % (error_e))
        
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
    ax.set_title('Exact $P(t,x)$', fontsize = 10)
    
    ########     Learned p(t,x)     ###########
    ax = plt.subplot(gs[0:1, 1])
    h = plot_solution(t,x,u_pred,ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('Learned $P(t,x)$', fontsize = 10)
    
    ########      Exact e(t,x,y)     ###########     
    ax = plt.subplot(gs[1:2, 0])
    h = plot_solution(t,x,e,ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('Exact $\\varepsilon(t,x)$', fontsize = 10)

    ########     Learned e(t,x,y)     ###########
    ax = plt.subplot(gs[1:2, 1])
    h = plot_solution(t,x,e_pred,ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('Learned $\\varepsilon(t,x)$', fontsize = 10)
    
    savefig('./figures/turbulence_1D_dissipation', crop = False)
    
    scipy.io.savemat('turbulence_1D_dissipation_swish_noise3_results_%s.mat' %(time.strftime('%d_%m_%Y')),
                     {'t':t, 'x':x, 'u':u, 'e':e, 'u_pred':u_pred, 'e_pred':e_pred})
    
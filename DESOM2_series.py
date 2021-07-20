
print('\n\n running DESOM2_series')
import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, LeakyReLU
from keras.models import Model
from keras import optimizers
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from DESOM2_architectures import *

parser = argparse.ArgumentParser()
parser.add_argument('--save_path',type=str)
parser.add_argument('--gamma', default=8e-6, type=float)
parser.add_argument('--map_size', nargs='+', default=[10, 10], type=int)
parser.add_argument('--iterations', default=40_000, type=int)
parser.add_argument('--som_iterations', default=35_000, type=int)
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--Tmax', default=15, type=float)
parser.add_argument('--Tmin', default=0.5, type=float)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--seed',type=int)
parser.add_argument('--latent_width', type=int)
args = parser.parse_args()



# -------------------------------- Log into Job Report ----------------------
print('map_size:',args.map_size)
print('gamma:',args.gamma)
print('lr:',args.lr)
print('iterations:',args.iterations)
print('som_iteration:',args.som_iterations)
print('batch_size:',args.batch_size)
print('T: '+str(args.Tmax)+' -> '+str(args.Tmin))
print('seed:',args.seed)
print('save_path:',args.save_path)
print('latent_width:',args.latent_width)

np.random.seed(args.seed)
tf.random.set_seed(args.seed)



#----------------------------------- Prepare Data (Fingerprints) ------------------------
#------ Load Fingerprints ---------
X_table = Table.read('RF_fingerprint.fits')
X_data = X_table['train_fp'] + X_table['test_fp'] # shape should be (4609,225)

#------ calculate 99.9-percentile -----------
percentile_vector = np.zeros(225)
for i in range(225):
    distribution = X_data[:,i]
    percentile_vector[i] = np.percentile(distribution, 99.9)

#------ apply clipping -------
new_FPs = np.zeros((4609, 225))
for i in range(225):
    new_FPs[:,i] = np.clip(X_data[:,i], 0, percentile_vector[i])

#-------- normalize along node, then along fingerprint --------
z2_data = MinMaxScaler().fit_transform(new_FPs)
z3_data = MinMaxScaler().fit_transform(z2_data.T).T
X_data = z3_data.reshape(-1,15,15,1)

# -------- SPLIT FPs -----------
X_train, x_temp = train_test_split(X_data, train_size=0.9) #80% for training set
X_test, val_set = train_test_split(x_temp, train_size=0.5) #10% for test, 10% for validation



#-----------------------------------Train Model ---------------------------
som = DESOM(input_dims = X_data.shape[1], map_size = args.map_size, latent_dims = args.latent_width)
optimizer = optimizers.Adam(args.lr)
som.initialize()
som.compile(args.gamma, optimizer)
som.init_som_weights(X_train)
som.fit(X_train,
        val_set,
        iterations = args.iterations,
        som_iterations = args.som_iterations,
        batch_size = args.batch_size,
        Tmax= args.Tmax, Tmin= args.Tmin,
        save_path='results/desom2/'+args.save_path+'/')

ae_recon, bmu = som.model.predict(X_test)
bmu = np.argmin(bmu, axis=1)

#-------------------------------- Save Plots -------------------------
#-------- Save LR plot -----------
training_history = np.genfromtxt('results/desom2/{}/desom2_log_{}.csv'.format(args.save_path, args.latent_width), delimiter=',')

# Save LR plots
fig,ax=plt.subplots(2,1,sharex=True)
ax[0].semilogy(training_history[:,3],label='loss') #LR for AE
ax[0].semilogy(training_history[:,8],label='val_loss') #LR for AE
ax[1].semilogy(training_history[:,4],label='loss') #LR for SOM 
ax[1].semilogy(training_history[:,9],label='val_loss') #LR for SOM 
ax[1].set_xlabel('iterations')
ax[1].legend()
plt.savefig('results/desom2/{}/z{}_LR.png'.format(args.save_path, args.latent_width))
plt.clf()

#--------- Save a Reconstruction example ---------
plt.figure()
plt.plot(X_test[0].reshape(X_data.shape[1]), label='input')
plt.plot(ae_recon[0].reshape(X_data.shape[1]), label='recon')
plt.title('Model Reconstruction')
plt.legend()
plt.savefig('results/desom2/{}/z{}_reconstruction_example.png'.format(args.save_path, args.latent_width))
plt.clf()

# ------- SAVE DECODED PROTOTYPES -------------
decoded_prototypes = som.decode(som.prototypes)
fig, ax = plt.subplots(args.map_size[0], args.map_size[1], figsize=(10,10))
for k in range(args.map_size[0] * args.map_size[1]):
    x = decoded_prototypes[k]
    ax[k // args.map_size[1]][k % args.map_size[1]].imshow(x.reshape(15, 15))
    ax[k // args.map_size[1]][k % args.map_size[1]].axis('off')
plt.subplots_adjust(hspace=1.05, wspace=1.05)
plt.savefig('results/desom2/{}/z{}_prototypes.png'.format(args.save_path, args.latent_width))
plt.clf()


#-------- Calculate (and log) MSE for training and test sets from AE --------------
z = mean_squared_error(X_test.reshape(-1, X_data.shape[1]), ae.reshape(-1, X_data.shape[1]))

# Log
print('MSE (test set) {}. latent: {}'.format(z, args.latent_width))
print('MSE (train set):{}'.format(training_history[3][-1]))

som.model.summary()

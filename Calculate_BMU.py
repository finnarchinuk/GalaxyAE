''' Loads a model and calculates best-matching-unit (where on the SOM the spectrum lands)
Also, calculates a correction for log_MS_dens to use for sSFR
(log_MS_dens is how much mass is in the region,
sSFR is how much star formation there is, normalized by how much mass there is)
'''

print('\n\n running Calculate_BMU.py')
import time
import matplotlib.pyplot as plt
import numpy as np
from keras import Model
from astropy.io import fits
from astropy.table import Table, Column
from astropy.cosmology import FlatLambdaCDM

from DESOM_streamlined import DESOM
from Utils import normalize_4050_region

SUFFIX = 'june11b' #which dataset to use?
UPDATED_SUFFIX = 'june11bmu' #where to save these modified dataset


#----------------------------- Model Path -----------------------------------
MODEL, map_size = 'june12a', (15,15)
model_path = 'results/desom1/'+MODEL+'/DESOM_model_final.h5'

#-------------------------- Load the DESOM model ----------------------------
def load_desom_trained(map_size, input_dims, model_path, latent):
  som = DESOM(input_dims= input_dims, map_size = map_size, latent=latent)
  som.initialize()
  som.load_weights(model_path)
  return som

som = load_desom_trained(map_size, 4544, model_path, latent=256)


def calc_bmu(data, spec, som):
  ''' expects data, spectra, som model
      calculates bmu of spectra on model
      returns data'''
  _, tmp_bmus = som.model.predict(spec)
  tmp_bmus2 = np.argmin(tmp_bmus, axis=1)
  data['bmu'] = tmp_bmus2

def calc_sSFR(data):
  ''' takes data
      fixes MSdens, adds sSFR
      returns data'''
  cosmod = FlatLambdaCDM(name='Concordance', H0=70.0, Om0=0.3,
                         Tcmb0=2.725, Neff=3.04, Ob0=0.0463)
  pixscale = 0.000138889 * 3600. # arcsec (seems to be same for all cubes)
  pixscale = pixscale/3600. * np.pi/180. # radians
  comd_bax = cosmod.comoving_distance(data['redshift']).value * 1.0e3 #comoving dist in kpc
  spaxsize_bax = pixscale * comd_bax/(1.0 + data['redshift']) # kpc
  spaxsize_bax = spaxsize_bax**2. # kpc^2
  data['logMS_dens_kpc'] = data['logMs_dens'] - np.log10(spaxsize_bax)  # solar mass per kpc^2
  data['sSFR'] = data['logSFR_dens'] - data['logMS_dens_kpc']



# --------- load train_data -------------
# update datatable, test_data, and unused_data (cause why not)
x_train = Table.read('x_train1m_' + SUFFIX + '.fits')
y_train = Table.read('y_train1m_' + SUFFIX + '.fits')
w_train = Table.read('w_train1m_' + SUFFIX + '.fits')

normalize_4050_region(x_train, y_train, w_train)
calc_bmu(y_train,
         x_train['raw'],
         som)
calc_sSFR(y_train)
y_train.write('y_train1m_' + UPDATED_SUFFIX + '.fits')
del x_train, y_train, w_train



# ---------- load test_data ----------
x_test = Table.read('x_test1m_' + SUFFIX + '.fits')
y_test = Table.read('y_test1m_' + SUFFIX + '.fits')
w_test = Table.read('w_test1m_' + SUFFIX + '.fits')

normalize_4050_region(x_test, y_test, w_test)
calc_bmu(y_test,
         x_test['raw'],
         som)
calc_sSFR(y_test)
y_test.write('y_test1m_' + UPDATED_SUFFIX + '.fits')
del x_test, y_test



# ------------ load unused_data ------------
# (more for interest than anything)
x_unused = Table.read('x_unused1m_' + SUFFIX + '.fits')
y_unused = Table.read('y_unused1m_' + SUFFIX + '.fits')
w_unused = Table.read('w_unused1m_' + SUFFIX + '.fits')

normalize_4050_region(x_unused, y_unused, w_unused)
calc_bmu(y_unused,
         x_unused['raw'],
         som)
calc_sSFR(y_unused)
y_unused.write('y_unused1m_' + UPDATED_SUFFIX + '.fits')



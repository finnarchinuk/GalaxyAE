'''
runs each galaxy through a DESOM model, determines where each spectrum lands
also adds on a morphology for that galaxy.
'''


print('\n\n generating fingerprints')

from astropy.table import Table, vstack
import numpy as np


# load data and tables
SUFFIX = 'DATE'
MAP_SIZE = 15 * 15
my_train = Table.read('y_train1m_' + SUFFIX + '.fits')
my_test = Table.read('y_test1m_' + SUFFIX + '.fits')
my_unused = Table.read('y_unused1m_' + SUFFIX + '.fits')


unique_ids = np.unique(vstack([my_train, my_test, my_unused])['mangaID'])

train_fp = np.zeros((len(unique_ids), MAP_SIZE))
test_fp = np.zeros((len(unique_ids), MAP_SIZE))
unused_fp = np.zeros((len(unique_ids), MAP_SIZE))


# generate FPs for each galaxy
for gal_index, gal_id in enumerate(unique_ids):
  # select the galaxy by 'mangaID'
  mask_train = (my_train['mangaID'] == gal_id)
  mask_test = (my_test['mangaID'] == gal_id)
  mask_unused = (my_unused['mangaID'] == gal_id)

  # select 'bmu' for that galaxy
  temp_train = my_train[mask_train]['bmu']
  temp_test = my_test[mask_test]['bmu']
  temp_unused = my_unused[mask_unused]['bmu']

  # fill FP
  for BMU in range(MAP_SIZE):
    bmu_mask_train = (temp_train == BMU)
    train_fp[gal_index, BMU] = bmu_mask_train.sum()
    #
    bmu_mask_test = (temp_test == BMU)
    test_fp[gal_index, BMU] = bmu_mask_test.sum()
    #
    bmu_mask_unused = (temp_unused == BMU)
    unused_fp[gal_index, BMU] = bmu_mask_unused.sum()

    
# calculate average age of spectra in galaxy
# (this is used to help select an average galaxy later for plotting)
av_age = np.zeros(len(unique_ids))
stack_data = vstack([my_train, my_test]) #only looking at used spectra

for gal_index, gal_id in enumerate(unique_ids):
    mask = (stack_data['mangaID'] == gal_id)
    av_age[gal_index] = np.mean(stack_data[mask]['logage'])


# define table for fingerprints
FP_table = Table(names = ['mangaID', 'train_fp', 'test_fp', 'unused_fp', 'av_age'],
                 data = [unique_ids, train_fp, test_fp, unused_fp, av_age])


# save fingerprints
FP_table.write('RF_fingerprint.fits')


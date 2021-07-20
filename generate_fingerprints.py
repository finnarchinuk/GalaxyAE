'''
runs each galaxy through a DESOM model, determines where each spectrum lands
also adds on a morphology for that galaxy.
'''


print('\n\n generating fingerprints')

from astropy.table import Table, vstack
import numpy as np


# load data and tables
SUFFIX = 'june11bmu'
MAP_SIZE = 15 * 15

vac_table = Table.read('full_vac.fits') # generated from a value added catalogue (see morpho_table.py)
my_train = Table.read('y_train1m_' + SUFFIX + '.fits')
my_test = Table.read('y_test1m_' + SUFFIX + '.fits')
my_unused = Table.read('y_unused1m_' + SUFFIX + '.fits')


unique_ids = np.unique(vstack([my_train, my_test, my_unused])['mangaID'])

train_fp = np.zeros((len(unique_ids), MAP_SIZE))
test_fp = np.zeros((len(unique_ids), MAP_SIZE))
unused_fp = np.zeros((len(unique_ids), MAP_SIZE))

# generate FPs for each galaxy
for gal_index, gal_id in enumerate(unique_ids):
  mask_train = (my_train['mangaID'] == gal_id)
  mask_test = (my_test['mangaID'] == gal_id)
  mask_unused = (my_unused['mangaID'] == gal_id)
  #
  temp_train = my_train[mask_train]['bmu']
  temp_test = my_test[mask_test]['bmu']
  temp_unused = my_unused[mask_unused]['bmu']
  #
  for BMU in range(MAP_SIZE):
    hmask_train = (temp_train == BMU)
    train_fp[gal_index, BMU] = hmask_train.sum()
    #
    hmask_test = (temp_test == BMU)
    test_fp[gal_index, BMU] = hmask_test.sum()
    #
    hmask_unused = (temp_unused == BMU)
    unused_fp[gal_index, BMU] = hmask_unused.sum()


  
# define table for fingerprints
FP_table = Table(names = ['mangaID', 'train_fp', 'test_fp', 'unused_fp'],
                 data = [unique_ids, train_fp, test_fp, unused_fp])
                 

# Add galaxy parameters to fingerprint table
for column_name in vac_table.colnames:
    new_column = list()
    for i in range(len(FP_table)):
        mask = (FP_table['mangaID'][i] == vac_table['mangaID'])
        new_column.append(vac_table[k][mask][0])
    temp_column = Column(name=column_name, data=np.array(new_column))
    FP_table.add_column(temp_column)


# save fingerprints
FP_table.write('RF_fingerprint.fits')


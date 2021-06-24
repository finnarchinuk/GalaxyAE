'''
This is a small script that pulls galaxy morphology from a value added catalogue
https://www.sdss.org/dr16/data_access/value-added-catalogs/?vac_id=manga-visual-morphologies-from-sdss-and-desi-images

This saves a table with three columns: ['mangaID','type','suffix']
This table ends up having 4696 galaxies, though some are blanks.
'''

from astropy.io import fits
from astropy.table import Table, Column
import numpy as np

file_path = 'SOME_PATH/manga_visual_morpho-1.0.1.fits'
catalogue = fits.open(file_path)


# generate simple table
simple_table = Table([catalogue[1].data.field('MANGAID'),
                     catalogue[1].data.field('Type')],
                     names = ('mangaID','type'))


# extract suffix from identifier
zoo_gal_suffixes = np.zeros(len(simple_table),dtype=int)
for i in range(len(simple_table)):
    try:
        zoo_gal_suffixes[i] = int(simple_table['mangaID'][i].split('-')[1])
    except:
        pass

temp_column = Column(name='suffix',data=(zoo_gal_suffixes))
simple_table.add_column(temp_column)

# save it.
simple_table.write('gal_morph.fits')

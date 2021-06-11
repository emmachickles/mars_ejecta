path='./unseen_data/'
model='vgg19_model.h5'
model_name='vgg19'

import os
import numpy as np
import fnmatch
from model import *
from tensorflow.keras.models import load_model

model = load_model(model)

fnames = fnmatch.filter(os.listdir(path), '*x.npy')
fnames = np.core.defchararray.strip(fnames, chars='x.npy')

for fname in fnames:
    print(fname)
    ids = np.loadtxt(path+fname+'fnames.txt', dtype='str')

    x = np.load(path+fname+'x.npy')
    x = normalize(x)
    x = np.expand_dims(x, -1)
    x = np.repeat(x, 3, axis=-1)
    x = preprocess_input(x)

    y = model.predict(x)
    y = y[:,1]
    np.savetxt(path+fname+'y.txt', y, delimiter=',')
    n = len(ids)
    print('Total imgs: '+str(n))
    print('Ejecta: '+str(np.count_nonzero(y))+'/'+str(n))
    

    
    



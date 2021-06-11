# =============================================================================
#
# 201127 - crater_finder_image.png
# Finds crater center and radius in remote sensing images
# / Emma Chickles
# 
# =============================================================================

import numpy as np
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from skimage.feature import canny
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage import color
from skimage.filters import gaussian
import tensorflow as tf

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

from tensorflow.keras.layers import *
from tensorflow.keras import regularizers
from tensorflow.keras import activations
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.applications.vgg19 import VGG19 as model_pretrained
from tensorflow.keras.applications.vgg19 import preprocess_input
from evaluation import *
from model import *
import os

# data_dir = './data_orig/'
data_dir = './data_gabor/gabor-'
output_dir = './plots210315/'

width, height= 224, 224
crop = 40


# >> what to run
predict_hough = False # >> predict crater location using Hough transform
eval_hough = False # >> manually validate crater location
train_cnn = True # >> use CNN 

# >> other options
detect_edges = True # >> calculate canny edge detector before Hough transform
blur_sigma=1
num_peaks = 1
hough_radii = np.arange(224./10, 224./6, 2)
# sigma=2
# low_threshold=10
# high_threshold=20
sigma=2
low_threshold=None
high_threshold=None

preprocess_balance_train = True
preprocess_balance_val = True

vgg=True

if vgg:
    p = {'lr': 5e-6, 'activation': 'elu', 'num_dense': 0,
         'kernel_initializer': 'glorot_normal', 'batch_size': 32, 'epochs': 30}
    
else:
    p = {'num_conv_blocks': 1,
     'dropout': 0.2,
     'kernel_size': 3,
     'activation': 'elu',
     'num_dense': 2,
     'dense_units': 512,
     'lr': 0.00001,
     'epochs': 30,
     'batch_size': 16,
     'kernel_initializer': 'glorot_normal',
     'num_consecutive': 2,
     'latent_dim': 30,
     'l1': 0.0,
     'l2': 0.0,
     'num_filters': [16, 16, 32, 32, 64], 'batch_norm': 1}

# :: load data ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

x_train = np.load(data_dir + 'x_train.npy')
x_val = np.load(data_dir + 'x_val.npy')
y_train = np.loadtxt(data_dir + 'y_train.txt')
y_val = np.loadtxt(data_dir + 'y_val.txt')

if preprocess_balance_train:
    print('Creating a balanced training set ...')
    num_ejecta = np.count_nonzero(y_train)
    inds = np.nonzero(y_train == 0.)[0][num_ejecta:]
    y_train = np.delete(y_train, inds)
    x_train = np.delete(x_train, inds, axis=0)
    
if preprocess_balance_val:
    print('Creating a balanced validation set ...')
    num_ejecta = np.count_nonzero(y_val)
    inds = np.nonzero(y_val == 0.)[0][num_ejecta:]
    y_val = np.delete(y_val, inds)
    x_val = np.delete(x_val, inds, axis=0)
    
# :: some functions :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def canny_experiments(img, sigma=[1,2,3], low_threshold=[10,30,50],
                      high_threshold=[40,60,80], blur_sigma=1):
    if blur_sigma > 0:
        img = gaussian(img, sigma=blur_sigma)
    
    fig, ax = plt.subplots(4, figsize=(5,15))
    ax[0].imshow(img, cmap='gray')
    for row in range(3):
        edges = canny(img, sigma=sigma[row])
        ax[row+1].set_title('Sigma: '+str(sigma[row]))
        ax[row+1].imshow(edges, cmap='gray')
    fig.tight_layout()
    fig.savefig(output_dir+'canny_sigma.png')
    plt.close(fig)
    
    fig, ax = plt.subplots(4, figsize=(5,15))
    ax[0].imshow(img, cmap='gray')
    for row in range(3):
        edges = canny(img, low_threshold=low_threshold[row])
        ax[row+1].set_title('Low threshold: '+str(low_threshold[row]))
        ax[row+1].imshow(edges, cmap='gray')
    fig.tight_layout()
    fig.savefig(output_dir+'canny_low.png')
    plt.close(fig)
    
    fig, ax = plt.subplots(4, figsize=(5,15))
    ax[0].imshow(img, cmap='gray')
    for row in range(3):
        edges = canny(img, high_threshold=high_threshold[row])
        ax[row+1].set_title('High threshold: '+str(high_threshold[row]))
        ax[row+1].imshow(edges, cmap='gray')
    fig.tight_layout()
    fig.savefig(output_dir+'canny_high.png')
    plt.close(fig)
# canny_experiments(x_train[0])

def hough_experiment(img, hough_radii=[224./10, 224./6], num_peaks=4,
                     sigma=2, low_threshold=None, high_threshold=None):
    edges = canny(img, sigma=sigma, low_threshold=low_threshold,
                  high_threshold=high_threshold)
    hough_res = hough_circle(edges, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=num_peaks)
    
    fig, ax = plt.subplots(ncols=2, figsize=(9,4.5))
    ax[0].imshow(img, cmap='gray')
    img = color.gray2rgb(img)
    for j in range(num_peaks):
        circy, circx = circle_perimeter(int(cy[j]), int(cx[j]),
                                int(radii[j]), shape=img.shape)
        if j == 0:
            img[circy, circx] = (220, 20, 20)
        else:
            img[circy, circx] = (20, 20, 220)
    ax[1].imshow(img, cmap='gray')
    fig.savefig(output_dir+'hough_circles.png')
    plt.close(fig)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Use Hough transform to locate craters ::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

if predict_hough:

    circ_train = np.empty((len(x_train),3))
    circ_val = np.empty((len(x_val),3))
    
    fail_train = []
    fail_val = []
    
    for i in range(len(x_train)+len(x_val)):
        if i < 30:
            debug=True
        else:
            debug=False
    
        if i < len(x_train):
            print('Training set '+str(i)+'/'+str(len(x_train)))
            img = x_train[i]
        else:
            print('Validation set '+str(i-len(x_train))+'/'+str(len(x_val)))
            img = x_val[i-len(x_train)]
        
        if debug:
            fig, ax = plt.subplots(nrows=3, ncols=1)
            ax[0].imshow(img, cmap='gray')
            
        # img = gaussian(img, sigma=blur_sigma)
        # if debug:
        #     ax[1].imshow(img, cmap='gray')
            
        if detect_edges:
            edges = canny(img, sigma=sigma, low_threshold=low_threshold,
                          high_threshold=high_threshold)
            if debug: ax[1].imshow(edges, cmap='gray')
            hough_res = hough_circle(edges, hough_radii)
            
        else:
            hough_res = hough_circle(img, hough_radii)
            
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                                   total_num_peaks=num_peaks)
        
        if debug:
            img = color.gray2rgb(img)
            for j in range(num_peaks):
                if len(cy) > 0:
                    circy, circx = circle_perimeter(int(cy[j]), int(cx[j]),
                                                    int(radii[j]), shape=img.shape)
                    img[circy, circx] = (220, 20, 20)
            ax[2].imshow(img, cmap='gray')
            fig.savefig(output_dir+'hough_example'+str(i)+'.png')
            plt.close(fig)
            
        if len(cy) == 0:
            if i < len(x_train):
                fail_train.append(i)
            else:
                fail_val.append(i-len(x_train))
            cy, cx, radii = [0], [0], [0]
        if i < len(x_train):
            circ_train[i] = np.array([cx[0], cy[0], radii[0]])
        else:
            circ_val[i-len(x_train)] = np.array([cx[0], cy[0], radii[0]])
            
    np.save(output_dir+'circ_train.npy', circ_train)
    np.save(output_dir+'circ_val.npy', circ_val)
    np.save(output_dir+'circ_fail_train.npy', np.array(fail_train))
    np.save(output_dir+'circ_fail_val.npy', np.array(fail_val))

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: evaluate Hough transform predictions :::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
if eval_hough:
    plt.ion()
    
    circ_train = np.load(output_dir+'circ_train.npy')
    circ_val = np.load(output_dir+'circ_val.npy')
    fail_train = np.load(output_dir+'circ_fail_train.npy')
    fail_val = np.load(output_dir+'circ_fail_val.npy')
    

    
    if os.path.exists(output_dir+'circ_success_train.txt'):
        success_train = np.loadtxt(output_dir+'circ_success_train.txt',
                                   dtype='int')
        success_train = list(success_train)
        success_val = np.loadtxt(output_dir+'circ_success_val.txt', dtype='int')
        success_val = list(success_val)
        start = max(success_train)
    else:
        success_train = []
        success_val = []
        start=0        
        
    
    # >> look at each picture
    for i in range(start,len(x_train)+len(x_val)):
        if i<len(x_train) and i in fail_train or \
            i>len(x_train) and i-len(x_train) in fail_val:
                pass
        else:
            if i < len(x_train):
                print('Training set '+str(i)+'/'+str(len(x_train)))
                img = x_train[i]
                cx, cy, rad = circ_train[i]
            else:
                print('Validation set '+str(i-len(x_train))+'/'+str(len(x_val)))
                img = x_val[i-len(x_train)]
                cx, cy, rad = circ_val[i-len(x_train)]
            img = color.gray2rgb(img)
            circy, circx = circle_perimeter(int(cy), int(cx), int(rad),
                                            shape=img.shape)
            img[circy, circx] = (220, 20, 20)
            
            plt.figure()
            plt.imshow(img, cmap='gray')
            
            # >> if success the left click, if not then right click
            res = plt.ginput(n=1, mouse_add=1, mouse_stop=2, timeout=400)
            print(res)
            plt.close()
            
            if len(res) > 0:
                if i < len(x_train):
                    if i not in success_train:
                        success_train.append(i)
                        with open(output_dir+'circ_success_train.txt', 'a') as f:
                            f.write(str(i)+'\n')
                else:
                    if i-len(x_train) not in success_val:
                        success_val.append(i-len(x_train))
                        with open(output_dir+'circ_success_val.txt', 'a') as f:
                            f.write(str(i-len(x_train))+'\n')
                
    # np.save(output_dir+'circ_success_train.npy', np.array(success_train))
    # np.save(output_dir+'circ_success_val.npy', np.array(success_val))     
     
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: train CNN to locate crater :::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
if train_cnn:
    
    # -- some preprocessing ---------------------------------------------------
    
    success_train = np.loadtxt(output_dir+'circ_success_train.txt', dtype='int')
    success_val = np.loadtxt(output_dir+'circ_success_val.txt', dtype='int')
    circ_train = np.load(output_dir+'circ_train.npy')
    circ_val = np.load(output_dir+'circ_val.npy')
    
    x_train = x_train[success_train]
    x_val = x_val[success_val]
    y_train = circ_train[success_train]
    y_val = circ_val[success_val]
    
    # x_train = x_train.astype('float16')
    # x_train = x_train.astype('float16')
    
    if crop > 0:
        x_train = x_train[:,crop:-1*crop,crop:-1*crop]
        x_val = x_val[:,crop:-1*crop,crop:-1*crop]
        y_train[:,:2] = y_train[:,:2] - crop
        y_val[:,:2] = y_val[:,:2] - crop
        width = width - 2*crop
        height = height - 2*crop
    
    x_train = np.expand_dims(x_train, -1)
    x_val=  np.expand_dims(x_val, -1)
    
    # -- initialize model -----------------------------------------------------
    
    if vgg:
        x_train = np.repeat(x_train, 3, axis=-1)
        x_train = preprocess_input(x_train) # !! don't really understand
        x_val = np.repeat(x_val, 3, axis=-1)
        x_val = preprocess_input(x_val)  
        # x_train = np.load(output_dir+'x_train_vgg19.npy')
        # x_val = np.load(output_dir+'x_val_vgg19.npy')
        
        pretrained_init = model_pretrained(weights='imagenet', include_top=False,
                                            input_shape=(width,height,3))
        input_img = Input(shape=(width, height, 3))
        for i in range(1, len(pretrained_init.layers)):
            layer = pretrained_init.layers[i]
            if i == 1:
                x = layer(input_img)
            else:
                x = layer(x)
        x = Flatten()(x)
        for i in range(p['num_dense']):
            x = Dense(4096, activation=p['activation'],
                      kernel_initializer=p['kernel_initializer'])(x)
        x = Dense(3, activation=p['activation'],
                  kernel_initializer=p['kernel_initializer'])(x)   
        model = Model(input_img, x)
        
        # model = Sequential()
        # model.add(model_pretrained(weights='imagenet', include_top=False,
        #                            input_shape=(width,height,3)))
        # model.add(Flatten())
        # model.add(Dense(4096, activation=p['activation'],
        #                 kernel_initializer=p['kernel_initializer']))
        # model.add(Dense(4096, activation=p['activation'],
        #                 kernel_initializer=p['kernel_initializer']))
        # model.add(Dense(3, activation=p['activation'],
        #                 kernel_initializer=p['kernel_initializer']))
        
    else:
        x_train = normalize(x_train)
        x_val = normalize(x_val)  
        input_img = Input(shape=(width, height, 1))
        if type(p['num_filters']) != type([]):
            p['num_filters'] = [p['num_filters']]*p['num_conv_blocks']    
        x = input_img
        for i in range(p['num_conv_blocks']):
            for j in range(p['num_consecutive']):
                x = Conv2D(p['num_filters'][i], 
                           kernel_size=(p['kernel_size'], p['kernel_size']),
                           activation=p['activation'], padding='same',
                           kernel_initializer=p['kernel_initializer'])(x) 
                if p['batch_norm'] == 1:
                    x = BatchNormalization()(x)            
                x = Activation(p['activation'])(x)
            x = Dropout(p['dropout'])(x)
            x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
        x = Flatten()(x)
        for i in range(p['num_dense']):
            x = Dense(p['dense_units'], activation=p['activation'],
                      kernel_initializer=p['kernel_initializer'])(x)
        x = Dense(3, activation=p['activation'],
                  kernel_initializer=p['kernel_initializer'])(x)
        
        model = Model(input_img, x)
    opt = optimizers.Adam(lr=p['lr'])
    model.compile(loss='mean_squared_error', optimizer=opt)
    model.summary()        
    
    # -- train model ----------------------------------------------------------
    
    history = model.fit(x_train, y_train, batch_size=p['batch_size'],
                    epochs=p['epochs'], shuffle=True,
                    validation_data=(x_val, y_val))
    
    epoch_plot(history, output_dir, 'cda-vgg19-')
    y_pred = model.predict(x_val)
    
    # -- save predicted results for failed hough ------------------------------
    
    fails_train = np.setdiff1d(np.arange(len(x_train)), success_train)
    fail_pred = model.predict(x_train[fails_train])
    
    for i in range(len(x_train)):
        if i not in success_train:
            img = np.expand_dims(x_train[i], 0)
            circ_train[i] = model.predict(img)[0]
    circ_train[:,:2] += crop
    np.save(output_dir+'circ_train_cnn.npy', circ_train)  
    print('Saved '+output_dir+'circ_train_cnn.npy')      
    
            
    fails_val = np.setdiff1d(np.arange(len(x_val)), success_val)
    fail_pred = model.predict(x_val[fails_val])
    
    for i in range(len(x_val)):
        if i not in success_val:
            img = np.expand_dims(x_val[i], 0)
            circ_val[i] = model.predict(img)[0]
    circ_val[:,:2] += crop
    np.save(output_dir+'circ_val_cnn.npy', circ_val)
    print('Saved '+output_dir+'circ_val_cnn.npy')

    # -- reload x_val data for plotting ---------------------------------------
    
    x_val = np.load(data_dir + 'x_val.npy')
    y_val = np.loadtxt(data_dir + 'y_val.txt')
    if preprocess_balance_val:
        print('Creating a balanced validation set ...')
        num_ejecta = np.count_nonzero(y_val)
        inds = np.nonzero(y_val == 0.)[0][num_ejecta:]
        y_val = np.delete(y_val, inds)
        x_val = np.delete(x_val, inds, axis=0)
    x_val = x_val[success_val]
    y_val = circ_val[success_val]
    if crop > 0:
        x_val = x_val[:,crop:-1*crop,crop:-1*crop]
        y_val[:,:2] = y_val[:,:2] - crop
    
    # -- plot predictions -----------------------------------------------------
    nrows, ncols = 3, 3
    fig, ax = plt.subplots(nrows, ncols)
    for row in range(nrows):
        for col in range(ncols):
            ind = row*nrows+col
            img=color.gray2rgb(x_val[ind])
            circy, circx = circle_perimeter(int(y_val[ind][1]), int(y_val[ind][0]),
                                            int(y_val[ind][2]), shape=img.shape)
            img[circy, circx] = (220, 20, 20)
            circy, circx = circle_perimeter(int(y_pred[ind][1]), int(y_pred[ind][0]),
                                            int(y_pred[ind][2]), shape=img.shape)
            img[circy, circx] = (20, 20, 220)
            ax[row,col].imshow(img, cmap='gray')
    fig.savefig(output_dir+'cda_results.png')
    print('Saved '+output_dir+'cda_results.png')
            
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::  
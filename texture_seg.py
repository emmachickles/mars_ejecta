# =============================================================================
#
# 20201115 - texture_seg.py
#
# =============================================================================

from evaluation import *

import matplotlib.pyplot as plt
import numpy as np
import pdb
import os

from skimage.filters import gabor
from skimage.filters import gabor_kernel
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from sklearn.manifold import TSNE
from sklearn import svm

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

from scipy import ndimage

# -- inputs -------------------------------------------------------------------

output_dir = './plots210219/'
data_dir = './data_orig/'

preprocess_balance_train = True
preprocess_balance_val = True

name = 'gabor'

gabor_features = True
# bank_theta = np.array([0, 45, 90, 135]) * np.pi / 180.
# bank_freq = [0.25, 0.5, 0.7]
bank_theta = np.linspace(0, 180, 16) * np.pi / 180.
bank_freq = np.linspace(0.25, 0.75, 5)
n_stds = 0.2

hog_features = False
pixels_per_cell=(8,8)
cells_per_block=(3,3)
orientations=4
transform_sqrt=True

lbp_features = False
vanilla = True
radius = 3
no_points = 8 * radius

height, width = 224, 224
sector_size = 16
num_sectors = int(height / sector_size)

classify_ann = False
classify_svm = True

# -----------------------------------------------------------------------------

def extract_gabor_features(images, bank_theta, bank_freq, height, sector_size):
    feat = [] # >> all feature vectors
    filt = [] # >> all filtered images
    
    num_sectors = int(height / sector_size)
    
    count=0
    for img in images:
        if count % 10 == 0:
            print('Extracting Gabor features: '+str(count)+'/'+str(len(images)))
        
        filtered_images = []
        feature_vector = []
        
        # >> apply gabor filter
        for theta in bank_theta:
            for freq in bank_freq:
                real, imag = gabor(img, freq, theta)
                filtered_images.append(real)
                 
                # >> compute variance in each sector
                for x in range(num_sectors):
                    for y in range(num_sectors):
                        sector_var = np.var(real[x*sector_size:(x+1)*sector_size,
                                                 y*sector_size:(y+1)*sector_size])
                        feature_vector.append(sector_var)
        
        feat.append(feature_vector)
        filt.append(filtered_images)    
        count += 1              
    feat = np.array(feat)    
    
    return feat, filt
  
def plot_gabor_imgs(x, filt_x, ind, bank_theta, bank_freq, output_dir='./',
                    prefix=''):
    fig, ax= plt.subplots(len(bank_theta)+1, len(bank_freq),
                          figsize=(3*len(bank_freq),3*len(bank_theta)))
    ax[0,0].set_title('Original')
    ax[0,0].imshow(x[ind], cmap='gray')
    for col in range(len(bank_freq)):
        ax[0,col].axis('off')
    for row in range(len(bank_theta)):
        for col in range(len(bank_freq)):
            ax[row+1,col].set_title('Theta: '+str(round(bank_theta[row], 3))+\
                                    '\nFrequency: '+str(bank_freq[col]))
            ax[row+1,col].imshow(filt_x[ind][row*len(bank_freq)+col],
                                 cmap='gray')
            
    for a in ax.flatten():
        a.set_xticklabels([])
        a.set_yticklabels([])
            
    fig.tight_layout()
    fig.savefig(output_dir+'gabor_'+prefix+'example.png', dpi=300)
    plt.close(fig)
    
def plot_gabor_filter_bank(bank_theta, bank_freq, output_dir, n_stds=0.5):
    '''Default n_stds=3'''
    fig, ax= plt.subplots(len(bank_theta), len(bank_freq),
                          figsize=(2*len(bank_freq),2*len(bank_theta)))
    for row in range(len(bank_theta)):
        for col in range(len(bank_freq)):
            ax[row,col].set_title('Theta: '+str(round(bank_theta[row], 3))+\
                                    '\nFrequency: '+str(bank_freq[col]))
            gk = gabor_kernel(frequency=bank_freq[col], theta=bank_theta[row],
                              n_stds=n_stds)
            ax[row,col].imshow(gk.real, cmap='gray')    
    for a in ax.flatten():
        a.set_xticklabels([])
        a.set_yticklabels([])
        
    fig.tight_layout()
    fig.savefig(output_dir+'gabor_filter_bank.png', dpi=300)
    plt.close(fig)
    
def plot_gabor_var(x, filt_x, ind, bank_theta, bank_freq, win_cols=3, win_rows=3,
                   output_dir='./'):
    fig, ax= plt.subplots(len(bank_theta)+1, len(bank_freq)*2, figsize=(15,15))
    ax[0,0].set_title('Original')
    ax[0,0].imshow(x[ind], cmap='gray')
    for col in range(len(bank_freq)*2):
        ax[0,col].axis('off')
    for row in range(len(bank_theta)):
        for col in range(len(bank_freq)):
            ax[row+1,col*2].set_title('Theta: '+str(bank_theta[row])+\
                                    '\nFrequency: '+str(bank_freq[col])+\
                                        '\nGabor filtered image')
            img = filt_x[ind][row*len(bank_freq)+col]
            ax[row+1,col*2].imshow(img, cmap='gray')
            
            # >> compute variance image
            win_mean = ndimage.uniform_filter(img, (win_rows, win_cols))
            win_sqr_mean = ndimage.uniform_filter(img**2, (win_rows, win_cols))
            win_var = win_sqr_mean - win_mean**2
            ax[row+1,col*2+1].set_title('Variance map')
            ax[row+1,col*2+1].imshow(win_var, cmap='gray')
            
    for a in ax.flatten():
        a.set_xticklabels([])
        a.set_yticklabels([])
            
    fig.tight_layout()
    fig.savefig(output_dir+'gabor_var.png', dpi=300)
    plt.close(fig)
  
def plot_hog_experiments(img, pixel_per_cell_vals=[8,16,32],
                         orientation_vals=[4,6,8], transform_sqrt=False,
                         block_norm='L2-Hys', output_dir='./'):
    '''Get some intuition about the HOG parameters.'''
    fig, ax = plt.subplots(len(pixel_per_cell_vals)+1,len(orientation_vals),
                           figsize=(15,15))
    ax[0,1].set_title('Original')
    ax[0,1].imshow(img, cmap='gray')
    ax[0,0].axis('off')
    ax[0,2].axis('off')
    
    for row in range(len(pixel_per_cell_vals)):
        for col in range(len(orientation_vals)):
            out, hog_image = hog(img, visualize=True, orientations=orientation_vals[col],
                                 pixels_per_cell=(pixel_per_cell_vals[row],
                                                  pixel_per_cell_vals[row]),
                                 cells_per_block=(1,1), transform_sqrt=transform_sqrt,
                                 block_norm=block_norm)
            ax[row+1,col].set_title('feat_vect length: '+str(len(out))+\
                                    '\npixel_per_cell: '+str(pixel_per_cell_vals[row])+\
                              '\norientation: '+str(orientation_vals[col]))

            ax[row+1,col].imshow(hog_image, cmap='gray')
    fig.tight_layout()  
    fig.savefig(output_dir+'hog_experiments.png')
 
    
 
def get_pixel(img, center, x, y): 
    '''https://www.geeksforgeeks.org/create-local-binary-pattern-of-an-image-
    using-opencv-python/'''
      
    new_value = 0
      
    try: 
        # If local neighbourhood pixel  
        # value is greater than or equal 
        # to center pixel values then  
        # set it to 1 
        if img[x][y] >= center: 
            new_value = 1
              
    except: 
        # Exception is required when  
        # neighbourhood value of a center 
        # pixel value is null i.e. values 
        # present at boundaries. 
        pass
      
    return new_value 
   
# Function for calculating LBP 
def lbp_calculated_pixel(img, x, y): 
   
    center = img[x][y] 
   
    val_ar = [] 
      
    # top_left 
    val_ar.append(get_pixel(img, center, x-1, y-1)) 
      
    # top 
    val_ar.append(get_pixel(img, center, x-1, y)) 
      
    # top_right 
    val_ar.append(get_pixel(img, center, x-1, y + 1)) 
      
    # right 
    val_ar.append(get_pixel(img, center, x, y + 1)) 
      
    # bottom_right 
    val_ar.append(get_pixel(img, center, x + 1, y + 1)) 
      
    # bottom 
    val_ar.append(get_pixel(img, center, x + 1, y)) 
      
    # bottom_left 
    val_ar.append(get_pixel(img, center, x + 1, y-1)) 
      
    # left 
    val_ar.append(get_pixel(img, center, x, y-1)) 
       
    # Now, we need to convert binary 
    # values to decimal 
    power_val = [1, 2, 4, 8, 16, 32, 64, 128] 
   
    val = 0
      
    for i in range(len(val_ar)): 
        val += val_ar[i] * power_val[i] 
          
    return val 

def vanilla_lbp(img, height=224, width=224):
    img_lbp = np.zeros((height, width), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i,j] = lbp_calculated_pixel(img, i, j)
            
    x, counts = np.unique(img_lbp.ravel(), return_counts=True)
    
    # >> make sure feature vector is (256,)
    if len(x) != 256:
        missing_bins = np.setdiff1d(np.arange(256), x)
        missing_bins = np.sort(missing_bins)
        for nbin in missing_bins:
            counts = np.insert(counts, nbin, 0)
    
    hist = counts / sum(counts)
    return img_lbp, hist

def extract_lbp_features(images, radius, no_points):
    img_lbp = local_binary_pattern(img, no_points, radius, method='uniform')
    x, counts = np.unique(img_lbp.ravel(), return_counts=True)
    
    # >> make sure feature vector is (26,)
    if len(x) != 26:
        missing_bins = np.setdiff1d(np.arange(26), x)
        missing_bins = np.sort(missing_bins)
        for nbin in missing_bins:
            counts = np.insert(counts, nbin, 0)
    
    hist = counts / sum(counts)
    return img_lbp, hist    

def plot_lbp_experiments(img, npoints=[16,24,40], radii=[3,5,7], method='uniform',
                         output_dir='./'):
    '''Get some intuition about the lbp parameters.'''
    fig, ax = plt.subplots(len(npoints)+1,len(radii), figsize=(15,15))
    ax[0,1].set_title('Original')
    ax[0,1].imshow(img, cmap='gray')
    ax[0,0].axis('off')
    ax[0,2].axis('off')
    
    for row in range(len(npoints)):
        for col in range(len(radii)):
            img_lbp = local_binary_pattern(img, npoints[row], radii[col],
                                           method=method)
            ax[row+1,col].set_title('no_points: '+str(npoints[row])+\
                                    '\nradius: '+str(radii[col]))
            ax[row+1,col].imshow(img_lbp, cmap='gray')
    fig.tight_layout()  
    fig.savefig(output_dir+'lbp_experiments.png')
    
def plot_filtered_imgs(x, inds, filt_x, output_dir='./', name='filtered'):
    
    n_cols = len(inds)
    fig, ax = plt.subplots(2,n_cols)
    for col in range(n_cols):
        ax[0,col].set_title('Original')
        ax[0,col].imshow(x[inds[col]], cmap='gray')
        ax[0,col].set_xticklabels([])
        ax[0,col].set_yticklabels([])
        
        ax[1,col].set_title(name+' image')
        ax[1,col].imshow(filt_x[inds[col]], cmap=plt.get_cmap('jet'))
        ax[1,col].set_xticklabels([])
        ax[1,col].set_yticklabels([])
    fig.tight_layout()
    fig.savefig(output_dir+name+'_example.png', dpi=300)

# -----------------------------------------------------------------------------

# >> get training and testing set
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
    
    
# plot_inds = [np.nonzero(y_train)[0][0], np.nonzero(y_train)[0][1],
#              np.nonzero(y_train==0.)[0][0]]
plot_inds = list(range(40))

x_train = x_train[np.nonzero(y_train==1.)]
y_train = y_train[np.nonzero(y_train==1.)]
x_train = x_train[:40]
y_train = y_train[:40]
x_val = x_val[:10]
y_val = y_val[:10]

# -- derive gabor features ----------------------------------------------------
if gabor_features:
    print('Generating Gabor features...')
    if os.path.exists(output_dir+'gabor_filt_train.npy'):
        feat_train = np.load(output_dir+'gabor_feat_train.npy')
        filt_train = np.load(output_dir+'gabor_filt_train.npy')
        feat_val = np.load(output_dir+'gabor_feat_val.npy')
        filt_val = np.load(output_dir+'gabor_filt_val.npy')
    else:
        feat_train, filt_train = \
            extract_gabor_features(x_train, bank_theta, bank_freq, height, sector_size)
        np.save(output_dir+'gabor_filt_train.npy', filt_train)
        np.save(output_dir+'gabor_feat_train.npy', feat_train)

        feat_val, filt_val = \
            extract_gabor_features(x_val, bank_theta, bank_freq, height, sector_size)
        np.save(output_dir+'gabor_filt_val.npy', filt_val)
        np.save(output_dir+'gabor_feat_val.npy', feat_val)
    
    plot_gabor_filter_bank(bank_theta, bank_freq, output_dir, n_stds=n_stds)
    
    for ind in plot_inds:
        plot_gabor_imgs(x_train, filt_train, ind, bank_theta, bank_freq,
                        output_dir, str(ind)+'-'+str(y_train[ind])+'-')
    
    plot_gabor_var(x_train, filt_train, plot_inds[0], bank_theta, bank_freq,
                   output_dir=output_dir)
    
    # ind = plot_inds[0]
    # fig, ax = plt.subplots(3,2)
    # ax[0,0].set_title('Original')
    # ax[0,0].imshow(x_train[ind], cmap='gray')
    # ax[1,0].set_title('Orientation: 0 deg')
    # ax[1,0].imshow(filt_train[ind][0], cmap='gray')
    # ax[1,1].set_title('Orientation: 45 deg')
    # ax[1,1].imshow(filt_train[ind][1], cmap='gray')    
    # ax[2,0].set_title('Orientation: 90 deg')
    # ax[2,0].imshow(filt_train[ind][2], cmap='gray')
    # ax[2,1].set_title('Orientation: 135 deg')
    # ax[2,1].imshow(filt_train[ind][3], cmap='gray')
    # ax[0,1].axis('off')
    # fig.tight_layout()
    # fig.savefig(output_dir+'gabor_example_'+str(ind)+'.png')
    
    
# -- derive HOG features ------------------------------------------------------
if hog_features:
    print('Generating HOG features...')
    feat_train = []
    filt_train = []
    count = 0
    for img in x_train:
        if count % 10 == 0:
            print('Extracting HOG features: '+str(count)+'/'+str(len(x_train)))
        out, hog_image = hog(img, visualize=True, pixels_per_cell=pixels_per_cell,
                             orientations=orientations, transform_sqrt=transform_sqrt,
                             cells_per_block=cells_per_block)
        feat_train.append(out)
        filt_train.append(hog_image)
        count += 1
        
    feat_val = []
    filt_val = []
    for img in x_val:
        out, hog_image = hog(img, visualize=True, pixels_per_cell=pixels_per_cell,
                             orientations=orientations, transform_sqrt=transform_sqrt,
                             cells_per_block=cells_per_block)
        feat_val.append(out)
        filt_val.append(hog_image)        
    
    plot_filtered_imgs(x_train, plot_inds, filt_train, output_dir, name)
    # fig, ax = plt.subplots(2,3)
    # ax[0,0].set_title('Original')
    # ax[0,0].imshow(x_train[plot_inds[0]], cmap='gray')
    # ax[1,0].set_title('HOG image')
    # ax[1,0].imshow(filt_train[plot_inds[0]], cmap='gray')
    # ax[0,1].set_title('Original')
    # ax[0,1].imshow(x_train[plot_inds[1]], cmap='gray')
    # ax[1,1].set_title('HOG image')
    # ax[1,1].imshow(filt_train[plot_inds[1]], cmap='gray')
    # ax[0,2].set_title('Original')
    # ax[0,2].imshow(x_train[plot_inds[2]], cmap='gray')
    # ax[1,2].set_title('HOG image')    
    # ax[1,2].imshow(filt_train[plot_inds[2]], cmap='gray')    
    # fig.tight_layout()
    # fig.savefig(output_dir+'hog_example.png')    
    
# -- derive lbp features ------------------------------------------------------
if lbp_features:
    print('Generating lbp features...')
    feat_train = []
    filt_train = []
    count = 0
    for img in x_train:
        if count % 10 == 0:
            print('Calculating LBP features: '+str(count)+'/'+str(len(x_train)))
        if vanilla:
            img_lbp, hist = vanilla_lbp(img)
        else:
            img_lbp, hist = extract_lbp_features(x_train, radius, no_points)
        feat_train.append(hist)
        filt_train.append(img_lbp)
        count += 1

    feat_val = []
    filt_val = []
    count = 0
    for img in x_val:
        if vanilla:
            img_lbp, hist = vanilla_lbp(img)
        else:
            img_lbp, hist = extract_lbp_features(x_train, radius, no_points)            
        feat_val.append(hist)
        filt_val.append(img_lbp)
        count += 1
        
    plot_filtered_imgs(x_train, plot_inds, filt_train, output_dir, name)
    
if classify_ann:
    print('Training ANN...')
    # >> classify with ANN
    input_img = Input(shape=(len(bank_freq)*len(bank_theta)*num_sectors**2,))
    x = Dense(128, activation='relu')(input_img)
    x = Dense(64, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(1, activation='softmax')(x)
    ANN = Model(input_img, x)
    opt = optimizers.Adam(lr=0.000001)
    ANN.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    ANN.summary()
    history = ANN.fit(feat_train, y_train, epochs=10, batch_size=32,
                      shuffle=True, validation_data=(feat_val, y_val))
    y_pred = ANN.predict(feat_val)

if classify_svm:
    print('Training SVM...')
    # >> classify with SVM
    clf = svm.SVC()
    clf.fit(feat_train, y_train)
    y_pred = clf.predict(feat_val)
    
print('Make plots...')
make_confusion_matrix(y_val, y_pred, output_dir, name)
plot_tsne(x_val, y_val, feat_val, output_dir, name)
plot_tsne_pred(x_val, y_val, y_pred, feat_val, output_dir, name)         
    
                
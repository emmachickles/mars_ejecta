# ==============================================================================
#
# 200913 - preprocessing.py
# Preprocessing images and DEMs for machine learning.
# / Emma Chickles
# 
# <outputs>
# x_train.npy
# x_test.npy
# y_train.csv
# y_test.csv
# </outputs>
# 
# ==============================================================================

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
# import cv2
import random
# import omnigenus as om
import numpy as np
import pdb
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from skimage.filters import gabor
from skimage.filters import gabor_kernel



def get_resized_img(gdmp, width, height, padding=True):
    '''Resizes detrend map.'''
    dim = (width, height)

    # >> detrend
    gdmp.detrend(mapnum=2)
    imap = -1
    if padding:
        # >> image is larger than dim
        if gdmp.pdim[0] > width:
            gdmp.resize(dim)
            img = gdmp.maps[imap]

        else: # >> image is smaller than dim
            # TODO use multiple inputs (get multiple maps)
            # bottom = height - gdmp.pdim[1]
            # right = width - gdmp.pdim[0]
            # top, left = 0, 0
            bottom = int(np.ceil(float(height - gdmp.pdim[1])/2))
            top= int(np.floor(float(height - gdmp.pdim[1])/2))
            right = int(np.ceil(float(width - gdmp.pdim[0])/2))
            left= int(np.floor(float(width - gdmp.pdim[0])/2))
            minval = np.min(gdmp.maps[imap])
            img = cv2.copyMakeBorder(gdmp.maps[imap], top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=minval)

    else:
        gdmp.resize(dim)
        img = gdmp.maps[imap]

    return img

def normalize(x):
    xmin = np.min( np.min(x, axis=1, keepdims=True), axis=2, keepdims=True )
    x = x - xmin
    xmax = np.max( np.max(x, axis=1, keepdims=True), axis=2, keepdims=True )
    x = x / xmax
    return x

def standardize(x):

    avg = np.mean(np.mean(x, axis=2, keepdims=True), axis=1, keepdims=True)
    x += -avg
    stdevs = np.std(np.std(x, axis=2, keepdims=True), axis=1, keepdims=True)
    stdevs[ np.nonzero(stdevs == 0.) ] = 1e-8
    x = x / stdevs
    return x

def plot_set(x, inds, out, draw_circ=False, circ=False, show_class=False,
             y=False):
    x = x.astype(np.float32)
    fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(15,10))
    for i in range(4):
        for j in range(5):
            ind = inds[int(5*i + j)]
            ax[i,j].set_xlim([0, np.shape(x)[1]])
            ax[i,j].set_ylim([0, np.shape(x)[1]])
            ax[i,j].imshow(x[ind], cmap='gray')
            if draw_circ:
                circ_x = circ[ind][1]
                circ_y = circ[ind][2]
                circ_rad = circ[ind][0]
                ht.draw_circle(ax[i,j], False, circ_x, circ_y, circ_rad,
                               color='b')
            if show_class:
                ax[i,j].set_title('Class: ' + str(int(y[ind])))
    fig.tight_layout()
    fig.savefig(out)

def data_augmentation_experiments(data_dir='./data_orig/', output_dir='./',
                                  ind=0, nrows=5, ncols=4, rot_range=360,
                                  b_range=[0.5, 1.5], shear_range=40):
    '''Takes .npy files as inputs, not the raw images.
    Args:
        * data_dir: directory with '''
    
    # >> get image
    x_train = np.load(data_dir + 'x_train.npy')
    img = x_train[ind]
    img = np.expand_dims(img, -1)
    img = np.expand_dims(img, 0)
    
    fig, ax = plt.subplots(nrows+1, ncols, figsize=(ncols*2.5,(nrows+1)*2.5))
    
    # >> plot original image
    ax[0,0].set_title('Original')
    ax[0,0].imshow(img[0], cmap='gray')
    for col in range(1,ncols):
        ax[0,col].axis('off')
    
    for col in range(ncols):
        if col == 0:
            datagen = ImageDataGenerator(rotation_range=rot_range)
            ax[1,col].set_title('rotation_range: '+str(rot_range))
        elif col == 1:
            datagen = ImageDataGenerator(brightness_range=b_range)
            ax[1,col].set_title('brightness_range: '+str(b_range))
        elif col == 2:
            datagen=ImageDataGenerator(horizontal_flip=True,
                                       vertical_flip=True)
            ax[1,col].set_title('horizontal and vertical flips')
        elif col == 3:
            datagen=ImageDataGenerator(shear_range=shear_range)
            ax[1,col].set_title('shear_range: '+str(shear_range))
            
        # >> generate batch of images
        it = datagen.flow(img, batch_size=1)
        
        # >> plot results
        for row in range(nrows):
            # >> generate batch of images
            batch = it.next()
            
            # >> convert to unsigned integers for viewing
            new_img = batch[0].astype('uint8')
            ax[row+1,col].imshow(new_img, cmap='gray')
    
    for a in ax.flatten():
        a.set_xticklabels([])
        a.set_yticklabels([])
    
    fig.tight_layout()
    fig.savefig(output_dir+'data_augmentation_examples.png', dpi=300)
    

def data_augmentation_x_train(data_dir='./data_orig/',
                              output_dir='./data_augmented/',
                              preprocess_balance_train=True, batch_size=4,
                              rotation_range=360, brightness_range=[0.5,1.5],
                              horizontal_flip=True, vertical_flip=True):
    x_train = np.load(data_dir + 'x_train.npy')
    y_train = np.loadtxt(data_dir + 'y_train.txt')
    height, width = x_train.shape[1], x_train.shape[2]
    
    if preprocess_balance_train:
        print('Creating a balanced training set ...')
        num_ejecta = np.count_nonzero(y_train)
        inds = np.nonzero(y_train == 0.)[0][num_ejecta:]
        y_train = np.delete(y_train, inds)
        x_train = np.delete(x_train, inds, axis=0)
        
    x_train = np.expand_dims(x_train, -1)
    
    new_y_train = np.empty((0))
    new_x_train = np.empty((0,width,height,1))
    datagen = ImageDataGenerator(rotation_range=rotation_range,
                                 brightness_range=brightness_range,
                                 horizontal_flip=horizontal_flip,
                                 vertical_flip=vertical_flip)
    
    for i in range(len(x_train)):
        if i % 20 == 0: print(str(i)+'/'+str(len(x_train)))
        img = np.expand_dims(x_train[i], 0)
        new_x_train = np.append(new_x_train, img, axis=0)
        new_y_train = np.append(new_y_train, y_train[i])
        
        # >> randomly transform the image
        it = datagen.flow(img, batch_size=batch_size)
        batch = it.next()
        
        new_x_train = np.append(new_x_train, batch, axis=0)
        new_y_train = np.append(new_y_train, [y_train[i]]*batch_size)
        
    print('Shuffling...')
    inds = np.arange(len(new_x_train))
    random.Random(4).shuffle(inds)
    new_x_train = new_x_train[inds]
    new_y_train = new_y_train[inds]
    np.save(output_dir+'x_train.npy', new_x_train)
    np.savetxt(output_dir+'y_train.txt', new_y_train, delimiter=',', fmt='%d')
    
    plot_set(new_x_train, inds=np.arange(20), y=new_y_train,
             out=output_dir+'x_train_augmented.png', show_class=True)
    

def data_augmentation_imgs(x_train, y_train, fnames, fnames_train,
                          debug_data_augmentation=True,
                          width=240, height=240, out='./'):
    for i in range(len(y_train)):
        if i % 100 == 0:
            print(str(i) + '/' + str(len(y_train)))
        if y_train[i] == 1.:

            if debug_data_augmentation:
                fig, ax = plt.subplots(3,3)
                ax[0,0].axis('off')
                ax[0,2].axis('off')
                ax[0,1].imshow(x_train[i].astype(np.float32), cmap='gray')
                ax[0,1].set_title('Original')
                
            # >> flip horizontally
            tmp = np.fliplr(x_train[i])
            x_train = np.append(x_train, np.expand_dims(tmp, axis=0), axis=0)
            if debug_data_augmentation:
                ax[2,0].imshow(tmp.astype(np.float32), cmap='gray')
                ax[2,0].set_title('Flipped horizontally')

            # >> flip vertically
            tmp = np.flipud(x_train[i])
            x_train = np.append(x_train, np.expand_dims(tmp, axis=0), axis=0)
            if debug_data_augmentation:
                ax[2,1].imshow(tmp.astype(np.float32), cmap='gray')
                ax[2,1].set_title('Flipped vertically')
            

            # >> rotate 90 degrees
            tmp = x_train[i]
            for j in range(3):
                tmp = np.rot90(tmp)
                x_train = np.append(x_train, np.expand_dims(tmp, axis=0), axis=0)
                if debug_data_augmentation:
                    ax[1,j].imshow(tmp.astype(np.float32), cmap='gray')
                    ax[1,j].set_title('Rotated '+str((j+1)*90)+' degrees')
                

            # >> crop
            cut = int(width * 0.1)
            tmp = x_train[i][:,:-1 * cut]
            tmp = tmp[:,cut : ]
            tmp = tmp[:-1 * cut]
            tmp = tmp[cut : ]
            tmp = cv2.resize(tmp, (width, height), interpolation=cv2.INTER_AREA)
            x_train = np.append(x_train, np.expand_dims(tmp, axis=0), axis=0)
            if debug_data_augmentation:
                ax[2,2].imshow(tmp.astype(np.float32), cmap='gray')
                ax[2,2].set_title('Cropped')
                fig.tight_layout()
                fig.savefig(out + 'debug_data_augmentation.png')
                debug_data_augmentation=False

            for j in range(6):
                y_train = np.append(y_train, 1.)
                fnames_train = np.append(fnames_train, fnames[i])  
                
    return x_train, y_train, fnames_train

def ejecta_detection_preprocessing(imgpaths=['/mnt/d6/cfassett/jpg/',
                                             '/mnt/d6/cfassett/jpg2/'], out='./',
                                   width=500, height=500, train_partition=0.9,
                                   val_partition=0.05, test_partition=0.05,
                                   classification_excel='./fassett classifications 191200.xlsx',
                                   data_augmentation=True,
                                   gabor_filter=False, freq=0.5, theta=0.):
    '''Args:
        * imgpaths : contains raw .jpg images
        * width, height : target shape
        * train_partition + val_partition + test_partition = 1.
        * data_augmentation : deprecated 112620. Now use ImageDataGenerator
          while training to save on memory.
    TODO: consider using Tensorflow methods instead of cv2'''


    skip = ['08-001140+P18_007974_1897_XN_09N180W_B18_016558_1895_XI_09N180W-DEM-001_ctx-reproj-strchd_1040.jpg']
    
    # >> read Caleb's Excel sheet
    df = pd.read_excel(classification_excel)
    names = list(df['name'])
    attribs = list(df['attribs'])

    img_names = []
    for imgpath in imgpaths:
        img_names.append(os.listdir(imgpath))
    
    x = [] # >> images
    y = [] # >> classification (0=no ejecta, 1=ejecta preserved)
    fnames = []
    orig_shapes = []
    orig_shapes_ejecta = []
    orig_labels = []
    orig_labels_nan = []
    fnames_failed = []
    orig_labels_failed = []
    
    count = 0
    for i in range(len(names)):
        if i % 100 == 0:
            print(str(i) + '/' + str(len(names)))
        if type(attribs[i]) != np.float and names[i] not in skip:
            label = attribs[i][0]

            print('Opening ' + str(names[i]))
            # >> open image
            img=None
            for j in range(len(imgpaths)):
                if names[i] in img_names[j]:
                    img = cv2.imread(imgpaths[j] + names[i], 0)

            if type(img) == type(None):
                print('Failed to open ' + str(names[i]))
                orig_labels_nan.append(attribs[i])
            else:
                orig_labels.append(attribs[i])
                fnames.append(names[i])
                if label == 'e':
                    y.append(1.)
                    orig_shapes_ejecta.append([img.shape[0], img.shape[1]])
                else:
                    y.append(0.)
                orig_shapes.append([img.shape[0], img.shape[1]])
                # >> crop to get a square image
                num_rows = img.shape[0]
                num_cols = img.shape[1]
                if num_rows > num_cols:
                    row_ind = int(num_rows/2 - num_cols/2)
                    img = img[row_ind : row_ind+num_cols]
                elif num_cols > num_rows:
                    col_ind = int(num_cols/2 - num_rows/2)
                    img = img[:, col_ind : col_ind+num_rows]

                if gabor_filter:
                    theta_bank = np.linspace(0, 2*np.pi, 7)[:-1]
                    if count < 30:
                        fig, ax = plt.subplots(2, figsize=(10,20))
                        fig1, ax1 = plt.subplots(4, 3, figsize=(12,15))
                        ax[0].imshow(img, cmap='gray')
                        ax1[0][1].imshow(img, cmap='gray')

                    real_imgs = []
                    for theta in  theta_bank:   
                        real, imag = gabor(img, freq, theta)
                        real_imgs.append(real)
                    real_imgs = np.array(real_imgs)
                    img = np.sum(real_imgs, axis=0).astype('float32')
                    
                    if count < 30:
                        ax[1].imshow(img, cmap='gray')
                        fig.tight_layout()
                        fig.savefig(out+'gabor_example_'+str(count)+'.png',
                                    dpi=300)
                        plt.close(fig)

                        for img_i in range(6):
                            row = img_i//3 + 1
                            col = img_i - 3*(img_i//3)
                            ax1[row][col].imshow(real_imgs[img_i], cmap='gray')
                        ax1[3][1].imshow(img, cmap='gray')
                        fig1.tight_layout()
                        fig1.savefig(out+'gabor_sum_example_'+str(count)+'.png', dpi=300)
                        plt.close(fig1)
                        
                        count += 1
                    else:
                        pdb.set_trace()

                # >> resize image to (width, height)
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
                x.append(img)

        else:
            print('Skipping ' + str(names[i]))
            orig_labels_failed.append(attribs[i])

                            
    # >> shuffle
    print('Shuffling...')
    # x = np.array(x).astype(np.float32) # >> we do not want to do this

    x = np.array(x)
    y = np.array(y)
    fnames = np.array(fnames)
    inds = np.arange(len(x))
    random.Random(4).shuffle(inds)
    x = x[inds]
    y = y[inds]
    fnames = fnames[inds]

    # # >> normalize
    # print('Normalizing...')
    # # x = normalize(x)
    # x = standardize(x)
    
    # >> partition to training, validation and test sets
    print('Partititoning data...')
    split_ind = int(train_partition * len(x))
    val_split_ind = int((train_partition + val_partition) * len(x))
    x_train = x[:split_ind]
    y_train = y[:split_ind]
    x_val = x[split_ind:val_split_ind]
    y_val = y[split_ind:val_split_ind]
    x_test = x[val_split_ind:]
    y_test = y[val_split_ind:]

    fnames_train = fnames[:split_ind]
    fnames_val = fnames[split_ind:val_split_ind]
    fnames_test = fnames[val_split_ind:]

    if data_augmentation:

        print('Augmenting data...')
        
        # >> 112620: I now use ImageDataGenerator instead while training model
        x_train, y_train, fnames_train = \
            data_augmentation_imgs(x_train,y_train,fnames,fnames_train,
                                    width=width, height=height, out=out)
        

        # >> re-shuffle
        print('Re-shuffling training data')
        inds = np.arange(len(x_train))
        random.Random(4).shuffle(inds)
        x_train = x_train[inds]
        y_train = y_train[inds]
        fnames_train = fnames_train[inds]

    print('Fraction of training images with ejecta: ' + \
          str(float(np.count_nonzero(y_train))/ len(y_train)))
    print('Fraction of validation images with ejecta: ' + \
          str(float(np.count_nonzero(y_val))/ len(y_val)))
    print('Fraction of testing images with ejecta: ' + \
          str(float(np.count_nonzero(y_test))/ len(y_test)))
    

    # >> save training and testing sets
    print('Saving!')
    np.save(out + 'x_train.npy', x_train)
    np.save(out + 'x_val.npy', x_val)    
    np.save(out + 'x_test.npy', x_test)
    np.savetxt(out+'y_train.txt', y_train, delimiter=',', fmt='%d')
    np.savetxt(out+'y_val.txt', y_val, delimiter=',', fmt='%d')    
    np.savetxt(out+'y_test.txt', y_test, delimiter=',', fmt='%d')
    np.savetxt('fnames_train.txt', fnames_train, delimiter=',', fmt='%s')
    np.savetxt('fnames_val.txt', fnames_val, delimiter=',', fmt='%s')    
    np.savetxt('fnames_test.txt', fnames_test, delimiter=',', fmt='%s')

    # >> make some plots
    x_train_plot = x_train.astype(np.float32)
    plot_set(x_train_plot, np.arange(20), out=out+'x_train_preprocessed.png',
             draw_circ=False, show_class=True, y=y_train)
    x_val_plot = x_val.astype(np.float32)
    plot_set(x_val_plot, np.arange(20), out=out+'x_val_preprocessed.png',
             draw_circ=False, show_class=True, y=y_val)    
    x_test_plot = x_test.astype(np.float32)
    plot_set(x_test_plot, np.arange(20), out=out+'x_test_preprocessed.png',
             draw_circ=False, show_class=True, y=y_test)

    # >> make histogram of original sizes
    orig_shapes = np.array(orig_shapes)
    plt.figure()
    plt.hist(np.sort(orig_shapes[:,1]), bins=50)
    plt.xlabel('number of columns')
    plt.ylabel('number of labelled images')
    plt.savefig(out + 'resized_images.png')
    plt.close()
    
    plt.figure()
    plt.hist(np.sort(np.array(orig_shapes_ejecta)[:,1]), bins=50)
    plt.xlabel('number of columns')
    plt.ylabel('number of images with ejecta')
    plt.savefig(out + 'resized_ejecta_images.png')
    plt.close()
    
    return x_train, x_val, x_test, y_train, y_val, y_test, fnames_train, fnames_test, orig_shapes, orig_labels, orig_labels_nan, orig_labels_failed, orig_shapes_ejecta

def quality_assessment_preprocessing():
    '''
    <inputs>
        classified.txt from quick_classify.py
    </inputs>
    '''

    dempath = '/mnt/d6/picm/mars-fass3-updated/'
    output_dir = './test061520/'

    # CNN requires all images to be the same size, so all images are resized
    height = 500
    width = 500

    classifiedtxt = './classified2.txt' # >> classified as noisy 1, not noisy 0


    # :::::::::::::::::::::::::;:::::::::::::::::::::::::::::::::::::::::::::::::::

    dim = (width, height)

    x_train = []
    x_test = []
    y_train = []
    y_test = []

    with open(classifiedtxt, 'r') as f:
        lines = f.readlines()
        np.random.shuffle(lines)
        fnames = [line.split(',')[0][:-9]+'.bkl.gz' for line in lines]
        classes = np.array([line.split()[0][-1] for line in lines], 'int')

    # !! eventually needs to handle more than two classes
    class_1_inds = np.nonzero(classes == 1)[0]
    class_2_inds = np.nonzero(classes == 3)[0]

    num_samples = min(len(class_1_inds), len(class_2_inds))

    for i in range(num_samples):
        for j in range(2): # !! 2 = num_classes
            if j == 0:
                file = fnames[class_1_inds[i]]
                c = classes[class_1_inds[i]]
            if j == 1:
                file = fnames[class_2_inds[i]]
                c = classes[class_2_inds[i]]
            print(file)        
            gdmp = om.unpickle(dempath+file)
            
            img = get_resized_img(gdmp, width, height)
            
            # img = cv2.resize(gdmp.maps[2], dim, interpolation = cv2.INTER_AREA)
            if i <= int(0.9*num_samples): 
                x_train.append(img)
                y_train.append(c)
                with open('f_train.txt', 'a') as f:
                    f.write(file + '\n')
            else:
                x_test.append(img)
                y_test.append(c)
                with open('f_test.txt', 'a') as f:
                    f.write(file + '\n')

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # >> change labels from (1,3) to (0,1) = (not noisy, noisy)
    y_train[np.nonzero(y_train==1)] = 0.
    y_train[np.nonzero(y_train==3)] = 1.
    y_test[np.nonzero(y_test==1)] = 0.
    y_test[np.nonzero(y_test==3)] = 1.

    # -- normalize ----------------------------------------------------------------
    x_train = normalize(x_train)
    x_test = normalize(x_test)
    # -- save training and testing sets -------------------------------------------
    np.save('x_train.npy', x_train)
    np.save('x_test.npy', x_test)
    np.savetxt('y_train.csv', y_train, delimiter=',', fmt='%d')
    np.savetxt('y_test.csv', y_test, delimiter=',', fmt='%d')

    # -- visualize training, testing sets -----------------------------------------
    inds = np.nonzero(y_train==0)[0]
    fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(15,10))
    for i in range(4):
        for j in range(5):
            ax[i,j].imshow(x_train[inds[int(5*i+j)]])
    fig.savefig('train_0.png')

    inds = np.nonzero(y_train==1)[0]
    fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(15,10))
    for i in range(4):
        for j in range(5):
            ax[i,j].imshow(x_train[inds[int(5*i+j)]])
    fig.savefig('train_1.png')

    inds = np.nonzero(y_test==0)[0]
    fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(15,10))
    for i in range(4):
        for j in range(5):
            ax[i,j].imshow(x_test[inds[int(5*i+j)]])
    fig.savefig('test_0.png')

    inds = np.nonzero(y_test==1)[0]
    fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(15,10))
    for i in range(4):
        for j in range(5):
            ax[i,j].imshow(x_test[inds[int(5*i+j)]])
    fig.savefig('test_1.png')

if __name__ == '__main__':
    x_train, x_val, x_test, y_train, y_val, y_test, fnames_train, fnames_test, orig_shapes, orig_labels, orig_labels_nan, orig_labels_failed, orig_shapes_ejecta =\
        ejecta_detection_preprocessing(width=224, height=224, data_augmentation=False)
    
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # xmin = np.min(np.min(x_train, axis=1, keepdims=True), axis=2, keepdims=True)
    # x_train = x_train-xmin
    # xmax = np.max(np.max(x_train, axis=1, keepdims=True), axis=2, keepdims=True)
    # x_train = x_train/xmax

    # xmin = np.min(np.min(x_test, axis=1, keepdims=True), axis=2, keepdims=True)
    # x_test = x_test-xmin
    # xmax = np.max(np.max(x_test, axis=1, keepdims=True), axis=2, keepdims=True)
    # x_test = x_test/xmax

# ==============================================================================
#
# 20200914 - cnn.py
# VGG-inspired CNN for binary classification of ejecta preservation.
# / Emma Chickles
# Trains a convolutional neural network on scaled and normalized images to
# output a single value (0 or 1) representing the presence or absence of crater
# ejecta.
# Note that this CNN doesn't initialize with the pretrained weights from VGG,
# just uses the same architecture.
#
# TODO:
# * better format for hyperparameter optimization text file
#
# ==============================================================================

# -- inputs --------------------------------------------------------------------

output_dir = './plots210527-all-2/'
# data_dir = './data_orig/'
# data_dir = './data_gabor/gabor-'
data_dir = './data_all/'
do_test = False # >> validate or test
# data_dir = './data_augmented/'
# >> model_name can be 'simple_CNN', 'autoencoder', 'vgg16', 'vgg19', 'ResNet50'
#    'InceptionV3', 'Xception', 'fine_tuning', 'InceptionResNetV2'
pretrained_models = ['vgg16', 'vgg19', 'ResNet50',
    'InceptionV3', 'Xception', 'fine_tuning', 'InceptionResNetV2']
# model_name = 'vgg19'
model_name = 'ResNet50'
output_rad = False
width, height = 224, 224
reduction_factor = 0.

# >> do preprocessing?
preprocess_norm = False
preprocess_balance_train = True # >> ensure training set is balanced
preprocess_balance_val = True # >> ensure validation set is balanced
remove_crater = False # >> requires an array of radii and centers in x and y

# preprocess_gabor = False # >> not in use
gabor_weights = False

# >> do hyperparameter optimization?
hyperparam_opt = False
hyperparam_opt_diag = False

# >> train model?
run_model = True
model_init = './vgg19_model.h5'
data_aug = False
data_aug_factor = 3
rot_range = 360
b_range = [0.5, 1.5]
hor_flip, ver_flip = True, True

# >> plot results?
plot = True

# -- hyperparameters ----------------------------------------------------------

# >> parameter set for run_model
if model_name == 'simple_CNN' or model_name == 'autoencoder':
    p = {'num_conv_blocks': 5,
     'dropout': 0.1,
     'kernel_size': 5,
     'activation': 'elu',
     'num_dense': 1,
     'dense_units': 128,
     'lr': 0.000007,
     'epochs': 10,
     'batch_size': 32,
     'kernel_initializer': 'glorot_normal',
     'num_consecutive': 2,
     'latent_dim': 30,
     'l1': 0.0,
     'l2': 0.0,
     'num_filters': [16, 16, 32, 32, 64], 'batch_norm': 1}
    
else:
    p = {'num_pretrained_layers': None, 'lr': 1e-6, 'activation': 'elu',
         'kernel_initializer': 'glorot_normal', 'batch_size': 32, 'epochs': 40,
         'trainable': True, 'num_dense': 0, 'gabor': False}
    
p_mlp = {'hidden_units': [32, 16, 8, 4], 'epochs': 50, 'lr':0.0001}
    
# >> parameter space for hyperparameter optimization
if model_name == 'simple_CNN' or model_name=='autoencoder':
    params = {'num_conv_blocks':[1,2,3,4,5],
          'num_filters': [16, 32, [4,8,16,32,64], [8,16,32,64,128], [8, 8, 16, 16, 32],
                          [16, 16, 32, 32, 64]],
          'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
          'kernel_size': [5,7,11,13,15], 'activation':['elu', 'relu'], 'num_dense': [1],
          'dense_units': [128], 'lr': [0.00001],
          'epochs': [50],
          'batch_size': [64], 'kernel_initializer': ['glorot_normal'],
          'num_consecutive': [1,2], 'latent_dim': [25,30,35,45],
          'l1':[0.], 'l2':[0.], 'batch_norm': [0,1]}
else: 
    params = {'num_pretrained_layers': [None],
              'lr': [0.000008, 0.000005, 0.000002, 0.000001, 0.0000005, 0.00001],
              'activation': ['elu', 'relu'],
              'kernel_initializer': ['random_normal', 'glorot_normal', 'glorot_uniform'],
              'batch_size': [32], 'epochs': [10], 'trainable': [False]}
    
# -- import libraries ---------------------------------------------------------

# import matplotlib as mpl
# mpl.use('Agg')
from evaluation import *
from model import *
import numpy as np
from preprocessing import *
from itertools import product
import random
from tensorflow.keras.applications.vgg19 import preprocess_input   

# ------------------------------------------------------------------------------

# >> get training and testing set
x_train = np.load(data_dir + 'x_train.npy').astype('float32')
if do_test:
    fname = 'x_test.npy'
else:
    fname = 'x_val.npy'
x_val = np.load(data_dir + fname).astype('float32')
y_train = np.loadtxt(data_dir + 'y_train.txt')
if do_test:
    fname = 'y_test.txt'
else:
    fname = 'y_val.txt'
y_val = np.loadtxt(data_dir + fname)
# fnames_train = np.loadtxt(data_dir + 'fnames_train.txt', dtype='str')
# fnames_test = np.loadtxt(data_dir + 'fnames_test.txt', dtype='str')
# y_pred = np.loadtxt('y_pred.txt', delimiter=',')

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
    
# >> normalize
if preprocess_norm:
    print('Normalizing ...')
    x_train = normalize(x_train)
    x_val = normalize(x_val)
    
if remove_crater:
    
    success_train = np.loadtxt('./plots210111/circ_success_train.txt').astype('int')
    circ_train = np.load('./plots210111/circ_train_cnn.npy')
    
    x_center = circ_train[:,0]
    y_center = circ_train[:,1]
    rad = circ_train[:,2] * 1.5
    
    for i in range(len(x_train)):
        if i == 0:
            fig, ax = plt.subplots(2)
            ax[0].imshow(x_train[i], cmap='gray')
            
        grid = np.indices(x_train[i].shape)
        inds = np.nonzero((grid[0]-y_center[i])**2+(grid[1]-x_center[i])**2<rad[i]**2)
        x_train[i][inds] = 0.
        if i == 0:
            ax[1].imshow(x_train[i], cmap='gray')
            fig.savefig(output_dir+'remove_crater_ex.png')
    
    success_val = np.loadtxt('./plots210111/circ_success_val.txt').astype('int')
    circ_val = np.load('./plots210111/circ_val_cnn.npy')
    
    x_center = circ_val[:,0]
    y_center = circ_val[:,1]
    rad = circ_val[:,2] * 1.5
    
    for i in range(len(x_val)):
        grid = np.indices(x_val[i].shape)
        inds = np.nonzero((grid[0]-y_center[i])**2+(grid[1]-x_center[i])**2<rad[i]**2)
        x_val[i][inds] = 0.
    
# >> reshape y_train and y_val
new_y_train = []
new_y_val = []
for i in range(len(y_train)):
    if y_train[i] == 0.:
        new_y_train.append([1., 0.])
    else:
        new_y_train.append([0., 1.])
y_train = np.array(new_y_train)
        
for i in range(len(y_val)):
    if y_val[i] == 0.:
        new_y_val.append([1., 0.])
    else:
        new_y_val.append([0., 1.])

y_val = np.array(new_y_val)

# >> reshape x_train and x_val
x_train = np.expand_dims(x_train, -1)
x_val=  np.expand_dims(x_val, -1)

# >> extra preprocessing for pretrained models, requires 3-channel images
if model_name in pretrained_models:
    x_train = np.repeat(x_train, 3, axis=-1)
    x_train = preprocess_input(x_train) # !! don't really understand
    x_val = np.repeat(x_val, 3, axis=-1)
    x_val = preprocess_input(x_val)      

# :: hyperparameter optimization ::::::::::::::::::::::::::::::::::::::::::::::
if hyperparam_opt:
    
    # >> find all possible parameter sets
    p_combinations = list(product(*params.values()))
    random.shuffle(p_combinations)
    
    # >> randomly reduce the size of the parameter space to save memory
    if reduction_factor > 0.:
        p_combinations = p_combinations[:int(-1*reduction_factor*len(p_combinations))]
        
    # >> initialize text file to save hyperparameter results in
    with open(output_dir+model_name+'_hyperparam_results.txt', 'a') as f:
        f.write(','.join(params.keys())+',loss,val_loss,val_accuracy\n')
        
    # >> run the model for each parameter set
    for i in range(len(p_combinations)):
        
        # >> make parameter set dictionary
        p1 = {}
        for j in range(len(params.keys())):
            p1[list(params.keys())[j]] = p_combinations[i][j]
        print(p1)
            
        # >> run model
        if model_name == 'autoencoder':
            history, model = autoencoder(x_train, x_val, p1)
            bottleneck_val, bottleneck_ind = get_bottleneck(model, x_val,
                                                            output_dir, '_val')
            bottleneck_train, bottleneck_ind = get_bottleneck(model, x_train,
                                                              output_dir, '_train')  
            history_mlp, model_mlp = mlp(model, x_train, y_train, x_val, y_val, p1,
                                     p_mlp, bottleneck_val=bottleneck_val,
                                     bottleneck_train=bottleneck_train)        
            y_pred = model_mlp.predict(bottleneck_val)
        elif model_name == 'simple_cnn':
            history, model = simple_cnn(x_train, y_train, x_val, y_val, p1)
            y_pred = model.predict(x_val)
        else:
            history, model = \
                run_pretrained_model(x_train, y_train, x_val, y_val,
                                     model_name, p1, width=width,
                                     height=height, data_aug=data_aug,
                                     rot_range=rot_range, b_range=b_range,
                                     ver_flip=ver_flip, hor_flip=hor_flip)
            y_pred = model.predict(x_val)
            
        # >> create confusion matrix
        accuracy, recall, precision = \
            make_confusion_matrix(y_val, y_pred, output_dir, model_name)        
        
        # >> save results to text file
        for j in range(len(params.keys())):
            with open(output_dir+model_name+'_hyperparam_results.txt', 'a') as f:
                f.write(str(p_combinations[i][j])+',')        
        with open(output_dir+model_name+'_hyperparam_results.txt', 'a') as f:
            f.write(str(history.history['loss'][-1])+','+\
                    str(history.history['val_loss'][-1])+','+str(accuracy)+'\n')
    
        # >> delete big objects from memory
        del history
        del model
        if model_name == 'autoencoder':
            del bottleneck_val
            del bottleneck_train
            del history_mlp
            del model_mlp


# :: hyperparameter optimization diagnosis ::::::::::::::::::::::::::::::::::::
if hyperparam_opt_diag:
    # !! streamline this
    
    # >> read hyperparameter results text file
    with open(output_dir+model_name+'_hyperparam_results.txt', 'r') as f:
        lines = f.readlines()
        key_list = lines[0].split(',')

    # >> change num_filters (a list) parameter into something we can compare (ints)
    if 'num_filters' in key_list:
        ind = key_list.index('num_filters')
        key_list.remove('num_filters')
        key_list.insert(0, 'max_num_filters')
        key_list.insert(0, 'min_num_filters')
    
    # >> initialize a dictionary to place hyperparameter sets and results 
    res = {}     
    for key in key_list:
        res[key] = []
        
    for line in lines[1:]:
        
        # >> check if num_filters is in a list
        if '[' in line: # >> checks if num_filters is in a list
            num_filters = line.split('[')[1].split(']')[0]
            # >> turn into an array
            num_filters = np.array(num_filters.split(',')).astype('int')
            min_num_filters = np.min(num_filters)
            max_num_filters = np.max(num_filters)
            
            # >> add parameter set to data_arr
            new_line = list(np.concatenate([[min_num_filters, max_num_filters],
                                       line.split('[')[0].split(','),
                                       line.split(']')[1].split(',')]))

    
        else:
            if 'num_filters' in key_list:
                num_filters = line.split(',')[ind]
                new_line = list(np.concatenate([[num_filters, num_filters],
                                                line.split(',')[:ind],
                                                line.split(',')[ind+1:]]))
            else:
                new_line = line.split(',')
            
        while '' in new_line:
            new_line.remove('')
        for i in range(len(key_list)):
            res[key_list[i]].append(new_line[i])
                
                
    # >> convert values to floats, if appropriate
    for key in key_list:
        if key in ['min_num_filters', 'max_num_filters', 'num_conv_blocks',
                   'kernel_size', 'num_dense', 'dense_units', 'epochs',
                   'batch_size', 'num_consecutive', 'latent_dim']:
            res[key] = np.array(res[key]).astype('int')
        elif res[key][0] == 'None':
            res[key] = np.array([None]*len(res[key]))
        elif res[key][0] in ['True', 'False']:
            res[key] = np.array([el=='True' for el in res[key]])
        elif key != 'activation' and key != 'kernel_initializer':
            res[key] = np.array(res[key]).astype('float')
        
    # >> make correlation heat map plot
    res['val_accuracy'] = res.pop('val_accuracy\n') # >> rename key
    key_list = list(res.keys())
    correlation_heatmap(res, output_dir, model_name)     
    
    # data_arr = np.loadtxt(input_file, delimiter=',', dtype='str', skiprows=1)
    best_ind = np.argmax(res['val_accuracy'])
    p_best = {}
    for i in range(len(key_list)):
        p_best[key_list[i]] = res[key_list[i]][best_ind]
        
    # >> get num_filters
    if 'num_filters' in key_list:
        line = lines[best_ind+1]
        if '[' in line: # >> checks if num_filters is in a list
            num_filters = line.split('[')[1].split(']')[0]
            # >> turn into an array
            num_filters = np.array(num_filters.split(',')).astype('int')
        else:
            num_filters = line.split(',')[ind]        
        p_best['num_filters'] = list(num_filters)
        
    print('Best hyperparameter set: '+str(p_best))
    print('Best loss: '+str(p_best['loss']))
    print('Best validation loss: '+str(p_best['val_loss']))
    print('Best validation accuracy: '+str(p_best['val_accuracy']))
    
    # >> change number of epochs
    p_best['epochs'] = p['epochs']
    p = p_best

# :: train model ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

if run_model:
    if model_name == 'simple_CNN':
        history, model = simple_cnn(x_train, y_train, x_val, y_val, p)
        model.save(output_dir+model_name+'_model.h5')
        
    elif model_name == 'autoencoder':
        history, model = autoencoder(x_train, x_val, p)
        model.save(output_dir+model_name+'_model.h5')
        bottleneck_val, bottleneck_ind = get_bottleneck(model, x_val,
                                                        output_dir, '_val')
        bottleneck_train, bottleneck_ind = get_bottleneck(model, x_train,
                                                          output_dir, '_train')  
        history_mlp, model_mlp = mlp(model, x_train, y_train, x_val, y_val, p,
                                 p_mlp, bottleneck_val=bottleneck_val,
                                 bottleneck_train=bottleneck_train)
        model_mlp.save(output_dir+model_name+'_mlp.h5')
        epoch_plot(history, output_dir, model_name)
        epoch_plot(history_mlp, output_dir, model_name+'_mlp')
    
    elif model_name == 'fine_tuning':
        history, model = do_fine_tuning(x_train, y_train, x_val, y_val, model_name,
                                        p, model_init)
        model.save(output_dir+model_name+'_model.h5')
    
    else:
        history, model = run_pretrained_model(x_train, y_train, x_val,
                                              y_val, model_name, p)        
        model.save(output_dir+model_name+'_model.h5')
    
    
    epoch_plot(history, output_dir, model_name)     
else:
    # model = load_model(output_dir+model_name+'_model.h5')
    model = load_model(model_init)
    if model_name == 'autoencoder':
        model_mlp = load_model(output_dir+model_name+'_mlp.h5')

# :: plot results :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
if plot:
    # >> detect ejecta in validation set
    if model_name == 'autoencoder':
        bottleneck_val = np.loadtxt(output_dir+model_name+'bottleneck_val.txt')
        bottleneck_train = np.loadtxt(output_dir +model_name+'bottleneck_train.txt')    
        x_pred = model.predict(x_val)
        y_pred = model_mlp.predict(bottleneck_val)
    else:
        bottleneck_val, bottleneck_ind = \
            get_bottleneck(model, x_val, output_dir, '_val')
        y_pred = model.predict(x_val)
    
    # >> save results
    if model_name != 'autoencoder':
        np.savetxt(output_dir+'y_pred.txt', y_pred, delimiter=',', fmt='%d')    
    else:
        np.save(output_dir+'x_pred.txt', np.squeeze(x_pred, -1))
    
    # >> model architecture text file
    save_model_summary(output_dir, model, prefix=model_name+'_')
    
    # >> saliency maps
    if model != 'autoencoder':
        plot_saliency_maps_tf(model, x_val, y_val, y_pred, output_dir, model_name)

    # >> make sure images are grayscale and float32
    x_val = x_val.astype(np.float32)
    y_val = y_val.astype(np.float32)    
    if model_name == 'autoencoder':
        x_pred = x_pred.astype(np.float32)
    else:
        y_pred = y_pred.astype(np.float32)    
    if model_name in pretrained_models:
        x_val = x_val[:,:,:,0]
    
    # >> validation set results
    if model_name != 'autoencoder':
        plot_test_set(x_val, y_val, y_pred, output_dir+model_name+'_res.png',
                      width=width, height=height)
    else:
        plot_input_output(x_val, x_pred, output_dir, model_name)
    
    # >> confusion matrix (from evaluation.py)
    accuracy, recall, precision = \
        make_confusion_matrix(y_val, y_pred, output_dir, model_name)
        
    # >> plot false positives and false negatives (from evaluation.py)
    plot_FP_FN(x_val, y_val, y_pred, output_dir, model_name)
        
    # >> visualize filters from the 1st conv layer
    visualize_filters(model, model_name, nrows=2, ncols=4, output_dir=output_dir)
    
    # >> visualize intermediate activation maps
    inds = [np.nonzero(y_val[:,0])[0][0], np.nonzero(y_val[:,1])[0][0]]
    plot_intermediate_activations(x_val, model, inds=inds, output_dir=output_dir,
                                  model_name=model_name, nrows=2, ncols=4)
    
    # >> visualize latent space
    if model_name == 'autoencoder':
        plot_tsne(x_val, y_val, bottleneck_val, output_dir, model_name)
        plot_latent_space(bottleneck_val, output_dir, model_name)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    

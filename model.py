import numpy as np
from skimage.filters import gabor_kernel
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras import activations
from tensorflow.keras import optimizers
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def set_gabor_weights(model, bank_freq=[0.25, 0.375, 0.5, 0.675],
                      layer_num=[1], n_stds=0.2):
    '''layer.weights include [weights, biases], where
    weights have shape (kernel_size, kernel_size, channels, fiters) and
    biases have shape (filters)
    
    cv2 also has a get_gabor_kernel function where you can specify the number
    of pixels in kernel. I'm unfortunately using skimage, so I set n_stds=0.5
    to fix pixel size as (3 x 3 pixels). So this is hardwired! :( 
        
    Currently, this sets the same Gabor filter for all channels
    '''
    
    for num in layer_num:
        kernel_size, _, n_channels, n_filters = \
            model.layers[num].weights[0].shape.as_list()
        
        # >> set biases to zeroes
        biases = np.zeros(n_filters)
        
        # >> set weights for each filter and channel to a gabor filter
        bank_theta = np.linspace(0, 180, int(n_filters/len(bank_freq))) * np.pi / 180.
        
        weights = np.zeros(model.layers[num].get_weights()[0].shape)
        filt = 0
        for freq in bank_freq:
            for theta in bank_theta:
                gk = gabor_kernel(frequency=freq, theta=theta, n_stds=n_stds)
                for chl in range(n_channels):
                    weights[:,:,chl,filt] = gk.real
                    
                filt += 1
                
        model.layers[num].set_weights([weights, biases])
        model.layers[num].trainable = False
    
    return model

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


def plot_test_set(x_val, y_val, y_pred, out, output_rad=False, draw_circ=False,
                  width=500, height=500, nrows=5):
    # >> let's plot some of the testing set with their actual and predicted circles
    fig, ax = plt.subplots(nrows=nrows, ncols=nrows, figsize=(15,15))
    plt.subplots_adjust(hspace=0)
    for i in range(nrows):
        for j in range(nrows):
            ind = int(nrows*i+j)

            ax[i,j].set_xlim([0, width])
            ax[i,j].set_ylim([0, height])
            ax[i,j].imshow( np.reshape(x_val[ind], (width, height)), cmap='gray')
            ax[i,j].set_title('True class: '+str(np.argmax(y_val[ind])) + \
                              '\nPredicted class: '+str(np.argmax(y_pred[ind])))
            ax[i,j].set_xticklabels([])
            ax[i,j].set_yticklabels([])

            # >> draw recentered circle
            if draw_circ:
                if output_rad:
                    x = y_val[ind][1]
                    y = y_val[ind][2]
                    rad = y_val[ind][0]
                else:
                    x = y_val[ind][0]
                    y = y_val[ind][1]
                    rad = 50.
                # x = y_val[ind][0]
                # y = y_val[ind][1]
                # rad = rad_test[ind]
                ht.draw_circle(ax[i,j], False, x, y, rad, color='b',
                               marker=True)


                # >> draw predicted circle
                if output_rad:
                    x = y_pred[ind][1]
                    y = y_pred[ind][2]                
                    rad = y_pred[ind][0]
                else:
                    x = y_pred[ind][0]
                    y = y_pred[ind][1]
                    rad = 50.
                # x = y_pred[ind][0]
                # y = y_pred[ind][1]
                ht.draw_circle(ax[i,j], False, x, y, rad, color='r',
                               marker=True)

    plt.tight_layout()
    fig.savefig(out)

def plot_input_output(x_val, x_pred, output_dir, model_name='',
                  width=224, height=224, nrows=5):
    # >> let's plot some of the testing set with their reconstructions
    fig, ax = plt.subplots(nrows=nrows, ncols=2)
    plt.subplots_adjust(wspace=0)
    for i in range(nrows):

        ax[i,0].set_xlim([0, width])
        ax[i,0].set_ylim([0, height])
        ax[i,0].imshow( np.reshape(x_val[i], (width, height)), cmap='gray')
        # ax[i,0].set_xticklabels([])
        # ax[i,0].set_yticklabels([])
        
        ax[i,1].set_xlim([0, width])
        ax[i,1].set_ylim([0, height])
        ax[i,1].imshow( np.reshape(x_pred[i], (width, height)), cmap='gray')
        # ax[i,1].set_xticklabels([])
        # ax[i,1].set_yticklabels([])        

    plt.tight_layout()
    plt.tight_layout()
    fig.savefig(output_dir+model_name+'_input_output.png', dpi=300)

def save_model_summary(output_dir, model, prefix=''):
    with open(output_dir + prefix + 'model_summary.txt', 'a') as f:
        model.summary(print_fn=lambda line: f.write(line + '\n'))

def get_activations(model, x_val, ind):
    layer_outputs = [layer.output for layer in model.layers][1:]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(np.expand_dims(x_val[ind], 0))
    return activations

def plot_intermediate_activations(x_val, model, inds=[0], output_dir='./',
                                  nrows=4, ncols=4, model_name=''):
    for i in range(len(inds)):
        ind = inds[i]
        activations = get_activations(model, x_val, ind)

        # >> plot input
        fig, ax = plt.subplots()
        ax.imshow(np.squeeze(x_val[ind], axis=-1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        fig.savefig(output_dir + 'intermed_act_img'+str(i)+'_0input.png',
                    dpi=300)
        plt.close()

        for j in range(len(activations)):
            name = model.layers[j+1].name

            if model_name == 'autoencoder' and j == len(activations)-1:
                fig, ax = plt.subplots()
                ax.imshow(np.squeeze(activations[j][0], -1))
                ax.set_xticklabels([])
                ax.set_yticklabels([])       
            
            elif name.split('_')[0] in ['conv2d', 'max']:
                # >> plot first 16 filters
                fig, ax = plt.subplots(nrows, ncols)
                for r in range(nrows):
                    for c in range(ncols):
                        ax[r,c].imshow(activations[j][0][:,:,c*nrows+r])
                        ax[r,c].set_xticklabels([])
                        ax[r,c].set_yticklabels([])

            elif name.split('_')[0] == 'dense':
                fig, ax = plt.subplots()
                ax.imshow(activations[j])
                ax.set_xticklabels([])
                ax.set_yticklabels([])

            fig.tight_layout()
            fig.savefig(output_dir+model_name+'_intermed_act_img'+str(i)+'_'+\
                        str(j+1)+name+'.png', dpi=300)
            plt.close()

def visualize_filters(model, model_name, nrows=4, ncols=4, output_dir='./',
                      cmap='gray'):
    '''Don't take too literally! Doesn't include biases'''
    layer_inds = np.nonzero(['conv' in x.name for x in model.layers])[0]
    # >> for first layer, can plot filters as images
    filters, biases = model.layers[layer_inds[0]].get_weights()
    filters = np.squeeze(filters, axis=2)
    fig, ax = plt.subplots(nrows, ncols, constrained_layout=True)
    for r in range(nrows):
        for c in range(ncols):
            im=ax[r,c].imshow(filters[:,:,c*nrows+r], cmap=cmap)
            ax[r,c].set_xticklabels([])
            ax[r,c].set_yticklabels([])
    fig.colorbar(im, ax=ax[:,c])            
    fig.savefig(output_dir+model_name+'_layer0_weights.png', dpi=300)
    
# def plot_tsne():        
            
class Gray2VGGInput( Layer ) :
    '''From https://stackoverflow.com/questions/52065412/how-to-use-1-
    channel-images-as-inputs-to-a-vgg-model
    Must normalize input gray scale image to match the original image
    preprocessing used in the pretrained VGG network'''
    def build( self, x ) :
        self.image_mean = K.variable(value=np.array([103.939, 116.779, 123.68]).reshape([1,1,1,3]).astype('float32'),
                                     dtype='float32', name='imageNet_mean')
        
    def call( self, x ) :
        rgb_x = K.concatenate( [x,x,x], axis=-1)
        norm_x = rgb_x - self.image_mean
        return norm_x

    def compute_output_shape( self, input_shape ) :
        return input_shape[:3] + (3,)
    
def simple_cnn(x_train, y_train, x_val, y_val, p):
    width, height = np.shape(x_train)[1:3]
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
    x = Dense(2, activation='softmax',
              kernel_initializer=p['kernel_initializer'])(x)

    model = Model(input_img, x)
    opt = optimizers.Adam(lr=p['lr'])
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=['accuracy'])
    model.summary()

    if data_aug:
        aug = ImageDataGenerator(rotation_range=rot_range, brightness_range=b_range,
                                 vertical_flip=ver_flip, horizontal_flip=hor_flip)
        history = model.fit_generator(aug.flow(x_train, y_train, batch_size=p['batch_size']),
                                      validation_data=(x_val, y_val),
                                      steps_per_epoch=len(x_train)//p['batch_size'],
                                      epochs=p['epochs'])

    else:
        history = model.fit(x_train, y_train, epochs=p['epochs'],
                            batch_size=p['batch_size'],
                            shuffle=True, validation_data=(x_val, y_val))

    return history, model

def do_fine_tuning(x_train, y_train, x_val, y_val, model_name, p, model_init,
                   data_aug=False, rot_range=360, b_range=[0.5,1.5],
                   hor_flip=False, ver_flip=False):    
    print('Loading weights')
    model = load_model(model_init)
    opt = optimizers.Adam(lr=p['lr'])
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=['accuracy'])
    model.summary()
    # >> train model
    if data_aug:
        aug = ImageDataGenerator(rotation_range=rot_range, brightness_range=b_range,
                                 vertical_flip=ver_flip, horizontal_flip=hor_flip)
        history = model.fit_generator(aug.flow(x_train, y_train, batch_size=p['batch_size']),
                                      validation_data=(x_val, y_val),
                                      steps_per_epoch=len(x_train)//p['batch_size'],
                                      epochs=p['epochs'])
    else:
        history = model.fit(x_train, y_train, batch_size=p['batch_size'],
                            epochs=p['epochs'], shuffle=True,
                            validation_data=(x_val, y_val))

# def extract_feats():
    
#     if model_name == 'vgg16':
#         from tensorflow.keras.applications.vgg16 import VGG16 as model_pretrained
#         from tensorflow.keras.applications.vgg16 import preprocess_input
#     elif model_name == 'vgg19':
#         from tensorflow.keras.applications.vgg19 import VGG19 as model_pretrained
#         from tensorflow.keras.applications.vgg19 import preprocess_input
#     elif model_name == 'ResNet50':
#         from tensorflow.keras.applications.resnet import ResNet50 as model_pretrained
#         from tensorflow.keras.applications.resnet import preprocess_input
#     elif model_name == 'Xception':
#         from tensorflow.keras.applications.xception import Xception as model_pretrained
#         from tensorflow.keras.applications.xception import preprocess_input
#     elif model_name == 'InceptionV3':
#         from tensorflow.keras.applications.inception_v3 import InceptionV3 as model_pretrained
#         from tensorflow.keras.applications.inception_v3 import preprocess_input
#     elif model_name == 'InceptionResNetV2':
#         from tensorflow.keras.applications import InceptionResNetV2 as model_pretrained
#         from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input 
        
    

def run_pretrained_model(x_train, y_train, x_val, y_val, model_name, p,
                         width=224, height=224, data_aug=False, rot_range=360,
                         b_range=[0.5,1.5], hor_flip=False, ver_flip=False):      
    
    if model_name == 'vgg16':
        from tensorflow.keras.applications.vgg16 import VGG16 as model_pretrained
        from tensorflow.keras.applications.vgg16 import preprocess_input
    elif model_name == 'vgg19':
        from tensorflow.keras.applications.vgg19 import VGG19 as model_pretrained
        from tensorflow.keras.applications.vgg19 import preprocess_input
    elif model_name == 'ResNet50':
        from tensorflow.keras.applications.resnet import ResNet50 as model_pretrained
        from tensorflow.keras.applications.resnet import preprocess_input
    elif model_name == 'Xception':
        from tensorflow.keras.applications.xception import Xception as model_pretrained
        from tensorflow.keras.applications.xception import preprocess_input
    elif model_name == 'InceptionV3':
        from tensorflow.keras.applications.inception_v3 import InceptionV3 as model_pretrained
        from tensorflow.keras.applications.inception_v3 import preprocess_input
    elif model_name == 'InceptionResNetV2':
        from tensorflow.keras.applications import InceptionResNetV2 as model_pretrained
        from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input 
        
    pretrained_init = model_pretrained(weights='imagenet',
                                       include_top=p['include_top'],
                                       input_shape=(224,224,3))
    # # >> random initialization
    # random_init = model_pretrained(weights=None, include_top=False,
    #                                input_shape=(224,224,3))
        
    input_img = Input(shape=(width, height, 3)) 
    
    # !! can currently only change initialization for VGG models... (TODO)
    if model_name in ['vgg16', 'vgg19']:
        
        if type(p['num_pretrained_layers']) == type(None):
            p['num_pretrained_layers'] = len(pretrained_init.layers)+1
            
        if p['include_top']:
            max_ind = len(pretrained_init.layers)-1
        else:
            max_ind = len(pretrained_init.layers)
        for i in range(1, max_ind):
            
            # # >> check whether to use pretrained or random weights
            # if i < p['num_pretrained_layers']:
            #     layer = pretrained_init.layers[i + 1]
            #     layer.trainable = p['trainable']
            # else:
            #     layer = random_init.layers[i + 1]
                
            layer = pretrained_init.layers[i]
            layer.trainable = p['trainable']

            # >> add layer to model
            if i == 1:
                x = layer(input_img)
            else:
                x = layer(x)
            
        if not p['include_top']:
            x = Flatten()(x)
        for i in range(p['num_dense']):
            x = Dense(4096, activation=p['activation'],
                      kernel_initializer=p['kernel_initializer'])(x)
        x = Dense(2, activation='softmax',
                  kernel_initializer=p['kernel_initializer'])(x)
    else:
        x = pretrained_init(input_img)
        x = AveragePooling2D(pool_size=(x.shape[1],x. shape[2]))(x)
        x = Flatten()(x)        
        x = Dense(2, activation='softmax',
                  kernel_initializer=p['kernel_initializer'])(x)
        
    model = Model(input_img, x)
    
    if p['gabor']:
        model = set_gabor_weights(model)
    
    opt = optimizers.Adam(lr=p['lr'])
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=['accuracy'])
    model.summary()
    # >> train model
    if data_aug:
        aug = ImageDataGenerator(rotation_range=rot_range, brightness_range=b_range,
                                 vertical_flip=ver_flip, horizontal_flip=hor_flip)
        aug.fit(x_train)
        history = model.fit(aug.flow(x_train, y_train, batch_size=p['batch_size']),
                            validation_data=(x_val, y_val),
                            steps_per_epoch=data_aug_factor*len(x_train)//p['batch_size'],
                            epochs=p['epochs'])    
    else:
        history = model.fit(x_train, y_train, batch_size=p['batch_size'],
                            epochs=p['epochs'], shuffle=True,
                            validation_data=(x_val, y_val))
    
    # >> remove weights from memory
    del pretrained_init
    # del random_init
    return history, model
   
def autoencoder(x_train, x_val, p):
    width, height = np.shape(x_train)[1:3]
    input_img = Input(shape=(width, height, 1))
    x = input_img
    if type(p['num_filters']) != type([]):
        p['num_filters'] = [p['num_filters']]*p['num_conv_blocks']
    for i in range(p['num_conv_blocks']):
        for j in range(p['num_consecutive']):
            x = Conv2D(p['num_filters'][i], kernel_size=p['kernel_size'],
                       activation=p['activation'], padding='same',
                       kernel_initializer=p['kernel_initializer'],
                       kernel_regularizer=regularizers.l1_l2(l1=p['l1'], l2=p['l2']))(x)      
        x = Dropout(p['dropout'])(x)
        x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    x = Flatten()(x)    
    x = Dense(p['latent_dim'], activation=p['activation'],
              kernel_initializer=p['kernel_initializer'],
              kernel_regularizer=regularizers.l1_l2(l1=p['l1'], l2=p['l2']))(x)
    # reduction_factor = 2**(p['num_consecutive']*p['num_conv_blocks'])
    reduction_factor = 2**p['num_conv_blocks']
    x = Dense((int((width/reduction_factor)**2 * p['num_filters'][i])),
              kernel_regularizer=regularizers.l1_l2(l1=p['l1'], l2=p['l2']))(x)
    x = Reshape((int(width/reduction_factor), int(height/reduction_factor),
                 p['num_filters'][i]))(x)
    for i in range(p['num_conv_blocks']):
        x = UpSampling2D(size=(2,2))(x)   
        x = Dropout(p['dropout'])(x)    
        for j in range(p['num_consecutive']):
            if i == p['num_conv_blocks']-1 and j == p['num_consecutive']-1:
                p['num_filters'][p['num_conv_blocks']-i-1]=1 # >> use 1 filters
            x = Conv2DTranspose(p['num_filters'][p['num_conv_blocks']-i-1],
                                kernel_size=p['kernel_size'],
                                activation=p['activation'], padding='same',
                                kernel_initializer=p['kernel_initializer'],
                                kernel_regularizer=regularizers.l1_l2(l1=p['l1'], l2=p['l2']))(x)      
            
    model = Model(input_img, x)
    opt = optimizers.Adam(lr=p['lr'])
    model.compile(loss='mean_squared_error', optimizer=opt)
    model.summary()
    
    history = model.fit(x_train, x_train, epochs=p['epochs'],
                        batch_size=p['batch_size'],
                        shuffle=True, validation_data=(x_val, x_val))    
    
    return history, model

def mlp(model, x_train, y_train, x_val, y_val, p, p_mlp,
        bottleneck_val=None, bottleneck_train=None):
    if type(bottleneck_val) == type(None):
        print('Getting bottleneck for x_val')
        bottleneck_val, bottleneck_ind = get_bottleneck(model, x_val)
    if type(bottleneck_train) == type(None):
        print('Getting bottleneck for x_train')
        bottleneck_train, bottleneck_ind = get_bottleneck(model, x_train)
        
    input_img = Input(shape=(p['latent_dim'],))
    x = input_img
    for i in range(len(p_mlp['hidden_units'])):
        x = Dense(p_mlp['hidden_units'][i], activation=p['activation'],
                  kernel_initializer=p['kernel_initializer'])(x)
    x = Dense(2, activation='softmax',
              kernel_initializer=p['kernel_initializer'])(x)        
    model_mlp = Model(input_img, x)
    opt = optimizers.Adam(lr=p_mlp['lr'])
    model_mlp.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model_mlp.summary()
    history_mlp = model_mlp.fit(bottleneck_train, y_train, epochs=p_mlp['epochs'],
                                batch_size=p['batch_size'], shuffle=True,
                                validation_data=(bottleneck_val, y_val)) 
    return history_mlp, model_mlp

def get_bottleneck(model, x_val, output_dir='./', suffix=''):
    dense_inds = np.nonzero(['dense' in x.name for x in model.layers])[0]
    bottleneck_ind = dense_inds[0]
                        
    activation_model = Model(inputs=model.input,
                             outputs=model.layers[bottleneck_ind].output)
    bottleneck = activation_model.predict(x_val)
    np.savetxt(output_dir+'bottleneck'+suffix+'.txt', bottleneck)
    
    return bottleneck, bottleneck_ind

def get_last_activation(model, x_val):
    dense_inds = np.nonzero(['dense' in x.name for x in model.layers])[0]
    ind = dense_inds[-1]
    activation_model = Model(inputs=model.input, outputs=model.layers[ind].output)    
    activation = activation_model.predict(x_val)
    return activation, ind


    
def plot_latent_space(bottleneck, output_dir='./', model_name='./', log=True):
    from matplotlib.colors import LogNorm
    latentDim = np.shape(bottleneck)[1]
    fig, axes = plt.subplots(nrows = latentDim, ncols = latentDim,
                             figsize=(15,15))    
    for i in range(latentDim):
        axes[i,i].hist(bottleneck[:,i], 50, log=log)
        axes[i,i].set_aspect(aspect=1)
        for j in range(i):
            if log:
                norm = LogNorm()
            axes[i,j].hist2d(bottleneck[:,j], bottleneck[:,i], bins=50,
                             norm=norm)
            # >> remove axis frame of empty plots
            axes[latentDim-1-i, latentDim-1-j].axis('off')

        # >> x and y labels
        axes[i,0].set_ylabel('\u03C6'+str(i), fontsize='x-small')
        axes[latentDim-1,i].set_xlabel('\u03C6'+str(i), fontsize='x-small')

    # >> removing axis
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    plt.subplots_adjust(hspace=0, wspace=0)

    plt.savefig(output_dir+model_name+'_latent_space.png', dpi=300)
    plt.close(fig)    
    
def plot_saliency_maps(model, x_val, bottleneck, bottleneck_ind, inds,
                       output_dir='./', model_name='', heatmap=True):
    
    latentDim = np.shape(bottleneck)[1]
    
    for ind in inds:
        for i in range(latentDim):
            out = visualize_saliency(model, bottleneck_ind, [i], x_val[ind])
            fig, ax = plt.subplots()
            ax.imshow(out, cmap='seismic')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            plt.savefig(output_dir + model_name + '_saliency_filter'+str(i)+\
                        '_ind'+str(ind)+'.png', dpi=300)
            plt.close(fig)
            
            if heatmap:
                out1 = overlay(np.squeeze(x_val[ind], -1), out)
                out1 = out1.astype(np.float32)
                fig, ax = plt.subplots()
                ax.imshow(out1, cmap='seismic')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                plt.savefig(output_dir + model_name + '_heatmap_filter'+str(i)+\
                            '_ind'+str(ind)+'.png', dpi=300)
                plt.close(fig)
                        
def plot_saliency_maps_tf(model, x_val, y_val, y_pred, output_dir, model_name,
                          nrows=3, ncols=3):
    '''https://github.com/keisen/tf-keras-vis/blob/master/examples/attentions.ipynb'''
    from tf_keras_vis.saliency import Saliency
    from tf_keras_vis.gradcam import Gradcam
    from tf_keras_vis.utils import normalize
    from matplotlib import cm
    
    # >> Find true postitives (TP), true negatives (TN), false positives (FP)
    # >> and false negatives (FN)
    TP, TN, FP, FN = [], [], [], []
    y_pred = np.round(y_pred)
    for i in range(len(x_val)):
        if y_val[i][0] == y_pred[i][0] and y_val[i][0] == 1: # >> TN
            TN.append(i)
        elif y_val[i][1] == y_pred[i][1] and y_val[i][1] == 1: # >> TP
            TP.append(i)
        elif y_val[i][0] != y_pred[i][0] and y_pred[i][0] == 1.: # >> FN
            FN.append(i)
        elif y_val[i][0] != y_pred[i][0] and y_pred[i][1] == 1.: # >> FP
            FP.append(i)
    
    # >> Preparing input data
    # inds = [TP[0], TN[0], FP[0], FN[0]]
    inds = [TN[0], FP[0], FN[0], TP[0]]
    X = x_val[inds]
    
    # >> define loss function
    def loss_0(output):
        return (output[0][0], output[1][0], output[2][0], output[3][0])
    def loss_1(output):
        return (output[0][1], output[1][1], output[2][1], output[3][1])    
    
    # >> Model-Modifier function to replace softmax activation function with
    # >> linear activation function
    def model_modifier(m):
        m.layers[-1].activation = tf.keras.activations.linear
        return m
    
    # -- vanilla saliency maps ------------------------------------------------
    
    # >> Create Saliency object
    saliency = Saliency(model, model_modifier=model_modifier, clone=False)
    
    # >> Generate saliency map
    saliency_map = saliency(loss_0, X)
    saliency_map = normalize(saliency_map)
    
    # >> Render
    f, ax = plt.subplots(2, 2)
    ax[0][0].set_title('True negative')
    ax[0][0].imshow(saliency_map[0], cmap='jet')
    ax[0][1].set_title('False positive')
    ax[0][1].imshow(saliency_map[1], cmap='jet')
    ax[1][0].set_title('False negative')
    ax[1][0].imshow(saliency_map[2], cmap='jet')
    ax[1][1].set_title('True positive')
    ax[1][1].imshow(saliency_map[3], cmap='jet')
    plt.tight_layout()
    plt.savefig(output_dir+model_name+'_saliency_non_ejecta.png', dpi=300)
    
    saliency_map = saliency(loss_1, X)
    saliency_map = normalize(saliency_map)
    
    # >> Render
    f, ax = plt.subplots(2, 2)
    ax[0][0].set_title('True positive')
    ax[0][0].imshow(saliency_map[0], cmap='jet')
    ax[0][1].set_title('True negative')
    ax[0][1].imshow(saliency_map[1], cmap='jet')
    ax[1][0].set_title('False positive')
    ax[1][0].imshow(saliency_map[2], cmap='jet')
    ax[1][1].set_title('False negative')
    ax[1][1].imshow(saliency_map[3], cmap='jet')
    plt.tight_layout()
    plt.savefig(output_dir+model_name+'_saliency_ejecta.png')
    
    titles = ['True positives', 'True negatives', 'False positives',
              'False negatives']
    suffixes = ['TP', 'TN', 'FP', 'FN']
    inds = [TP, TN, FP, FN]



    for i in range(4):

        
        # >> calculate saliency maps
        X = x_val[inds[i][:nrows*ncols]]
        
        def loss_1(output):
            res = []
            for i in range(len(X)):
                res.append(output[i][1])
            return res
        
        saliency_map = saliency(loss_1, X)
        saliency_map = normalize(saliency_map)
        fig, ax = plt.subplots(nrows, ncols)
        ax[0][ncols//2].set_title(titles[i])
        for row in range(min(nrows, len(X)//ncols)):
            for col in range(min(ncols, len(X)-ncols*row)):
                ax[row,col].imshow(saliency_map[row*ncols+col], cmap='jet')
        fig.tight_layout()
        fig.savefig(output_dir+model_name+'_saliency_'+suffixes[i]+'.png',
                    dpi=300)
        plt.close()
    
    # -- smoothGrad -----------------------------------------------------------
    
    saliency_map = saliency(loss_0,
                            X,
                            smooth_samples=20, # The number of calculating gradients iterations.
                            smooth_noise=0.20) # noise spread level.
    saliency_map = normalize(saliency_map)    
    f, ax = plt.subplots(2, 2)
    ax[0][0].set_title('True positive')
    ax[0][0].imshow(saliency_map[0], cmap='jet')
    ax[0][1].set_title('True negative')
    ax[0][1].imshow(saliency_map[1], cmap='jet')
    ax[1][0].set_title('False positive')
    ax[1][0].imshow(saliency_map[2], cmap='jet')
    ax[1][1].set_title('False negative')
    ax[1][1].imshow(saliency_map[3], cmap='jet')
    f.colorbar(cm.ScalarMappable(cmap='jet'))
    plt.tight_layout()
    plt.savefig(output_dir+model_name+'_smoothgrad_non_ejecta.png', dpi=300)

    saliency_map = saliency(loss_1,
                            X,
                            smooth_samples=20, # The number of calculating gradients iterations.
                            smooth_noise=0.20) # noise spread level.
    saliency_map = normalize(saliency_map)    
    f, ax = plt.subplots(2, 2)
    ax[0][0].set_title('True positive')
    ax[0][0].imshow(saliency_map[0], cmap='jet')
    ax[0][1].set_title('True negative')
    ax[0][1].imshow(saliency_map[1], cmap='jet')
    ax[1][0].set_title('False positive')
    ax[1][0].imshow(saliency_map[2], cmap='jet')
    ax[1][1].set_title('False negative')
    ax[1][1].imshow(saliency_map[3], cmap='jet')
    plt.tight_layout()
    plt.savefig(output_dir+model_name+'_smoothgrad_ejecta.png', dpi=300)
             
    # -- gradCAM --------------------------------------------------------------
    
    # gradcam = Gradcam(model,
    #                   model_modifier=model_modifier,
    #                   clone=False)
    
    # # Generate heatmap with GradCAM
    # cam = gradcam(loss_0,
    #               X,
    #               penultimate_layer=-1, # model.layers number
    #              )
    # cam = normalize(cam)    
    # f, ax = plt.subplots(2, 2)
    # ax[0][0].set_title('True positive')
    # ax[0][0].imshow(cam[0], cmap='jet')
    # ax[0][1].set_title('True negative')
    # ax[0][1].imshow(cam[1], cmap='jet')
    # ax[1][0].set_title('False positive')
    # ax[1][0].imshow(cam[2], cmap='jet')
    # ax[1][1].set_title('False negative')
    # ax[1][1].imshow(cam[3], cmap='jet')
    # plt.tight_layout()
    # plt.savefig(output_dir+model_name+'_gradcam_non_ejecta.png')
    # cam = gradcam(loss_1,
    #               X,
    #               penultimate_layer=-1, # model.layers number
    #              )
    # cam = normalize(cam)    
    # f, ax = plt.subplots(2, 2)
    # ax[0][0].set_title('True positive')
    # ax[0][0].imshow(cam[0], cmap='jet')
    # ax[0][1].set_title('True negative')
    # ax[0][1].imshow(cam[1], cmap='jet')
    # ax[1][0].set_title('False positive')
    # ax[1][0].imshow(cam[2], cmap='jet')
    # ax[1][1].set_title('False negative')
    # ax[1][1].imshow(cam[3], cmap='jet')
    # plt.tight_layout()
    # plt.savefig(output_dir+model_name+'_gradcam_ejecta.png')
                                   
    # # -- TP, FP, ... summary plots --------------------------------------------
    # >> use evaluation.py plot_FP_FN() [20201120]
    
    # x_val = x_val.astype('float32')
    # f, ax = plt.subplots(2, 2)
    # ax[0][0].set_title('True positive')
    # ax[0][0].imshow(x_val[TP[0]][:,:,0], cmap='gray')
    # ax[0][1].set_title('True negative')
    # ax[0][1].imshow(x_val[TN[0]][:,:,0], cmap='gray')
    # ax[1][0].set_title('False positive')
    # ax[1][0].imshow(x_val[FP[0]][:,:,0], cmap='gray')
    # ax[1][1].set_title('False negative')
    # ax[1][1].imshow(x_val[FN[0]][:,:,0], cmap='gray')
    # plt.tight_layout()
    # plt.savefig(output_dir+model_name+'_saliency_input.png')
    
    # fig, ax = plt.subplots(nrows, ncols)
    # ax[0][1].set_title('False negatives')
    # for i in range(nrows):
    #     for j in range(ncols):
    #         ax[i][j].imshow(x_val[FN[i*ncols+j]], cmap='gray')
    # plt.tight_layout()
    # plt.savefig(output_dir+model_name+'_FN.png')
    
    # fig, ax = plt.subplots(nrows, ncols)
    # ax[0][1].set_title('False positives')
    # for i in range(nrows):
    #     for j in range(ncols):
    #         ax[i][j].imshow(x_val[FP[i*ncols+j]], cmap='gray')
    # plt.tight_layout()
    # plt.savefig(output_dir+model_name+'_FP.png')
   
    # fig, ax = plt.subplots(nrows, ncols)
    # ax[0][1].set_title('True positives')
    # for i in range(nrows):
    #     for j in range(ncols):
    #         ax[i][j].imshow(x_val[TP[i*ncols+j]], cmap='gray')
    # plt.tight_layout()
    # plt.savefig(output_dir+model_name+'_TP.png')
    
    # fig, ax = plt.subplots(nrows, ncols)
    # ax[0][1].set_title('True negatives')
    # for i in range(nrows):
    #     for j in range(ncols):
    #         ax[i][j].imshow(x_val[TN[i*ncols+j]], cmap='gray')
    # plt.tight_layout()
    # plt.savefig(output_dir+model_name+'_TN.png')    
    
            
def plot_activation_max(model, bottleneck, bottleneck_ind, inds,
                       output_dir='./', model_name=''):
    out = visualize_activation(model, bottleneck_ind)
    fig, ax = plt.subplots()
    ax.imshow(np.squeeze(out, -1), cmap='gray')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.savefig(output_dir + model_name + '_act_max_all_filters.png', dpi=300)
    plt.close(fig)      
    
    latentDim = np.shape(bottleneck)[1]
    for i in range(latentDim):
        out = visualize_activation(model, bottleneck_ind, filter_indices=[i])
        fig, ax = plt.subplots()
        ax.imshow(np.squeeze(out, -1), cmap='gray')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.savefig(output_dir + model_name + '_act_max_filt'+str(i)+'.png',
                    dpi=300)
        plt.close(fig)            
        
def plot_saliency_corner_plot(model, bottleneck, bottleneck_ind, inds, 
                              output_dir='./', model_name=''):
    '''I currently run out of memory when I run this function :('''
    latentDim = np.shape(bottleneck)[1]
    for ind in inds:
        fig, axes = plt.subplots(nrows = latentDim, ncols = latentDim,
                                  figsize=(20,20))
        for i in range(latentDim):
            out = visualize_saliency(model, bottleneck_ind, [i], x_val[ind])
            axes[i,i].imshow(out, cmap='seismic')
            axes[i,i].set_aspect(aspect=1)
            
            for j in range(i):
                out = visualize_saliency(model, bottleneck_ind, [i,j], x_val[ind])
                axes[i,j].imshow(out, cmap='seismic')
                # >> remove axis frame of empty plots
                axes[latentDim-1-i, latentDim-1-j].axis('off')
    
            # >> x and y labels
            axes[i,0].set_ylabel('\u03C6'+str(i), fontsize='x-small')
            axes[latentDim-1,i].set_xlabel('\u03C6'+str(i), fontsize='x-small')
    
        # >> removing axis
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticklabels([])
        plt.subplots_adjust(hspace=0, wspace=0)
    
        plt.savefig(output_dir+model_name+'_saliency_corner_plot_ind'+str(ind)+'.png',
                    dpi=300)
        plt.close(fig)                

    
def hyperparam_opt_diagnosis(analyze_object, output_dir, model_name):
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)
    df = analyze_object.data
    
    with open(output_dir + 'best_params.txt', 'a') as f: 
        if model_name=='autoencoder':
            best_param_ind = np.argmin(df['val_loss'])
        else:
            best_param_ind = np.argmax(df['val_accuracy'])
        f.write(str(df.iloc[best_param_ind]) + '\n')
    
    # key_list = ['val_loss', 'val_accuracy']
    label_list = ['val_loss']
    key_list = ['val_loss']
        
    for i in range(len(label_list)):
        analyze_object.plot_line(key_list[i])
        plt.xlabel('round')
        plt.ylabel(label_list[i])
        plt.savefig(output_dir + model_name + '_'+label_list[i] + '_plot.png',
                    dpi=300)
    
    # >> kernel density estimation
    analyze_object.plot_kde('val_loss')
    plt.xlabel('val_loss')
    plt.ylabel('kernel density\nestimation')
    plt.savefig(output_dir + model_name + '_kde.png', dpi=300)
    
    analyze_object.plot_hist('val_loss', bins=50)
    plt.xlabel('val_loss')
    plt.ylabel('num observations')
    plt.tight_layout()
    plt.savefig(output_dir + model_name + '_hist_val_loss.png', dpi=300)
    
    # >> heat map correlation
    analyze_object.plot_corr('val_loss', ['acc', 'loss', 'val_acc'])
    plt.tight_layout()
    plt.savefig(output_dir + model_name + '_correlation_heatmap.png', dpi=300)
    
    # >> get best parameter set
    hyperparameters = list(analyze_object.data.columns)
    for col in ['round_epochs', 'val_loss', 'loss']:    
        hyperparameters.remove(col)
        
    p = {}
    for key in hyperparameters:
        p[key] = df.iloc[best_param_ind][key]
    
    return df, best_param_ind, p   
    
def correlation_heatmap(res, output_dir, model_name):            
    df = pd.DataFrame(res, columns=res.keys())
    print(df)
    corrMatrix = df.corr()
    print(corrMatrix)
    plt.figure(figsize=(15,15))
    sn.heatmap(corrMatrix, annot=True)
    plt.tight_layout()
    plt.savefig(output_dir + model_name + '_hyperparameter_correlation.png',
                dpi=300)

    return res

# =============================================================================
# 
# 20201115 - eval.py
# 
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sn

def epoch_plot(history, output_dir, model_name):
    epochs=len(history.history['loss'])
    fig, ax1 = plt.subplots()
    ax1.plot(history.history['loss'], 'b-', label='loss')
    ax1.plot(history.history['val_loss'], 'b--', label='val_loss')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax1.set_xticks(np.arange(0, epochs))
    ax1.tick_params('both', labelsize='x-small')
    ax1.legend()
    
    if 'accuracy' in list(history.history.keys()):
        ax2 = ax1.twinx()
        ax2.plot(history.history['accuracy'], 'r-', label='accuracy')
        ax2.plot(history.history['val_accuracy'], 'r--', label='val_accuracy')    
        ax2.set_ylabel('accuracy')
        ax2.legend(loc='lower left')
    fig.tight_layout()
    fig.savefig(output_dir + model_name + '_epoch.png')

def make_confusion_matrix(y_val, y_pred, output_dir='./', model_name=''):
    if len(y_val.shape) > 1:
        y_val = np.argmax(y_val, axis=-1)
        y_pred = np.argmax(y_pred, axis=-1)
    cm = confusion_matrix(y_val, y_pred)
    index = ['truth\nno ejecta','truth\nejecta preserved']
    columns = ['prediction\nno ejecta', 'prediction\nejecta preserved']
    df_cm = pd.DataFrame(cm, index=index, columns=columns)
    fig, ax = plt.subplots()
    sn.heatmap(df_cm, annot=True, fmt='g')
    ax.set_aspect(1)
    fig.savefig(output_dir + model_name + '_confusion_matrix.png')
    
    # >> save accuracy, recall and precision
    cm = np.array(cm).astype(np.float32)
    with open(output_dir + model_name + '_model_summary.txt', 'a') as f:
        f.write('\n')
        accuracy = (cm[0][0]+cm[1][1])/np.sum(cm)
        print('Accuracy: '+str(accuracy))
        f.write('Accuracy: '+str(accuracy)+'\n')
        recall = cm[1][1]/(cm[1][1]+cm[1][0])
        print('Recall: '+str(recall))
        f.write('Recall: '+str(recall)+'\n')
        precision = cm[1][1]/(cm[1][1]+cm[0][1])
        print('Precision: '+str(precision))
        f.write('Precision: '+str(precision)+'\n')
        
    return accuracy, recall, precision

def plot_tsne(x_val, y_val, bottleneck, output_dir='./', model_name=''):
    X = TSNE(n_components=2).fit_transform(bottleneck)
    if len(y_val.shape) > 1:
        y_val = np.argmax(y_val, axis=-1)
    colors=['r', 'b']
    plt.figure()
    plt.title('t-SNE')
    for i in range(2):
        class_inds = np.nonzero(y_val == i)
        if i == 0:
            label='No ejecta'
        else:
            label='With ejecta'
        plt.plot(X[class_inds][:,0], X[class_inds][:,1], '.', color=colors[i],
                 label=label)
        
    plt.legend()
    plt.savefig(output_dir + model_name + '_tsne.png')
    plt.close()

def plot_tsne_pred(x_val, y_val, y_pred, bottleneck, output_dir='./', model_name=''):
    X = TSNE(n_components=2).fit_transform(bottleneck)
    colors=['r', 'b']
    if len(y_val.shape) > 1:
        y_val = np.argmax(y_val, axis=-1)
    plt.figure()
    plt.title('t-SNE')
    for i in range(2):
        class_inds = np.nonzero(y_val == i)
        if i == 0:
            label='No ejecta (true)'
        else:
            label='With ejecta (true)'
        plt.plot(X[class_inds][:,0], X[class_inds][:,1], '.', color=colors[i],
                 label=label)
        
    plt.legend()
    plt.savefig(output_dir + model_name + '_true_tsne.png')
    plt.close()
    
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=-1)
    plt.figure()
    plt.title('t-SNE')
    for i in range(2):
        class_inds = np.nonzero(y_pred == i)
        if i == 0:
            label='No ejecta (predicted)'
        else:
            label='With ejecta (predicted)'
        plt.plot(X[class_inds][:,0], X[class_inds][:,1], '.', color=colors[i],
                 label=label)
        
    plt.legend()
    plt.savefig(output_dir + model_name + '_pred_tsne.png')
    plt.close()
    
def plot_FP_FN(x_val, y_val, y_pred, output_dir, name, nrows=3, ncols=3):
    # >> resize x_val, y_val and y_pred
    if len(y_val.shape) > 1:
        y_val = np.argmax(y_val, axis=-1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=-1)
    if x_val.shape[-1] == 3: # >> 3 identical channels (for pretrained models)
        x_val = x_val[:,:,:,0]
    
    # >> Find true postitives (TP), true negatives (TN), false positives (FP)
    # >> and false negatives (FN)
    TP, TN, FP, FN = [], [], [], []
    y_pred = np.round(y_pred)
    for i in range(len(x_val)):
        if y_val[i] == y_pred[i] and y_val[i] == 0: # >> TN
            TN.append(i)
        elif y_val[i] == y_pred[i] and y_val[i] == 1: # >> TP
            TP.append(i)
        elif y_val[i] != y_pred[i] and y_pred[i] == 0.: # >> FN
            FN.append(i)
        elif y_val[i] != y_pred[i] and y_pred[i] == 1.: # >> FP
            FP.append(i)    

    # >> plot success and fail summary
    x_val = x_val.astype('float32')
    f, ax = plt.subplots(2, 2)
    ax[0][0].set_title('True negative')
    ax[0][0].imshow(x_val[TN[0]], cmap='gray')
    ax[0][1].set_title('False positive')
    ax[0][1].imshow(x_val[FP[0]], cmap='gray')
    ax[1][0].set_title('False negative')
    ax[1][0].imshow(x_val[FN[0]], cmap='gray')
    ax[1][1].set_title('True positive')
    ax[1][1].imshow(x_val[TP[0]], cmap='gray')
    plt.tight_layout()
    plt.savefig(output_dir+name+'_TP_TN_FP_FN.png')
    plt.close(f)
    
    titles = ['True positives', 'True negatives', 'False positives',
              'False negatives']
    suffixes = ['TP', 'TN', 'FP', 'FN']
    inds = [TP, TN, FP, FN]
    for i in range(4):
        fig, ax = plt.subplots(nrows, ncols)
        ax[0][ncols//2].set_title(titles[i])
        for row in range(min(nrows, len(inds[i])//ncols)):
            for col in range(min(ncols,len(inds[i])-ncols*row)):
                ax[row,col].imshow(x_val[inds[i][row*ncols+col]], cmap='gray')
        fig.tight_layout()
        fig.savefig(output_dir+name+'_'+suffixes[i]+'.png')
        plt.close(fig)
        
        
        
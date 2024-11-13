import matplotlib.pyplot as plt
import os 
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_history(H, model_file):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,8))

    ax1.plot(H["train_acc"], label="train_acc")
    ax1.plot(H["val_acc"], label="val_acc")
    ax1.set_title('Model accuracy', fontsize=16)
    ax1.set_ylabel('Accuracy', fontsize=16)
    ax1.set_xlabel('Epoch', fontsize=16)
    ax1.legend(['Train', 'Val'], loc='upper left')

    ax2.plot(H["train_loss"], label="train_loss")
    ax2.plot(H["val_loss"], label="val_loss")
    ax2.set_title('Model loss', fontsize=16)
    ax2.set_ylabel('Loss', fontsize=16)
    ax2.set_xlabel('Epoch', fontsize=16)
    ax2.legend(['Train', 'Val'], loc='upper left')

    fig.tight_layout()
    fig.savefig(model_file+'.png', format='png', dpi=700)
    #fig.savefig(model_file+'.svg', format='svg', dpi=700)




def plot_confusion_matrix(target_labels, pred_labels, class_labels, path, save = False, name_png = None, fr = 'png', title='Confusion matrix'):
    confusion_mat = confusion_matrix(target_labels, pred_labels, normalize='true')
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)

    ax.set(xticks=np.arange(confusion_mat.shape[1]),
            yticks=np.arange(confusion_mat.shape[0]),
            xticklabels=class_labels, yticklabels=class_labels,
            xlabel='Predicted label',
            ylabel='True label',
            title='Confusion Matrix')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize = 20)
    plt.setp(ax.get_yticklabels(), fontsize=20)
    for i in range(confusion_mat.shape[0]):
        for j in range(confusion_mat.shape[1]):
            ax.text(j, i, format(confusion_mat[i, j] * 100, '.2f'),
                    ha="center", va="center", color="white" if confusion_mat[i, j] > np.max(confusion_mat) / 2 else "black",
                    fontsize=13)
    ax.set_xlabel('Predicted label', fontsize=20)  
    ax.set_ylabel('True label', fontsize=20) 
    ax.set_title(title, fontsize=20)  
    #fig.axis('off')
    fig.tight_layout()
    if save and name_png != None:
      fig.savefig(os.path.join(path, name_png + '.' + fr), format=fr, dpi=700)



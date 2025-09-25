import torch
import os
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

from scripts import models
from scripts import utility
from scripts import utility

FINETUNE_BATCH_SIZE = 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', type=str, help='Directory of the full dataset.')
    parser.add_argument('-log_dir', type=str, help='Directory for logging results.')
    parser.add_argument('-seed', type=int, default=21)
    parser.add_argument('-model_in_ensemble', nargs='*', type=str, default=['vit', 'swin'],
                        help='List of models to include in the ensemble. Default is [\'vit\', \'swin\']. Use in the following way: --myarg \'vit\' \'swin\' \'convnext\'')
    
    return parser.parse_args()


def main():
    args = parse_args()
    exp_path = os.path.join(args.log_dir,  str(args.seed))
    model_path = os.path.join(exp_path, 'models_dir')
    weights_dir = os.path.join(exp_path, 'model_weights')
    dest = os.path.join(exp_path, 'logits_dir')
    if not os.path.exists(dest):
        os.makedirs(dest)


    X_Train = np.load(os.path.join(args.data_dir, 'training_images.npy'))
    Y_Train = np.load(os.path.join(args.data_dir, 'training_labels.npy'))

    X_train, X_val, Y_train, Y_val = train_test_split(X_Train, Y_Train, test_size=0.2, random_state=args.seed, stratify = Y_Train)
    train_dataset = utility.FBD(X_train, Y_train, transform=utility.trainTransform)
    val_dataset = utility.FBD(X_val, Y_val, transform = utility.valTransform)
    train = torch.utils.data.DataLoader(train_dataset, batch_size=FINETUNE_BATCH_SIZE, shuffle=False)
    val = torch.utils.data.DataLoader(val_dataset, batch_size=FINETUNE_BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_list = []

    if 'swin' in args.model_in_ensemble:
        swin_model = torch.load(os.path.join(model_path, 'swin.pt'))
        swin_model.load_state_dict(torch.load(os.path.join(weights_dir,  str(args.seed) + "_" + "swin.pt")))
        model_list.append(swin_model)

    if 'convnext' in args.model_in_ensemble:
        convnext_model = torch.load(os.path.join(model_path, 'convnext.pt'))
        convnext_model.load_state_dict(torch.load(os.path.join(weights_dir,  str(args.seed) + "_" + "convnext.pt")))
        model_list.append(convnext_model)

    if 'vit' in args.model_in_ensemble:    
        vit_model = torch.load(os.path.join(model_path, 'vit.pt'))
        vit_model.load_state_dict(torch.load(os.path.join(weights_dir, str(args.seed) + "_" + "vit.pt")))
        model_list.append(vit_model)

    if 'swin_t' in args.model_in_ensemble:    
        swin_t_model = torch.load(os.path.join(model_path, 'swin_t.pt'))
        swin_t_model.load_state_dict(torch.load(os.path.join(weights_dir,  str(args.seed) + "_" + "swin_t.pt")))
        model_list.append(swin_t_model)

    if 'resnet' in args.model_in_ensemble:
        resnet_model = torch.load(os.path.join(model_path, 'resnet.pt'))
        resnet_model.load_state_dict(torch.load(os.path.join(weights_dir,  str(args.seed) + "_" + "resnet.pt")))
        model_list.append(resnet_model)

    if 'densenet' in args.model_in_ensemble:   
        densenet_model = torch.load(os.path.join(model_path, 'densenet.pt'))
        densenet_model.load_state_dict(torch.load(os.path.join(weights_dir, str(args.seed) + "_" + "densenet.pt")))
        model_list.append(densenet_model)

    if 'vgg' in args.model_in_ensemble:   
        vgg_model = torch.load(os.path.join(model_path, 'vgg.pt'))
        vgg_model.load_state_dict(torch.load(os.path.join(weights_dir, str(args.seed) + "_" + "vgg.pt")))
        model_list.append(vgg_model)

    model_dict = {
            #convnext_model: "convnext",
            swin_model : "swin",
            vit_model : "vit",
            #swin_t_model: 'swin_t',
            #resnet_model : "resnet",
            #densenet_model : "densenet",
            #vgg_model : "vgg"
    }


    combo = ""   
    for m in model_list:
        combo += model_dict[m] + '-'
    combo = combo[:-1]
    print("Models in ensemble: ", combo)
    
    mods = [m.to(device) for m in model_list]
    ensemble = models.Ensemble(mods)
    ensemble = ensemble.to(device)
    ensemble.eval()


    train_logits = []
    train_labels = []
    val_logits = []
    val_labels = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = ensemble(inputs)      
            train_logits.append(outputs.to('cpu'))
            train_labels.append(labels.to('cpu'))


    train_logits = torch.cat(train_logits, dim=0).numpy()  # Shape: (total_samples, 2, 4)
    train_labels = torch.cat(train_labels, dim=0).numpy()  # Shape: (total_samples,)


    np.save(os.path.join(dest, str(args.seed) + '_' + combo + '_train_logits.npy'), np.array(train_logits))
    np.save(os.path.join(dest, str(args.seed) + '_' + combo + '_train_labels.npy'), Y_train)


    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = ensemble(inputs)
            
            val_logits.append(outputs.to('cpu'))
            val_labels.append(labels.to('cpu'))

    val_logits = torch.cat(val_logits, dim=0).numpy()  # Shape: (total_samples, 2, 4)
    val_labels = torch.cat(val_labels, dim=0).numpy()  # Shape: (total_samples,)

    np.save(os.path.join(dest, str(args.seed) + '_' + combo + '_val_logits.npy'), np.array(val_logits))
    np.save(os.path.join(dest, str(args.seed) + '_' + combo + '_val_labels.npy'), Y_val)



if __name__ == '__main__':
    main()
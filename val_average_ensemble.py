import torch
import os
import numpy as np
import argparse

from scripts import models
from scripts import utility

FINETUNE_BATCH_SIZE = 8

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', type=str, help='Directory of the full dataset.')
    parser.add_argument('-log_dir', type=str, help='Directory for logging results.')
    parser.add_argument('-max_num_epochs', type=int, default=200, help='Max number of epochs. Default is 200.')
    parser.add_argument('-model_name', type=str, default='vit', help='Model that you want train. Default is vit.')
    parser.add_argument('-optimizer', type=str, default='adam')
    parser.add_argument('-scheduler', type=str, default='plateau')
    parser.add_argument('-lr', type=float, default=0.000001)
    parser.add_argument('-weight_decay', type=float, default=0.1)
    
    return parser.parse_args()



def main():
    args = parse_args()

    exp_path = os.path.join(args.log_dir, args.optimizer + '_' + args.scheduler + '_' + str(args.weight_decay) + '_' + str(args.lr),  str(args.seed))    
    model_path = os.path.join(exp_path, "models_dir")
    weights_dir = os.path.join(exp_path, "model_weights")
    logits_dir = os.path.join(exp_path, 'logits_dir')
    
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

    combo = ''
    for m in model_list:
        combo += model_dict[m] + '-'
    combo = combo[:-1]
    print("Models in ensemble: ", combo)

    val_logits = np.load(os.path.join(logits_dir, str(args.seed) + '_' + combo + '_val_logits.npy'))
    val_labels = np.load(os.path.join(logits_dir, str(args.seed) + '_' + combo + '_val_labels.npy'))
    val_dataset = utility.LogitsDataset(val_logits, val_labels)
    val = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=FINETUNE_BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)
    del val_dataset, val_logits, val_labels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_list = [m.to(device)for m in model_list]

    val_accuracy, _, _ = models.average_ensemble_logits(val)
    msg = '\nModels in ensemble: ' + combo + '\nVal Accuracy: '+ str(val_accuracy) + '%\n'
    print(msg)
    with open(exp_path + '/' + str(args.seed) + '_' + m + '_average_ensemble_training_results.txt', 'w') as file:
        file.write('Model id: ' + str(args.seed) +'\n' + msg + '\n') 



if __name__ == '__main__':
    main()
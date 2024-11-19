import torch
import os
import numpy as np
from sklearn.metrics import classification_report
import json
import argparse

from scripts import models
from scripts import utility
from scripts import visualize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', type=str, help='Directory of the full dataset.')
    parser.add_argument('-log_dir', type=str, help='Directory for logging results.')
    parser.add_argument('-seed', type=int, default=21)
    parser.add_argument('-max_num_epochs', type=int, default=200, help='Max number of epochs. Default is 200.')
    parser.add_argument('-optimizer', type=str, default='adam')
    parser.add_argument('-scheduler', type=str, default='plateau')
    parser.add_argument('-lr', type=float, default=0.000001)
    parser.add_argument('-weight_decay', type=float, default=0.01)
    parser.add_argument('-model_in_ensemble', nargs='*', type=str, default=['vit', 'swin'],
                        help='List of models to include in the ensemble. Default is [\'vit\', \'swin\']. Use in the following way: --myarg \'vit\' \'swin\' \'convnext\'')
    
    return parser.parse_args()



def main():
    args = parse_args()

    exp_path = os.path.join(args.log_dir,  str(args.seed))
    model_path = os.path.join(exp_path, "models_dir")
    weights_dir = os.path.join(exp_path, "model_weights")
    average_ensemble_plots = os.path.join(exp_path, "average_ensemble_plots")
    class_labels = ["Spalling", "Crack", "Stain", "Vegetation"]
    label_names = range(len(class_labels))

    if not os.path.exists(average_ensemble_plots):
        os.makedirs(average_ensemble_plots)


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_list = [m.to(device)for m in model_list]

    X_test = np.load(os.path.join(args.data_dir, "validation_images.npy"))
    Y_test = np.load(os.path.join(args.data_dir,"validation_labels.npy"))
    test_dataset = utility.FBD(X_test, Y_test, transform = utility.valTransform)
    test = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    
    ensemble_model = models.Ensemble(model_list)
    ensemble_model = ensemble_model.to(device)
    ensemble_model.eval()
    test_logits = []
    test_labels = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = ensemble_model(inputs)
            
            test_logits.append(outputs.to('cpu'))
            test_labels.append(labels.to('cpu'))

    test_logits = torch.cat(test_logits, dim=0).numpy()  
    test_labels = torch.cat(test_labels, dim=0).numpy()

    
    del ensemble_model, X_test, Y_test, test_dataset, test

    test_dataset = utility.LogitsDataset(test_logits, test_labels)
    test = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)

    combo = ''
    for m in model_list:
        combo += model_dict[m] + '-'
    combo = combo[:-1]
    print("combo: ", combo)
    

    accuracy, preds, targets = models.average_ensemble_logits(test)

    print(len(preds))
    print(len(targets))
    print("Test accuracy: ", accuracy)
    visualize.plot_confusion_matrix(targets, preds, class_labels, average_ensemble_plots, save = True, name_png = combo, fr='png', title=str(args.seed) + '_' + combo)
    class_report =  classification_report(targets, preds, labels=label_names, output_dict=True)
    results = {'accuracy': accuracy}
    eval_list = [results, class_report]
    with open(exp_path + '/' + str(args.seed) + '_' + combo + '_' + 'average_ensemble_metrics.json', 'w+') as file:
        json.dump(eval_list, file, indent=2)

if __name__ == '__main__':
    main()
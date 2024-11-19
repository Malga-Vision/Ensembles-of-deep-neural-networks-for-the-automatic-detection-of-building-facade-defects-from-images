import torch
import os
import numpy as np
import argparse
from torch.utils.data import WeightedRandomSampler
import time
from sklearn.metrics import classification_report
import json

from scripts import models
from scripts import visualize
from scripts import utility

FINETUNE_BATCH_SIZE = 8
PRED_BATCH_SIZE = 2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', type=str, help='Directory of the full dataset.')
    parser.add_argument('-log_dir', type=str, help='Directory for logging results.')
    parser.add_argument('-max_num_epochs', type=int, default=200, help='Max number of epochs. Default is 200.')
    parser.add_argument('-seed', type=int, default=21)
    parser.add_argument('-optimizer', type=str, default='adam')
    parser.add_argument('-scheduler', type=str, default='plateau')
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-weight_decay', type=float, default=0.01)
    parser.add_argument('-model_in_ensemble', nargs='*', type=str, default=['vit', 'swin'],
                        help='List of models to include in the ensemble. Default is [\'vit\', \'swin\']. Use in the following way: --myarg \'vit\' \'swin\' \'convnext\'')
    
    return parser.parse_args()


def main():
    args = parse_args()

    #exp_path = os.path.join(args.log_dir, args.optimizer + '_' + args.scheduler + '_' + str(args.weight_decay) + '_' + str(args.lr),  str(args.seed))
    exp_path = os.path.join(args.log_dir, str(args.seed))
    model_path = os.path.join(exp_path, "models_dir")
    weights_dir = os.path.join(exp_path, "model_weights")
    logits_dir = os.path.join(exp_path, 'logits_dir')
    learnable_ensemble_plots = os.path.join(exp_path, "learnable_ensemble_plots")
    learnable_ensemble_curves = os.path.join(exp_path, "learnable_ensemble_curves")
    ensemble_weights_path = os.path.join(exp_path, "learnable_ensemble_weights")
    class_labels = ["Spalling", "Crack", "Stain", "Vegetation"]
    label_names = range(len(class_labels))

    if not os.path.exists(learnable_ensemble_plots):
        os.makedirs(learnable_ensemble_plots)

    if not os.path.exists(learnable_ensemble_curves):
        os.makedirs(learnable_ensemble_curves)
    
    if not os.path.exists(ensemble_weights_path):
        os.makedirs(ensemble_weights_path)

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
    model_list = [m.to(device) for m in model_list]

    combo = ''
    for m in model_list:
        combo += model_dict[m] + '-'
    combo = combo[:-1]


    ############################# TRAINING #######################
    train_logits = np.load(os.path.join(logits_dir, str(args.seed) + '_' + combo + '_train_logits.npy'))
    train_labels = np.load(os.path.join(logits_dir, str(args.seed) + '_' + combo + '_train_labels.npy'))
    train_dataset = utility.LogitsDataset(train_logits, train_labels)
    sample_per_class =  utility.count_samples_per_class(train_dataset, len(class_labels))
    print("Train sample per class: ", sample_per_class)
    class_weights = 1. / sample_per_class
    y_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    samples_weight = np.array([class_weights[t] for t in y_labels])
    sampler = WeightedRandomSampler(weights = samples_weight, num_samples=len(train_dataset), replacement=False)
    train = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=FINETUNE_BATCH_SIZE, sampler = sampler, pin_memory=True, num_workers=4)

    val_logits = np.load(os.path.join(logits_dir, str(args.seed) + '_' + combo + '_val_logits.npy'))
    val_labels = np.load(os.path.join(logits_dir, str(args.seed) + '_' + combo + '_val_labels.npy'))
    val_dataset = utility.LogitsDataset(val_logits, val_labels)
    val = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=FINETUNE_BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)
    del val_dataset, train_dataset, y_labels, samples_weight, train_logits, train_labels, val_logits, val_labels

    ensemble_model = models.LearnableConvEnsemble(len(model_list))
    
    model_file = os.path.join(os.path.join(ensemble_weights_path, str(args.seed) + '_' + combo + '.pt'))
    start_time = time.time()
    H, best_epoch, train_acc, val_acc, val_loss, duration = models.training(args.max_num_epochs, ensemble_model, train, val, model_file, args.lr, opt=args.optimizer, sched=args.scheduler)
    train_duration = float(time.time() - start_time)
    print("Train duration (minutes): ", int(np.ceil(train_duration / 60)))

    visualize.plot_history(H, os.path.join(learnable_ensemble_curves, str(args.seed)+'_'+combo)) 

    print('#' * 50 + '  At the end  ' + '#' * 50)
    print(f' Best Train Accuracy: {train_acc :.3f}')
    print(f' Best Validation Loss:     {val_loss :.3f}')
    print(f' Best Validation Accuracy: {val_acc :.3f}')

    msg2 = 'Combination: ' + combo + '\nTraining Accuracy: ' + str(train_acc) + '%\nValidation Accuracy: '+ str(val_acc) + '%\nValidation loss: ' + str(val_loss) +"\nTraining time: " + str(duration) + "\n"

    with open(exp_path + '/' + str(args.seed) + '_' + combo + '_learnable_ensemble_training_results.txt', 'w') as file:
        file.write('Model id: ' + str(args.seed) +'\n' + msg2 + '\n') 

    del train, val

    ############################# TEST #######################
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

    ensemble_model = models.LearnableConvEnsemble(len(model_list))
    ensemble_model.load_state_dict(torch.load(os.path.join(ensemble_weights_path, str(args.seed) + '_' + combo + '.pt')))

    accuracy, preds, targets = models.evaluation_ensemble(ensemble_model, test)
    print("Test accuracy: ", accuracy)
    visualize.plot_confusion_matrix(targets, preds, class_labels, learnable_ensemble_plots, save = True, name_png = combo, fr='png', title=str(args.seed) + '_' + combo)
    class_report =  classification_report(targets, preds, labels=label_names, output_dict=True)


    print('Test accuracy: ' + str(accuracy))
    results = {'accuracy': accuracy}
    eval_list = [results, class_report]
    with open(exp_path + '/' + str(args.seed) + '_' + combo + '_' + 'learnable_ensemble_metrics.json', 'w+') as file:
        json.dump(eval_list, file, indent=2)

if __name__ == '__main__':
    main()
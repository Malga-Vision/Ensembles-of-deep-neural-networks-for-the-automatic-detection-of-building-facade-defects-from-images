import torch
import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler
import gc
from sklearn.metrics import classification_report
import json
import time
import argparse

from scripts import models
from scripts import utility
from scripts import visualize


FINETUNE_BATCH_SIZE = 8
PRED_BATCH_SIZE = 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', type=str, help='Directory of the full dataset.')
    parser.add_argument('-log_dir', type=str, help='Directory for logging results.')
    parser.add_argument('-seed', type=int, default=21)
    parser.add_argument('-model_id', type=int, default=1)
    parser.add_argument('-max_num_epochs', type=int, default=200, help='Max number of epochs. Default is 200.')
    parser.add_argument('-model_name', type=str, default='vit', help='Model that you want train. Default is vit.')
    parser.add_argument('-optimizer', type=str, default='adam')
    parser.add_argument('-scheduler', type=str, default='plateau')
    parser.add_argument('-lr', type=float, default=0.000001)
    parser.add_argument('-weight_decay', type=float, default=0.01)
    parser.add_argument('-dropout', type=float, default=0.5)
    
    return parser.parse_args()




def main():
    args = parse_args()
    class_labels = ["Spalling", "Crack", "Stain", "Vegetation"]
    
    num_classes = len(class_labels)
    label_names = range(num_classes)
    m = args.model_name
    
    exp_path = args.log_dir + '_' + args.optimizer + '_' + args.scheduler + '_' + str(args.weight_decay) + '_' + str(args.dropout) + '_' + str(args.lr) + '_' + str(args.seed)
    models_path = os.path.join(exp_path, "models_dir")
    weights_path = os.path.join(exp_path, "model_weights")
    plots_path = os.path.join(exp_path, "plots")
    curves_path = os.path.join(exp_path, "curves")

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    if not os.path.exists(curves_path):
        os.makedirs(curves_path)

    X_Train = np.load(os.path.join(args.data_dir, 'training_images.npy'))
    Y_Train = np.load(os.path.join(args.data_dir, 'training_labels.npy'))
    X_Test = np.load(os.path.join(args.data_dir, 'validation_images.npy'))
    Y_Test = np.load(os.path.join(args.data_dir, 'validation_labels.npy'))

    test_dataset = utility.FBD(X_Test, Y_Test, transform = utility.valTransform)
    test = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=PRED_BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)
    del test_dataset, X_Test, Y_Test


    msg0 = m + " id" + str(args.model_id)
    print(msg0)

    X_train, X_val, Y_train, Y_val = train_test_split(X_Train, Y_Train, test_size=0.2, random_state=args.seed, stratify=Y_Train)
    train_dataset = utility.FBD(X_train, Y_train, transform=utility.trainTransform)
    val_dataset = utility.FBD(X_val, Y_val, transform = utility.valTransform)

    sample_per_class =  utility.count_samples_per_class(train_dataset, num_classes)
    class_weights = 1. / sample_per_class
    y_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    samples_weight = np.array([class_weights[t] for t in y_labels])
    sampler = WeightedRandomSampler(weights = samples_weight, num_samples=len(train_dataset), replacement=False)

    train = torch.utils.data.DataLoader(dataset=train_dataset, sampler = sampler, batch_size=FINETUNE_BATCH_SIZE, pin_memory=True, num_workers=4)
    val = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=FINETUNE_BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)

    del train_dataset, val_dataset, X_Train, X_val, Y_Train, Y_val

    model_file = str(args.model_id) + '_' + m + '.pt'
    print("Training ", m)
    
    model = models.load_model(m, args.dropout, num_classes, models_path)
    model_weight_path = os.path.join(weights_path, model_file)
    start_time = time.time()
    history, best_epoch, train_acc, val_acc, val_loss, time_epoch = models.training(args.max_num_epochs, 
                                                                                    model, 
                                                                                    train, 
                                                                                    val, 
                                                                                    model_weight_path, 
                                                                                    args.lr, 
                                                                                    opt=args.optimizer, 
                                                                                    sched=args.scheduler, 
                                                                                    wd=args.weight_decay)
    train_duration = float(time.time() - start_time)
    msg2 = 'Training Accuracy: ' + str(train_acc) + '%\nValidation Accuracy: '+ str(val_acc) + '%\nValidation loss: ' + str(val_loss) +"\nTraining time: " + str(time_epoch) + "\n"
    print("Train duration (minutes): ", int(np.ceil(train_duration / 60)))
    print(msg2)

    visualize.plot_history(history, os.path.join(curves_path, model_file[:-3]))

    with open(exp_path + '/' + str(args.model_id) + '_' + m + '_training_results.txt', 'w') as file:
        file.write(msg0 + "\n" + m + '\n' + msg2 + '\n' + "Train duration (minutes): " + str(int(np.ceil(train_duration / 60))))
    
    ####################### TEST PHASE ##########################
    model = torch.load(os.path.join(models_path, m + '.pt'))
    model.load_state_dict(torch.load(os.path.join(weights_path, model_file)))

    test_accuracy, preds, targets = models.evaluation_model(model, test)
    visualize.plot_confusion_matrix(targets, preds, class_labels, plots_path, save = True, name_png = args.model_id + '_' + args.model_name, fr='png', title=str(args.model_id) + '_' + args.model_name)
    class_report =  classification_report(targets, preds, labels=label_names, output_dict=True)
    print('Test accuracy: ' + str(test_accuracy))
    results = {'accuracy': test_accuracy}
    eval_list = [results, class_report]
    with open(exp_path + '/' + str(args.model_id) + '_' + m + '_' + 'metrics.json', 'w+') as file:
        json.dump(eval_list, file, indent=2)

    gc.collect()
    torch.cuda.empty_cache()

    del model, history


if __name__ == '__main__':
    main()
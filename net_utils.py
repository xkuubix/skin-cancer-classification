import time
import torch
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, roc_auc_score

def one_hot(x, length):
    batch_size = x.size(0)
    x_one_hot = torch.zeros(batch_size, length, dtype=torch.float)
    for i in range(batch_size):
        x_one_hot[i, x[i]] = torch.tensor(1.0, dtype=torch.float)
    return x_one_hot

def train_net(net, train_dl, val_dl, criterion, optimizer, config, device, run=None, fold_index=None):
    net.to(device)
    best_val_loss = float('inf')
    best_model = None
    patience = config['net_train']['patience']
    early_stop_counter = 0
    start_time = time.time()

    for epoch in range(config['net_train']['epochs']):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_dl):
            if config['dataset']['mode'] in ['images', 'hybrid']:  
                image = data['image'].to(device)
            if config['dataset']['mode'] in ['radiomics', 'hybrid']:
                radiomic_features = data['features'].to(device)
            else:
                radiomic_features = None
            target_idx = data['label'].to(device)
            optimizer.zero_grad()
                
            if config['dataset']['mode'] in ['hybrid']:
                outputs = net(image, radiomic_features)
            elif config['dataset']['mode'] in ['images']:
                outputs = net(image)
            elif config['dataset']['mode'] in ['radiomics']:
                outputs = net(radiomic_features)

            if config['net_train']['criterion'] == 'bce':
                predicted = torch.round(torch.sigmoid(outputs.reshape(-1)))
                outputs = torch.sigmoid(outputs).reshape(-1)
            elif config['net_train']['criterion'] == 'ce':
                _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, target_idx)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # predicted = outputs.logits.argmax(-1)
            total += target_idx.size(0)
            correct += (predicted == target_idx).sum().item()
        # Validation
        net.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for i, data in enumerate(val_dl):
                if config['dataset']['mode'] in ['images', 'hybrid']:  
                    image = data['image'].to(device)
                if config['dataset']['mode'] in ['radiomics', 'hybrid']:
                    radiomic_features = data['features'].to(device)
                else:
                    radiomic_features = None
                target_idx = data['label'].to(device)
                
                if config['dataset']['mode'] in ['hybrid']:
                    outputs = net(image, radiomic_features)
                elif config['dataset']['mode'] in ['images']:
                    outputs = net(image)
                elif config['dataset']['mode'] in ['radiomics']:
                    outputs = net(radiomic_features)

                if config['net_train']['criterion'] == 'bce':
                    predicted = torch.round(torch.sigmoid(outputs.reshape(-1)))
                    outputs = torch.sigmoid(outputs).reshape(-1)
                elif config['net_train']['criterion'] == 'ce':
                    _, predicted = torch.max(outputs.data, 1)
                val_loss = criterion(outputs, target_idx)
                running_val_loss += val_loss.item()
                val_total += target_idx.size(0)
                val_correct += (predicted == target_idx).sum().item()

        avg_train_loss = running_loss / len(train_dl)
        avg_val_loss = running_val_loss / len(val_dl)
        train_accuracy = correct / total
        val_accuracy = val_correct / val_total
        if run:
            run[f"FOLD-{fold_index}/training/accuracy"].append(train_accuracy)
            run[f"FOLD-{fold_index}/validation/accuracy"].append(val_accuracy)
            run[f"FOLD-{fold_index}/training/loss"].append(avg_train_loss)
            run[f"FOLD-{fold_index}/validation/loss"].append(avg_val_loss)
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Epoch {epoch+1 :3}/{config['net_train']['epochs']}, train-loss: {avg_train_loss:.3f}, val-loss: {avg_val_loss:.3f}, train-acc: {train_accuracy:.3f}, val-acc: {val_accuracy:.3f}", end=' ')
        print(f"Time taken: {total_time//60:3.0f}m {total_time%60:02.0f}s")

        # Early stopping (min loss or max recall)
        # recall = recall_score(target_idx.cpu(), predicted.cpu(), average='weighted')
        if avg_val_loss < best_val_loss:
        # if recall > best_val_metric:
            best_val_loss = avg_val_loss
            # best_val_metric = recall
            best_model = net.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered!")
                break
        if run:
            run['patience_counter'] = patience - early_stop_counter
    
    print('Finished Training')
    return best_model

def test_net(net, test_dl, config, device, mapping_handler, run=None, fold_index=None):
    net.eval()
    correct = 0
    total = 0
    predicted_targets = []
    true_targets = []
    probabilities = []
    with torch.no_grad():
        for data in test_dl:
            if config['dataset']['mode'] in ['images', 'hybrid']:  
                image = data['image'].to(device)
            if config['dataset']['mode'] in ['radiomics', 'hybrid']:
                radiomic_features = data['features'].to(device)
            else:
                radiomic_features = None
            target_idx = data['label'].to(device)

            if config['dataset']['mode'] in ['hybrid']:
                outputs = net(image, radiomic_features)
            elif config['dataset']['mode'] in ['images']:
                outputs = net(image)
            elif config['dataset']['mode'] in ['radiomics']:
                outputs = net(radiomic_features)

            if config['net_train']['criterion'] == 'bce':
                predicted = torch.round(torch.sigmoid(outputs.reshape(-1)))
                probs = torch.sigmoid(outputs).reshape(-1)
            elif config['net_train']['criterion'] == 'ce':
                _, predicted = torch.max(outputs.data, 1)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
            probabilities.extend(probs)
            predicted = predicted.to(device)
            total += target_idx.size(0)
            correct += (predicted == target_idx).sum().item()
            predicted_targets.extend(predicted.tolist())
            true_targets.extend(target_idx.tolist())
            # probabilities.append(outputs.tolist())
    accuracy = correct / total
    balanced_accuracy = balanced_accuracy_score(true_targets, predicted_targets)
    print(f"Accuracy: {accuracy}")
    print('Classification Report:')
    print(classification_report(true_targets, predicted_targets, target_names=mapping_handler.keys()))
    print('Confusion Matrix:')
    print(confusion_matrix(true_targets, predicted_targets))
    
    print('Finished Testing')
    report = classification_report(true_targets, predicted_targets, output_dict=True, target_names=mapping_handler.keys())
    # print("TO DO: ROC AUC for single and multiclass")

    roc_all = roc_auc_score(true_targets, probabilities,
                            multi_class='ovr', average=None)
    
    roc_micro = roc_auc_score(true_targets, probabilities,
                              multi_class='ovr', average="micro")
    for item in range(len(mapping_handler.keys())):
        label = list(mapping_handler.keys())[item]
        report[label]['roc_auc'] = roc_all[item]

    report['weighted avg']["roc_auc"] = roc_auc_score(true_targets, probabilities,
                                                      multi_class='ovr', average="weighted")
    report['macro avg']['roc_auc'] = roc_auc_score(true_targets, probabilities,
                                                   multi_class='ovr', average="macro")
    report['roc_auc_micro'] = roc_micro
    report['balanced_accuracy'] = balanced_accuracy
    

    if run:
        run[f"FOLD-{fold_index}/test/report"].assign(report)

    return report
# %%


def get_proba(net, test_dl, config, device):
    net.eval()
    probabilities = []
    with torch.no_grad():
        for data in test_dl:
            if config['dataset']['mode'] in ['images', 'hybrid']:  
                image = data['image'].to(device)
            if config['dataset']['mode'] in ['radiomics', 'hybrid']:
                radiomic_features = data['features'].to(device)
            else:
                radiomic_features = None
            target_idx = data['label'].to(device)

            if config['dataset']['mode'] in ['hybrid']:
                outputs = net(image, radiomic_features)
            elif config['dataset']['mode'] in ['images']:
                outputs = net(image)
            elif config['dataset']['mode'] in ['radiomics']:
                outputs = net(radiomic_features)

            if config['net_train']['criterion'] == 'bce':
                predicted = torch.round(torch.sigmoid(outputs.reshape(-1)))
                probs = torch.sigmoid(outputs).reshape(-1)
            elif config['net_train']['criterion'] == 'ce':
                _, predicted = torch.max(outputs.data, 1)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
            probabilities.extend(probs)
    return probabilities
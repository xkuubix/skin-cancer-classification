import time
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix


def one_hot(x, length):
    batch_size = x.size(0)
    x_one_hot = torch.zeros(batch_size, length, dtype=torch.float)
    for i in range(batch_size):
        x_one_hot[i, x[i]] = torch.tensor(1.0, dtype=torch.float)
    return x_one_hot

def train_net(net, train_dl, val_dl, criterion, optimizer, n_classes, config, device):
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
            image = data['image']
            # print(torch.mean(image, dim=0), torch.std(image, dim=0))
            image = image.to(device)
            target_idx = data['label']
            target_idx = target_idx.to(device)
            target = one_hot(target_idx, n_classes).to(device)
            optimizer.zero_grad()
            outputs = net(image)
            loss = net.loss(outputs, target, size_average=True)
            # -------------------
            import torch.nn as nn
            # loss_fn = nn.MultiMarginLoss(p=2, margin=0.9)
            # print(f'{target_idx.shape=}')
            # print(f'{outputs.shape=}')
            # v_mag = torch.sqrt(torch.sum(outputs**2, dim=2, keepdim=True))
            # print(f'{v_mag.shape=}')
            # print(f'{loss=}')
            # loss = loss_fn(v_mag.squeeze(), target_idx)
            # print(f'{l=}')
            # -------------------
            # return
            # loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            predicted = net.predict(outputs).to(device)
            # _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target_idx).sum().item()
        # Validation
        net.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for i, data in enumerate(val_dl):
                image = data['image']
                image = image.to(device)
                target_idx = data['label']
                target_idx = target_idx.to(device)
                target = one_hot(target_idx, n_classes).to(device)
                outputs = net(image)
                # val_loss = criterion(outputs, labels)
                val_loss = net.loss(outputs, target, size_average=True)
                # -------------------
                # import torch.nn as nn
                # loss_fn = nn.MultiMarginLoss(p=2, margin=0.9)
                # print(f'{target_idx.shape=}')
                # print(f'{outputs.shape=}')
                # v_mag = torch.sqrt(torch.sum(outputs**2, dim=2, keepdim=True))
                # print(f'{v_mag.shape=}')
                # print(f'{loss=}')
                # val_loss = loss_fn(v_mag.squeeze(), target_idx)
                # print(f'{l=}')
                # -------------------
                running_val_loss += val_loss.item()

                predicted = net.predict(outputs).to(device)
                # _, predicted = torch.max(outputs.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target_idx).sum().item()
        
        avg_train_loss = running_loss / len(train_dl)
        avg_val_loss = running_val_loss / len(val_dl)
        train_accuracy = correct / total
        val_accuracy = val_correct / val_total

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Epoch {epoch+1}, train-loss: {avg_train_loss:.3f}, val-loss: {avg_val_loss:.3f}, train-accuracy: {train_accuracy:.3f}, val-accuracy: {val_accuracy:.3f}", end=' ')
        print(f"Time taken: {total_time//60:4.0f}:{total_time%60:2.0f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = net.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered!")
                break
    
    print('Finished Training')
    return best_model

def test_net(net, test_dl, n_classes, device):
    net.eval()
    correct = 0
    total = 0
    predicted_targets = []
    true_targets = []
    with torch.no_grad():
        for data in test_dl:
            image = data['image']
            image = image.to(device)
            target_idx = data['label']
            target_idx = target_idx.to(device)
            target = one_hot(target_idx, n_classes).to(device)
            outputs = net(image)

            predicted = net.predict(outputs)
            predicted = predicted.to(device)
            # _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target_idx).sum().item()
            predicted_targets.extend(predicted.tolist())
            true_targets.extend(target_idx.tolist())
    accuracy = correct / total
    print(f"Accuracy: {accuracy}")
    print('Classification Report:')
    print(classification_report(true_targets, predicted_targets))
    print('Confusion Matrix:')
    print(confusion_matrix(true_targets, predicted_targets))
    
    print('Finished Testing')
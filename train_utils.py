import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(criterion, epoch, model, model_type, optimizer, train_iter, batch_size, clip_val, log_interval, scheduler=None):
    """
    Function to run training epoch
    :param criterion: loss function to use
    :param epoch: current epoch index
    :param model: pytorch model object
    :param model_type: model type (only ae/ ae+clf), used to know if needs to calculate accuracy
    :param optimizer: optimizer to use
    :param train_iter: train dataloader
    :param batch_size: size of batch (for logging)
    :param clip_val: gradient clipping value
    :param log_interval: interval to log progress
    :param scheduler: learning rate scheduler, optional.
    :return mean train loss (and accuracy if in clf mode)
    """
    model.train()
    loss_sum = 0
    pred_loss_sum = 0
    correct_sum = 0

    num_samples_iter = 0
    for batch_idx, data in enumerate(train_iter, 1):
        if len(data) == 2:
            data, labels = data[0].to(device), data[1].to(device)
        else:
            data = data.to(device)
        # Zero gradients
        optimizer.zero_grad()

        num_samples_iter += len(data)  # Count number of samples seen in epoch (used for later statistics)

        # Forward pass & loss calculation
        model_out = model(data)
        if model_type == 'LSTMAE_CLF':
            # For MNIST classifier
            model_out, out_labels = model_out
            pred = out_labels.max(1, keepdim=True)[1]
            correct_sum += pred.eq(labels.view_as(pred)).sum().item()
            # Calculate loss
            mse_loss, ce_loss = criterion(model_out, data, out_labels, labels)
            loss = mse_loss + ce_loss
        elif model_type == 'LSTMAE_PRED':
            # For S&P prediction
            model_out, preds = model_out
            labels = data.squeeze()[:, 1:]  # Take x_t+1 as y_t
            preds = preds[:, :-1]  # Take preds up to T-1
            mse_rec, mse_pred = criterion(model_out, data, preds, labels)
            loss = mse_rec + mse_pred
            pred_loss_sum += mse_pred.item()
        else:
            # Calculate loss
            loss = criterion(model_out, data)

        # Backward pass
        loss.backward()
        loss_sum += loss.item()

        # Gradient clipping
        if clip_val is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)

        # Update model params
        optimizer.step()

        # LR scheduler step
        if scheduler is not None:
            scheduler.step()

        # print progress
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, num_samples_iter, len(train_iter.dataset),
                100. * num_samples_iter / len(train_iter.dataset), loss_sum / num_samples_iter))
    train_loss = loss_sum / len(train_iter.dataset)
    train_pred_loss = pred_loss_sum / len(train_iter.dataset)
    train_acc = round(correct_sum / len(train_iter.dataset) * 100, 2)
    acc_out_str = f'; Average Accuracy: {train_acc}' if model_type == 'LSTMAECLF' else ''
    print(f'Train Average Loss: {train_loss}{acc_out_str}')

    return train_loss, train_acc, train_pred_loss


def eval_model(criterion, model, model_type, val_iter, mode='Validation'):
    """
    Function to run validation on given model
    :param criterion: loss function
    :param model: pytorch model object
    :param model_type: model type (only ae/ ae+clf), used to know if needs to calculate accuracy
    :param val_iter: validation dataloader
    :param mode: mode: 'Validation' or 'Test' - depends on the dataloader given.Used for logging
    :return mean validation loss (and accuracy if in clf mode)
    """
    # Validation loop
    model.eval()
    loss_sum = 0
    correct_sum = 0
    with torch.no_grad():
        for data in val_iter:
            if len(data) == 2:
                data, labels = data[0].to(device), data[1].to(device)
            else:
                data = data.to(device)

            model_out = model(data)
            if model_type == 'LSTMAE_CLF':
                model_out, out_labels = model_out
                pred = out_labels.max(1, keepdim=True)[1]
                correct_sum += pred.eq(labels.view_as(pred)).sum().item()
                # Calculate loss
                mse_loss, ce_loss = criterion(model_out, data, out_labels, labels)
                loss = mse_loss + ce_loss
            elif model_type == 'LSTMAE_PRED':
                # For S&P prediction
                model_out, preds = model_out
                labels = data.squeeze()[:, 1:]  # Take x_t+1 as y_t
                preds = preds[:, :-1]  # Take preds up to T-1
                mse_rec, mse_pred = criterion(model_out, data, preds, labels)
                loss = mse_rec + mse_pred
            else:
                # Calculate loss for none clf models
                loss = criterion(model_out, data)

            loss_sum += loss.item()
    val_loss = loss_sum / len(val_iter.dataset)
    val_acc = round(correct_sum / len(val_iter.dataset) * 100, 2)
    acc_out_str = f'; Average Accuracy: {val_acc}' if model_type == 'LSTMAECLF' else ''
    print(f' {mode}: Average Loss: {val_loss}{acc_out_str}')
    return val_loss, val_acc

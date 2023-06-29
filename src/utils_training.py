def train_model(model, train_loader, val_loader, model_path, num_epochs=200, remove=False, learning_rate=0.001):
    """
    Train your model and evaluate on the validation set
    Defines a loss function and optimizer
    Training is done using the recursive approch for horizon > 1.
    You iteratively predict one step ahead and use the predicted value as
    an input for predicting the next time step.
    This means you make a prediction for the next time step, update the input sequence by appending
    the predicted value, and repeat the process until the end of the horizon.

    Parameters
    ----------
    model : any
        model to train.

    train_loader : DataLoader
        Train Dataloader.

    val_loader : Dataloader
        Valid Dataloader.

    model_path : string
        Where to save the model after training it.

    num_epoch : int=200
        How many epochs to train the model.

    remove : bool=False
        Remove the model after the training phase.
    """
    import os
    import torch
    import copy

    criterion = torch.nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):

        train_loss = 0.0

        for inputs, targets in train_loader:

            inputs, targets = inputs.to(device), targets.to(device)
            batch_size, horizon_size, num_nodes = targets.size()
            final_output = torch.empty((batch_size, 0, num_nodes)).to(device)
            outputs = model(inputs)

            final_output = torch.cat([final_output, outputs.unsqueeze(1)], dim=1).float()

            for i in range(1, horizon_size):

                outputs = model(torch.cat((inputs[:, i:, :], final_output), dim=1))
                final_output = torch.cat([final_output, outputs.unsqueeze(1)], dim=1)

            optimizer.zero_grad()
            loss = criterion(final_output, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        valid_losses = validate_model(val_loader, model, optimizer, criterion, valid_losses, model_path, num_epochs, train_loss, epoch)

    best_model = copy.deepcopy(model)
    best_model.load_state_dict(torch.load(model_path))
    if remove:
        os.remove(model_path)

    return best_model, train_losses, valid_losses


def validate_model(val_loader, model, optimizer, criterion, valid_losses, model_path, num_epochs, train_loss, epoch):
    """

    Train your model and evaluate on the validation set
    Defines a loss function and optimizer.


    Parameters
    ----------
    model : any
        model to train.

    val_loader : Dataloader
        Valid Dataloader.

    model_path : string
        Where to save the model after training it.

    num_epoch : int=200
        How many epochs to train the model.

    remove : bool=False
        Remove the model after the training phase.
    """
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_val_loss = float('inf')
    val_loss = 0.0

    for inputs, targets in val_loader:

        inputs, targets = inputs.to(device), targets.to(device)
        batch_size, horizon_size, num_nodes = targets.size()
        final_output = torch.empty((batch_size, 0, num_nodes)).to(device)
        outputs = model(inputs)
        final_output = torch.cat([final_output, outputs.unsqueeze(1)], dim=1)

        for i in range(1, horizon_size):

            outputs = model(torch.cat((inputs[:, i:, :], final_output), dim=1))
            final_output = torch.cat([final_output, outputs.unsqueeze(1)], dim=1)

        optimizer.zero_grad()
        loss = criterion(final_output, targets)
        val_loss += loss.item()

    val_loss /= len(val_loader)
    valid_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_path)
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    return valid_losses


def testmodel(best_model, test_loader, path=None, meanstd_dict =None, sensor_order_list =[], maximum= None):
    """
    Test model using test data :  Testing is done using the recursive approch for horizon > 1 

    Parameters
    ----------
    best_model : any
        model to test.

    test_loader : DataLoader
        Test Dataloader.

    path : string
        model path to load the model from

    meanstd_dict : dictionary
        if the data were center and reduced

    sensor_order_list : list
        List containing the sensor number in order of the data training because node number 
        and sensor number may be different in federated learning

    maximum : float
        if the data were normalize using maximum value

    Returns
    ----------
    y_pred: array
        predicted values by the model
    y_true : array
        actual values to compare to the prediction

    The returned array are in shape :
    (length of time serie, horizon of prediction , time serie dimension)
    """
    import numpy as np
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the best model and evaluate on the test set
    criterion = torch.nn.MSELoss()
    if path:
        best_model.load_state_dict(torch.load(path))
    best_model.double()
    best_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model.to(device)

    # Evaluate the model on the test set
    test_loss = 0.0
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size, horizon_size, num_nodes = targets.size()
            final_output = torch.empty((batch_size, 0, num_nodes)).to(device)
            outputs = best_model(inputs.double())
            final_output = torch.cat([final_output, outputs.unsqueeze(1)], dim=1)
            for i in range(1, horizon_size):
                outputs = best_model(torch.cat((inputs[:, i:, :], final_output), dim=1))
                final_output = torch.cat([final_output, outputs.unsqueeze(1)], dim=1)

            # Save the predictions and actual values for plotting later
            predictions.append(final_output.cpu().numpy())
            actuals.append(targets.cpu().numpy())

    # Concatenate the predictions and actuals
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    y_pred = predictions[:]
    y_true = actuals[:]
    if len(sensor_order_list) > 1:
        for k in range(len(sensor_order_list)):
            y_pred[k] = y_pred[k] * meanstd_dict[sensor_order_list[k]]['std'] + meanstd_dict[sensor_order_list[k]]['mean']
            y_true[k] = y_true[k] * meanstd_dict[sensor_order_list[k]]['std'] + meanstd_dict[sensor_order_list[k]]['mean']
    elif len(sensor_order_list) == 1:
        y_pred = y_pred * meanstd_dict[sensor_order_list[0]]['std'] + meanstd_dict[sensor_order_list[0]]['mean']
        y_true = y_true * meanstd_dict[sensor_order_list[0]]['std'] + meanstd_dict[sensor_order_list[0]]['mean']
    elif maximum:
        y_pred = predictions[:] * maximum
        y_true = actuals[:] * maximum
    return y_true, y_pred


def prepare_training_configs(config_file_path, PATH_EXPERIMENTS, params, df_PeMS):
    """
    Manually fill the params.nodes_to_filter value if left unspecified by the user
    ***REMARK FOR DEVS*** this seems unecessary, we might want to consider a lighter design
    """
    from shutil import copy
    import json

    copy(
        config_file_path,
        PATH_EXPERIMENTS / "config.json",
    )

    if params.nodes_to_filter == []:
        params.nodes_to_filter = list(df_PeMS.columns[:params.number_of_nodes])
        with open(PATH_EXPERIMENTS / "config.json", 'r') as file:
            data = json.load(file)
            data["nodes_to_filter"] = params.nodes_to_filter
            with open(PATH_EXPERIMENTS / "config.json", 'w') as file:
                json.dump(data, file, indent=4, separators=(',', ': '))

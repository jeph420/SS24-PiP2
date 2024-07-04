import torch, tqdm, time
def training_loop(
        network: torch.nn.Module,
        train_data: torch.utils.data.Dataset,
        eval_data: torch.utils.data.Dataset,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        show_progress: bool = False
):
    
    trainable_params = [i for i in network.parameters()]
    training_loss_averages = []
    eval_losses = []

    # Create an optimizer for updating network weights.
    optimizer = torch.optim.Adam(trainable_params, learning_rate)
    # Create DataLoaders for training_data and evaluation_data.
    train_loader = torch.utils.data.DataLoader(train_data, batch_size)
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size)
    # Run 'num_epochs' iterations over train_data.
    for epoch in range(num_epochs):
        if show_progress:
            for _ in tqdm.tqdm(range(1), desc=f"Epoch {epoch}", ncols=100):
                network.train()
                batch_loss = 0
                ## Loop over the DataLoader
                for _ in tqdm.tqdm(range(num_epochs), desc=f"   Training...", ncols=100, leave=False):
                    for data in train_loader:
                    ## Split the train_loader into inputs and targets (the inputs will be fed to the network)
                        inputs, targets = data
                        targets = targets.unsqueeze(1)
                    ## Feed the inputs to the network, and retrieve outputs (don't forget to reset the gradients)
                        optimizer.zero_grad()
                        outputs = network(inputs)
                    ## Compute Mean-Squared Error and Gradients of each batch.
                        loss = torch.nn.functional.mse_loss(outputs, targets)
                        loss.backward()
                    ## Update network's weights according to the loss (above step).
                        optimizer.step()
                    ## Collect batch-loss over the entire epoch - get average loss of the epoch.
                        batch_loss += loss
                training_loss_averages.append(batch_loss/len(train_loader))
                ### Iterate over the entire eval_data - compute and store the loss of the data.
                network.eval()
                batch_loss = 0
                for _ in tqdm.tqdm(range(num_epochs), desc=f"   Evaluation...", ncols=100, leave=False):
                    for data in eval_loader:
                        ## Split the eval_loader into inputs and targets.
                        inputs, targets = data
                        targets = targets.unsqueeze(1)
                        ## Feed the inputs to the network, and retrieve outputs.
                        outputs = network(inputs)
                        ## Compute and accumilate the loss of the batch.
                        loss = torch.nn.functional.mse_loss(outputs, targets)
                        batch_loss += loss
                eval_losses.append(batch_loss)
    
        else:
            network.train()
            batch_loss = 0
            ## Loop over the DataLoader
            for data in train_loader:
            ## Split the train_loader into inputs and targets (the inputs will be fed to the network)
                inputs, targets = data
                targets = targets.unsqueeze(1)
            ## Feed the inputs to the network, and retrieve outputs (don't forget to reset the gradients)
                optimizer.zero_grad()
                outputs = network(inputs)
            ## Compute Mean-Squared Error and Gradients of each batch.
                loss = torch.nn.functional.mse_loss(outputs, targets)
                loss.backward()
            ## Update network's weights according to the loss (above step).
                optimizer.step()
            ## Collect batch-loss over the entire epoch - get average loss of the epoch.
                batch_loss += loss
            training_loss_averages.append(batch_loss/len(train_loader))
            
            ### Iterate over the entire eval_data - compute and store the loss of the data.
            network.eval()
            batch_loss = 0
            for data in eval_loader:
                ## Split the eval_loader into inputs and targets.
                inputs, targets = data
                targets = targets.unsqueeze(1)
                ## Feed the inputs to the network, and retrieve outputs.
                outputs = network(inputs)
                ## Compute and accumilate the loss of the batch.
                loss = torch.nn.functional.mse_loss(outputs, targets)
                batch_loss += loss  
            eval_losses.append(batch_loss/len(eval_loader))

    #### return tuple([list_of_training_loss_averages, list_of_eval_losses]).
    return tuple([training_loss_averages, eval_losses])

if __name__ == "__main__":
    from a4_ex1 import SimpleNetwork
    from dataset import get_dataset

    torch.random.manual_seed(1234)
    train_data, eval_data = get_dataset()
    network = SimpleNetwork(32, [128, 64, 128], 1, True)
    train_losses, eval_losses = training_loop(network, train_data, eval_data, 
                                              num_epochs=10,batch_size=16, learning_rate=1e-3)
    for epoch, (tl, el) in enumerate(zip(train_losses, eval_losses)):
        print(f"Epoch: {epoch} --- Train loss: {tl:7.4f} --- Eval loss: {el:7.4f}")

import datetime
import torch
import time

from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module,
          train_loader, test_loader, epochs=25, train_loss_list: list = [], test_loss_list: list = [], device=device):
    model.to(device)
    model.train()

    writer = SummaryWriter(
        f'runs/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    print('Starting Training')
    curr_best_loss = torch.tensor(float('inf'))
    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f'Epoch[{epoch + 1}|{epochs}] Batch[{i + 1}|{len(train_loader)}]', end='\r')
        print(f'Epoch [{epoch + 1}|{epochs}] finished' + ' ' * 20)
        curr_loss = running_loss / len(train_loader)
        train_loss_list.append(curr_loss)
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
            test_loss_list.append(running_loss / len(train_loader))
            if test_loss < curr_best_loss:
                curr_best_loss = test_loss
                torch.save(model.state_dict(), 'best_model.pt')
            # print to tensorboard
            writer.add_scalar('Loss/train', running_loss /
                              len(train_loader), epoch)
            writer.add_scalar('Loss/test', test_loss / len(test_loader), epoch)
        model.train()
        end_time = time.time()
        est_time = (end_time - start_time) * (epochs - epoch - 1)
        print(
            f"Training loss: {curr_loss:.4f}, Test loss: {test_loss / len(test_loader):.4f}, " +
            f"Took: {end_time - start_time:.2f}s, est time: {str(datetime.timedelta(seconds=int(est_time)))}s")

    print('Finished Training')

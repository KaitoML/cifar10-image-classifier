import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Metric
from tqdm.auto import tqdm

def train_step(epoch: int, num_epochs: int, model: nn.Module, loss_fn: nn.Module, optimizer: Optimizer, accuracy_fn: Metric,
               device: str, dataloaders: dict[str, DataLoader]) -> tuple:
    """
    Runs one pass over the train dataloader for a multiclass classification task.

    Args:
        epoch: current epoch index
        num_epochs: total number of epochs
        model: model being trained
        loss_fn: loss function expecting raw logits
        optimizer: optimizer already bound to model's parameters
        accuracy_fn: accuracy metric, reset at the start of this function
        device: 'cuda' or 'cpu'
        dataloaders: dict with a 'train' key holding a DataLoader

    Returns:
        train_loss, train_acc: training loss and training accuracy values
    """

    accuracy_fn.reset()
    model.train()

    train_loss = 0

    loop = tqdm(dataloaders['train'], total=len(dataloaders['train']), desc=f"Epoch [{epoch+1}/{num_epochs}] (training)")
    for batch, (X, y) in enumerate(loop):
        X, y = X.to(device), y.to(device)

        y_logits = model(X)
        y_preds = torch.argmax(y_logits, dim=1)

        loss = loss_fn(y_logits, y)
        accuracy_fn.update(y_preds, y)

        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss = train_loss / (batch + 1)
        loop.set_postfix(loss=f"{running_loss:.7f}")

    loop.close()

    train_loss /= len(dataloaders['train'])
    train_acc = accuracy_fn.compute()

    return train_loss, train_acc

def test_step(epoch: int, num_epochs: int, model: nn.Module,
              loss_fn: nn.Module, accuracy_fn: Metric, device: str,
              dataloaders: dict[str, DataLoader]) -> tuple:
    """
    Runs one pass over the test dataloader in inference mode (torch.inference_mode).

    Args:
        epoch: current epoch index
        num_epochs: total number of epochs
        model: model being evaluated (switched to .eval() internally)
        loss_fn: loss function expecting raw logits
        accuracy_fn: accuracy metric, reset at the start of this function
        device: 'cuda' or 'cpu'
        dataloaders: dict with a 'test' key holding a DataLoader

    Returns:
        test_loss, test_acc: testing loss and testing accuracy values
    """
    accuracy_fn.reset()
    model.eval()

    test_loss = 0

    with torch.inference_mode():

        loop = tqdm(dataloaders['test'], total=len(dataloaders['test']), desc=f"Epoch [{epoch + 1}/{num_epochs}] (evaluation)")
        for batch, (X, y) in enumerate(loop):
            X, y = X.to(device), y.to(device)

            y_logits = model(X)
            y_preds = torch.argmax(y_logits, dim=1)

            loss = loss_fn(y_logits, y)
            accuracy_fn.update(y_preds, y)

            test_loss += loss.item()

            running_loss = test_loss / (batch + 1)
            loop.set_postfix(loss=f"{running_loss:.7f}")

        loop.close()

        test_loss /= len(dataloaders['test'])
        test_acc = accuracy_fn.compute()

    return test_loss, test_acc


def calculate_time(seconds: int) -> str:
    """
    Converts time to 'HH:MM:SS' format given total number of seconds.
    Used for printing our total time of the training algorithm

    Args:
        seconds: total number of seconds

    Returns:
        time_formatted (str): time in 'HH:MM:SS' format
    """

    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def eval_model(model: nn.Module, loss_fn: nn.Module,
               accuracy_fn: Metric, device: str,
               dataloaders: dict[str, DataLoader],
               training_time: str) -> dict[str, float | str]:
    """
    Final evaluation of a trained model on the test dataloader, with a progress bar.

    Essentially duplicates test_step without the per-epoch progress printing —
    intended for a single call after training finishes, not for use inside
    the epoch loop.

    Args:
        model: trained model (weights already loaded, if applicable, before the call)
        loss_fn: loss function expecting raw logits
        accuracy_fn: accuracy metric, reset at the start of this function
        device: 'cuda' or 'cpu'
        dataloaders: dict with a 'test' key holding a DataLoader
        training_time: total training time in 'HH:MM:SS' format

    Returns:
        dict with keys:
            'Model name' (str): model.__class__.__name__
            'Loss' (float): average test loss
            'Accuracy' (float): average test accuracy
    """
    accuracy_fn.reset()
    eval_loss = 0
    eval_acc = 0

    model.eval()
    with torch.inference_mode():

        loop = tqdm(dataloaders['test'], total=len(dataloaders['test']), desc='Evaluating model')
        for X, y in loop:
            X, y = X.to(device), y.to(device)

            y_logits = model(X)
            loss = loss_fn(y_logits, y)
            acc = accuracy_fn(torch.argmax(y_logits, dim=1), y)
            eval_loss += loss.item()
            eval_acc += acc.item()

        eval_loss /= len(dataloaders['test'])
        eval_acc /= len(dataloaders['test'])

        loop.close()

    summary = {
        'Model name': model.__class__.__name__,
        'Loss': float(eval_loss),
        'Accuracy': float(eval_acc),
        'Training time': training_time
    }

    return summary

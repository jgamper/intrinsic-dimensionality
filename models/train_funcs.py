import torch
import torch.nn.functional as F
from intrinsic.utils.general import parameter_count

def train(model, train_loader, optimizer, epoch_num, batch_log_interval, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % batch_log_interval == 0:
            print('Train Epoch: {} [{: 6d}/{: 6d} ({:2.0f}%)]\tLoss: {:.4f}'.format(
                epoch_num, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, test_loader, device):
    model.eval()
    test_loss, correct = 0., 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='mean').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    pct_correct = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), pct_correct))
    return pct_correct

def use_model(model, device, lr):  # , **kwargs
    _model = model.to(device)
    #optimizer = optim.SGD(_model.parameters(), lr=args.lr, momentum=momentum)
    optimizer = torch.optim.Adam(_model.parameters(), lr=lr)
    return _model, optimizer

def get_stats_for(model, n_epochs, train_loader, test_loader, batch_log_interval, lr, device):
    """

    :param model:
    :param n_epochs:
    :param train_loader:
    :param test_loader:
    :param batch_log_interval:
    :param lr:
    :return:
    """
    print("Total model parameters : %d" % (parameter_count(model),) )
    _model, optimizer = use_model(model, device, lr)
    for epoch in range(1, n_epochs + 1):
        train(model, train_loader, optimizer, epoch, batch_log_interval, device)
        pct_correct = test(_model, test_loader, device)
    return pct_correct
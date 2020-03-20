import torch
import torch.nn.functional as F

def parameter_count(model, just_grad=True):
    """
    Counts the number of model parameters that require grad update
    :param model:
    :param just_grad: If true counts only parameters that require gradient compute
    :return:
    """
    total=0
    for name, param in model.named_parameters():
        param_size = 1
        for d in list(param.data.size()):
            param_size *= d
        if just_grad:
            if param.requires_grad:
                total += param_size
        else:
            total += param_size
    return total

def train(model, train_loader, optimizer, device):
    model.train()
    train_loss, correct = 0., 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = output.detach().max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.detach().view_as(pred)).sum().item()

    train_loss /= len(train_loader.dataset)
    pct_correct = 100. * correct / len(train_loader.dataset)
    return pct_correct, train_loss

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
    return pct_correct, test_loss

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
    highest = 0
    print("Total model parameters : %d" % (parameter_count(model),) )
    _model, optimizer = use_model(model, device, lr)
    for epoch in range(1, n_epochs + 1):
        train(model, train_loader, optimizer, epoch, batch_log_interval, device)
        pct_correct, test_loss = test(_model, test_loader, device)
        if pct_correct > highest:
            highest = pct_correct
    return highest

def get_exponential_range(exp_max=5, num_max=5):
    """
    Returns a range of numbers raising exponentialy
    :param exp_max:
    :param num_max:
    :return:
    """
    array = (i * 10 ** exp for exp in range(2, exp_max) for i in range(1, num_max+1))
    return array


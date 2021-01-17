from data.imagenet import *
from gretta.augmentations import *
from gretta.optimizer import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def test(policy, model, criterion, test_loader):
    policy.eval()
    test_loss = 0.0
    test_total = 0
    test_score = 0
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            levels = policy(inputs)
            x = apply_transform(inputs, levels)
            x = normalize(x)
            outputs = model(x)
            loss = criterion(outputs, targets)
        _, preds = outputs.max(dim=1)
        test_loss += loss.item() * inputs.size(0)
        test_score += (preds == targets).sum().item()
        test_total += inputs.size(0)
    test_loss /= test_total
    test_acc = float(test_score) / float(test_total)
    return test_loss, test_acc


def test_clean(policy, model, criterion):
    return test(policy, model, criterion, clean_test_loader)


def test_corrupt(policy, model, criterion):
    accs = {}
    for c in CORRUPTIONS:
        for s in range(1, 6):
            loader = corrupt_loader(c, s)
            loss, acc = test(policy, model, criterion, loader)
            if c in accs:
                accs[c].append(acc)
            else:
                accs[c] = [acc]
    return accs


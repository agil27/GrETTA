import random
from easydict import EasyDict
from data import *
import pandas as pd

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def default_arg():
    args = EasyDict(
        {
            'model': 'resnext',
            'augmix': True,
            'opt': 'adam',
            'lr': 1e-6,
            'init': 'zero',
            'epochs': 2,
            'reg': 'none',
            'policy': 'resnet18',
            'sigmoid': False,
            'est': 'vanilla',
            'transform': 'geometry',
            'num_samples': 12,
            'paths': {
                'wideresnet': 'model/model_wrn_best.pth.tar',
                'resnext': 'model/model_resnext_best.pth.tar'
            },
            'student_paths': {
                'wideresnet': 'model/wrn_student_augmix.pth',
                'resnext': 'model/resnext_student_augmix.pth'
            },
            'checkpoints': 'checkpoints',
            'cifar': 'cifar',
            'cifarc': 'cifar_corrupt/CIFAR-100-C'
        }
    )
    return args


# test functions
def test_corrupt(tta, model, cifar_corrupt_dir):
    accs = []
    for c in CORRUPTIONS:
        loader = get_corrupt_loader(c, 256, cifar_corrupt_dir)
        acc = test(tta, model, loader)
        accs.append(acc)
    return accs, np.mean(accs)


def test(tta, model, loader):
    score, total = 0, 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        augmented = tta(inputs)
        outputs = model(augmented)
        _, preds = outputs.max(dim=1)
        score += (labels == preds).sum().item()
        total += inputs.size(0)
    acc = score / total
    return acc


# print result
def show_corrupted_acc(acc, avg_acc):
    df = {}
    for i, c in enumerate(CORRUPTIONS):
        df[c] = acc[i]
    df['average'] = avg_acc
    df = pd.DataFrame(df)
    print(df)

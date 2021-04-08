import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from models import *
from gretta.tta import *
from byol_pytorch import BYOL
from tqdm import tqdm, trange
import argparse
import numpy as np


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# define blackbox model
model = WideResNet(depth=40, num_classes=100, widen_factor=2, drop_rate=0.0)
model = model.to(device)
model = nn.DataParallel(model)
dict = torch.load('model/model_wrn_best.pth.tar')
model.load_state_dict(dict['state_dict'])
model.eval()
policy = resnet(num_levels).to(device)

# define datasets and dataloader
clean_test_data = datasets.CIFAR100(
    'cifar',
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

train_ratio = 0.2
train_len = int(train_ratio * len(clean_test_data))
test_len = len(clean_test_data) - train_len
partial_train_data, partial_test_data = torch.utils.data.random_split(clean_test_data, [train_len, test_len])

train_loader = DataLoader(
    partial_train_data,
    batch_size=256,
    shuffle=False,
    pin_memory=True,
    num_workers=4
)

test_loader = DataLoader(
    partial_test_data,
    batch_size=256,
    shuffle=False,
    pin_memory=True,
    num_workers=4
)

full_test_loader = DataLoader(
    clean_test_data,
    batch_size=256,
    shuffle=False,
    pin_memory=True,
    num_workers=4
)


# # BYOL pretrain
# learner = BYOL(
#     policy,
#     image_size = 32,
#     hidden_layer = 'avgpool',
#     use_momentum = False
# )
#
# opt = torch.optim.Adam(learner.parameters(), lr=3e-4)
# for epoch in trange(200):
#     for images, _ in full_test_loader:
#       images = images.to(device)
#       loss = learner(images)
#       opt.zero_grad()
#       loss.backward()
#       opt.step()
#     torch.save(policy.state_dict(), 'model/byol_pretrain.pth')


# train and test (overfitting on the test set)
def test(tta):
    score, total = 0, 0
    for inputs, labels in full_test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        augmented = tta(inputs)
        outputs = model(augmented)
        _, preds = outputs.max(dim=1)
        score += (labels == preds).sum().item()
        total += inputs.size(0)
    acc = score / total
    return acc


# parser = argparse.ArgumentParser()
# parser.add_argument('--pretrain', type=str, default='none')
# parser.add_argument('--opt', type=str, default='adam')
# parser.add_argument('--lr', type=float, default=1e-4)
# parser.add_argument('--init', type=str, default='zero')
# parser.add_argument('--epochs', type=int, default=110)
# parser.add_argument('--backward', type=str, default='full')
# parser.add_argument('--savedir', type=str, default='accs')
# args = parser.parse_args()


def overfit(args):
    assert args.pretrain in ['none', 'byol']
    if args.pretrain == 'byol':
        policy.load_state_dict(torch.load('model/byol_pretrain.pth'))

    assert args.init in ['zero', 'stochastic']
    if args.init == 'zero':
        policy.fc.weight.data.fill_(0.0)
        policy.fc.bias.data.fill_(0.0)

    assert args.opt in ['sgd', 'rmsprop', 'adam']
    lr = float(args.lr)
    if args.opt == 'adam':
        opt = torch.optim.Adam(policy.parameters(), lr=lr)
    elif args.opt == 'sgd':
        opt = torch.optim.SGD(policy.parameters(), lr=lr, momentum=0.9, weight_decay=0.01)
    else: # rmsprop
        opt = torch.optim.RMSprop(policy.parameters(), lr=lr, momentum=0.9, weight_decay=0.01)

    normalize = transforms.Normalize([0.5] * 3, [0.5] * 3)
    criterion = nn.CrossEntropyLoss()

    tta = GrETTA(
        model=model,
        policy=policy,
        opt='whitebox',
        normalize=normalize
    )

    accs = []
    epochs = int(args.epochs)
    assert args.backward in ['full', 'frozen']
    if args.backward == 'frozen':
        for param in policy.parameters():
            param.requires_grad = False
        num_features = policy.fc.in_features
        policy.fc = nn.Linear(num_features, num_levels).to(device)

    for epoch in range(epochs):
        for inputs, labels in full_test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            tta.update(inputs, labels, criterion, opt)
        acc = test(tta)
        accs.append(acc)

    np.savetxt(args.savedir, accs)


class Arg(object):
    def __init__(self):
        pass

args = Arg()

for pretrain in ['none', 'byol']:
    for opt in ['sgd', 'rmsprop', 'adam']:
        for lr in [1e-6, 1e-5, 1e-4]:
            for init in ['zero', 'stochastic']:
                for epochs in [110]:
                    for backward in ['full', 'frozen']:
                        args.pretrain = pretrain
                        args.opt = opt
                        args.lr = lr
                        args.init = init
                        args.epochs = epochs
                        args.backward = backward
                        args.savedir = 'curve/%s_%s_%f_%s_%d_%s' % (args.pretrain, args.opt, args.lr, args.init, args.epochs, args.backward)
                        overfit(args)




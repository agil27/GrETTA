import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from models import *
from gretta.tta import *
from byol_pytorch import BYOL
from tqdm import tqdm, trange
import argparse
import numpy as np
import os


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# define blackbox model
model = WideResNet(depth=40, num_classes=100, widen_factor=2, drop_rate=0.0)
model = model.to(device)
model = nn.DataParallel(model)
dict = torch.load('model/model_wrn_best.pth.tar')
model.load_state_dict(dict['state_dict'])
model.eval()


transform_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=[-15.0, 15.0], translate=[0.0, 0.2], scale=[0.8, 1.2]),
    transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.2, hue=0.2),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
])

train_data = datasets.CIFAR100(
    'cifar',
    train=True,
    transform=transform_aug
)

clean_test_data = datasets.CIFAR100(
    'cifar',
    train=False,
    transform=transforms.ToTensor(),
    download=False
)

train_ratio = 0.2
train_len = int(train_ratio * len(clean_test_data))
test_len = len(clean_test_data) - train_len
partial_train_data, partial_test_data = torch.utils.data.random_split(clean_test_data, [train_len, test_len])


train_loader = DataLoader(
    train_data,
    batch_size=256,
    shuffle=True,
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


CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]


def get_corrupt_loader(corruption, batch_size):
    test_data = datasets.CIFAR100(
        'cifar',
        train=False,
        transform=transforms.ToTensor(),
        download=False
    )
    cifar_corrupt_dir = 'cifar_corrupt/CIFAR-100-C'
    test_data.data = np.load(os.path.join(cifar_corrupt_dir, corruption + '.npy'))
    test_data.targets = torch.LongTensor(np.load(os.path.join(cifar_corrupt_dir, 'labels.npy')))
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    return test_loader

# pretrain student model as to compute the surrogate gradient
# student = resnet(100).to(device)
# criterion_student = nn.MSELoss()
# opt_student = torch.optim.Adam(student.parameters(), lr=1e-3)
# model.eval()
# student.train()
# for epoch in trange(110):
#     avg_loss = 0.0
#     total = 0
#     for inputs, _ in train_loader:
#         inputs = inputs.to(device)
#         with torch.no_grad():
#             outputs = model(inputs)
#         student_outputs = student(inputs)
#         loss = criterion_student(outputs, student_outputs)
#         opt_student.zero_grad()
#         loss.backward()
#         opt_student.step()
#         avg_loss += loss.item() * inputs.size(0)
#         total += inputs.size(0)
#     torch.save(student.state_dict(), 'checkpoints/student.pth')
#     avg_loss /= total
#     print(avg_loss)


def test_corrupt(tta):
    accs = []
    for c in CORRUPTIONS:
        loader = get_corrupt_loader(c, 256)
        acc = test(tta, loader)
        accs.append(acc)
    return accs, np.mean(accs)


def test(tta, loader):
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


def train(args):
    policy = resnet(num_levels).to(device)
    student = resnet(100).to(device)
    student.load_state_dict(torch.load(args.student))

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
        opt = torch.optim.SGD(policy.parameters(), lr=lr)
    else: # rmsprop
        opt = torch.optim.RMSprop(policy.parameters(), lr=lr)

    normalize = transforms.Normalize([0.5] * 3, [0.5] * 3)
    criterion = nn.CrossEntropyLoss()

    assert args.est in ['whitebox', 'vanilla', 'surrogate']
    tta = GrETTA(
        model=model,
        policy=policy,
        opt=args.est,
        student=args.student,
        num_samples=args.num_samples,
        normalize=normalize
    )

    epochs = int(args.epochs)
    assert args.backward in ['full', 'frozen']
    if args.backward == 'frozen':
        for param in policy.parameters():
            param.requires_grad = False
        num_features = policy.fc.in_features
        policy.fc = nn.Linear(num_features, num_levels).to(device)
    acc, avg_acc = test_corrupt(tta)
    print('original', acc, avg_acc)

    for epoch in trange(epochs):
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            tta.update(inputs, labels, criterion, opt)
        state = {'policy': policy.state_dict(), 'opt': opt.state_dict(), 'epoch': epoch}
        torch.save(state, 'checkpoints/cifar_whitebox_epoch_%03d.pth' % epoch)
    acc, avg_acc = test_corrupt(tta)
    print('augmented', acc, avg_acc)


parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', type=str)
parser.add_argument('--opt', type=str)
parser.add_argument('--lr', type=float)
parser.add_argument('--init', type=str)
parser.add_argument('--epochs', type=int)
parser.add_argument('--est', type=str)
parser.add_argument('--backward', type=str)
parser.add_argument('--savedir', type=str)
parser.add_argument('--student', type=str)
args = parser.parse_args()
train(args)
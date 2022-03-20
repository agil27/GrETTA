from third_party import resnext29, WideResNet
from tqdm import tqdm
from utils import *
import torch.nn as nn
import torchvision
from data import *
from gretta.policy import Policy
from gretta.tta import GrETTA
from gretta.transformations import get_num_levels

# random seed, and arguments
setup_seed(0)
args = default_arg()

# dataset and dataloader
train_data, clean_test_data, train_loader, full_test_loader = get_dataset_and_dataloader(args.cifar)

# set up model, student model and policy network
assert args.model in ['wideresnet', 'resnext']
if args.model == 'wideresnet':
    model = WideResNet(depth=40, num_classes=100, widen_factor=2, drop_rate=0.0)
else:  # resnext
    model = resnext29(num_classes=100)

# set up large vision model
model = model.to(device)
model = nn.DataParallel(model)
dict = torch.load(args.paths[args.model])
model.load_state_dict(dict['state_dict'])
model.eval()

# set up student model for Guided ES
student = torchvision.models.resnet18(pretrained=False)
num_features = student.fc.in_features
student.fc = nn.Linear(num_features, 100)
student = student.to(device)
student.load_state_dict(torch.load(args.student_paths[args.model])['weights'])

# set up policy network
assert args.sigmoid in ['True', 'False']
requires_sigmoid = (args.sigmoid == 'True')
assert args.transform in ['color', 'geometry', 'full']
num_levels = get_num_levels(args.transform)
policy = Policy(args.policy, num_levels, requires_sigmoid=requires_sigmoid).to(device)
policy.identity_init()

# set up optimizer
assert args.opt in ['sgd', 'rmsprop', 'adam']
lr = float(args.lr)
if args.opt == 'adam':
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
elif args.opt == 'sgd':
    opt = torch.optim.SGD(policy.parameters(), lr=lr)
else:  # rmsprop
    opt = torch.optim.RMSprop(policy.parameters(), lr=lr)

# normalize function
normalize = transforms.Normalize([0.5] * 3, [0.5] * 3)

# loss function
criterion = nn.CrossEntropyLoss()

# transformation-consistency regularization
assert args.reg in CORRUPTIONS or args.reg in ['test', 'none']
if args.reg in CORRUPTIONS:
    reg_loader = CycleLoader(get_corrupt_data(args.reg))
    reg = True
elif args.reg == 'test':
    reg_loader = CycleLoader(clean_test_data)
    reg = True
else:
    reg_loader = None
    reg = False

assert args.est in ['whitebox', 'vanilla', 'surrogate']

# build TTA class
tta = GrETTA(
    model=model,
    policy=policy,
    opt=args.est,
    student=student,
    num_samples=args.num_samples,
    normalize=normalize,
    reg=reg,
    transform=args.transform
)

epochs = int(args.epochs)

# checkpoints save dir
if not os.path.exists(args.checkpoints):
    os.makedirs(args.checkpoints)

# original model accuracy
acc, avg_acc = test_corrupt(tta, model, args.cifarc)
print('Original Accuracy')
show_corrupted_acc(acc, avg_acc)

# training pipeline
for epoch in range(epochs):
    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        if reg:
            reg_inputs, reg_label = next(reg_loader)
            reg_inputs = reg_inputs.to(device)
        else:
            reg_inputs = None
        tta.update(inputs, labels, criterion, opt, reg_inputs)
    state = {'policy': policy.state_dict(), 'opt': opt.state_dict(), 'epoch': epoch}
    torch.save(state, os.path.join(args.checkpoints, 'cifar_%s_epoch_%03d.pth' % (args.estimator, epoch)))

# test result
tta_acc, tta_avg_acc = test_corrupt(tta, model, args.cifarc)
print('After TTA Accuracy')
show_corrupted_acc(tta_acc, tta_avg_acc)

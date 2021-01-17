from tensorboardX import SummaryWriter
from network.blackbox import *
from network.policy import *
from gretta.metric import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', '-l', default=1e-8, type=float)
parser.add_argument('--batch_size', '-b', default=256, type=int)
parser.add_argument('--model', '-m', default='resnet50')
parser.add_argument('--policy', '-p', default='efficientnet')
parser.add_argument('--epochs', '-e', default=50, type=int)
parser.add_argument('--traintype', '-t', default='oracle')
parser.add_argument('--evaluation', '-v', default='clean')
parser.add_argument('--samples', '-s', default=12, type=int)
args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_levels = len(transform_list)
val_loader = get_val_loader(args.batch_size)
train_loader = get_train_loader(args.batch_size)


def schedule(optimizer, epoch):
    b = args.batch_size / 256.0
    k = args.num_epochs // 3
    if epoch < k:
        m = 1
    elif epoch < 2 * k:
        m = 0.1
    else:
        m = 0.01
    lr = args.lr * m * b
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def val(policy, model, criterion):
    val_loss = 0.0
    val_score = 0
    val_total = 0
    policy.eval()
    for inputs, targets in val_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            levels = policy(inputs)
            levels = torch.sigmoid(levels)
            x = apply_transform(inputs, levels)
            x = normalize(x)
            outputs = model(x)
            loss = criterion(outputs, targets)
        val_loss += loss.item() * inputs.size(0)
        val_total += inputs.size(0)
        _, preds = outputs.max(dim=1)
        val_score += (preds == targets).sum().item()

    val_acc = val_score / val_total
    val_loss = val_loss / val_total
    return val_acc, val_loss


def test(policy, model, criterion, writer):
    assert args.evaluation in ['clean', 'corrupt']
    if args.evaluation == 'clean':
        test_loss, test_acc = test_clean(policy, model, criterion, args.batch_size)
        writer.add_text(tag='test acc', text_string='%.6f%%' % (test_acc * 100.0,))
        writer.add_text(tag='test loss', text_string='%.8f' % test_loss)
    else:
        test_accs = test_corrupt(policy, model, criterion, args.batch_size)
        for c in test_accs.keys():
            writer.add_text(tag='test acc %s' % c, text_string='%.6f%%' % (test_accs[c] * 100.0,))


# gradient oracle / whitebox train
def whitebox_train(policy, model, criterion, optimizer):
    policy.train()
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        levels = policy(inputs)
        levels = torch.sigmoid(levels)
        x = apply_transform(inputs, levels)
        x = normalize(x)
        outputs = model(x)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return policy


def whitebox(writer, model, policy, optimizer, criterion):
    best_val_acc = 0.0
    best_weight = policy.state_dict()
    for epoch in range(args.num_epochs):
        # train phase
        schedule(optimizer, epoch)
        whitebox_train(policy, model, criterion, optimizer)
        # validation phase
    #     val_acc, val_loss = val(policy, model, criterion)
    #     if val_acc > best_val_acc:
    #         best_weight = policy.state_dict()
    #         best_val_acc = val_acc
    #         torch.save('checkpoints/whitebox.pth.tar', best_weight)
    #     writer.add_scalar(tag='val acc', scalar_value=val_acc, global_step=epoch + 1)
    #     writer.add_scalar(tag='val loss', scalar_value=val_loss, global_step=epoch + 1)

    # # test
    # policy.load_state_dict(best_weight)
    test(policy, model, criterion, writer)


# black_box
def blackbox_train(policy, model, criterion, optimizer, es, num_samples, student=None):
    policy.train()
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        levels = policy(inputs)
        levels = torch.sigmoid(levels)
        inputs_rep = inputs.repeat((num_samples, 1, 1, 1))
        targets_rep = targets.repeat((num_samples,))
        func = TTATarget(model, inputs_rep, targets_rep, criterion, normalize)
        if student is None:
            grad = es.step(levels, func)
        else:
            surr = StudentSurrogate(student, inputs, targets, criterion, normalize)
            grad = es.step(levels, func, surr)
        optimizer.zero_grad()
        levels.backward(grad)
        optimizer.step()


def blackbox(writer, model, policy, optimizer, criterion, num_samples=12, guided=False):
    if guided:
        es = SurrogateES(n=num_levels, num_samples=num_samples)
        student = resnet18_raw()
    else:
        es = VanillaES(n=num_levels, num_samples=num_samples)
        student = None

    best_val_acc = 0.0
    best_weight = policy.state_dict()
    for epoch in range(args.num_epochs):
        # train phase
        schedule(optimizer, epoch)
        blackbox_train(policy, model, criterion, optimizer, es, num_samples, student)
        # validation phase
        val_acc, val_loss = val(policy, model, criterion)
        if val_acc > best_val_acc:
            best_weight = policy.state_dict()
            best_val_acc = val_acc
            torch.save('checkpoints/blackbox.pth.tar', best_weight)
        writer.add_scalar(tag='val acc', scalar_value=val_acc, global_step=epoch + 1)
        writer.add_scalar(tag='val loss', scalar_value=val_loss, global_step=epoch + 1)

    # test
    policy.load_state_dict(best_weight)
    test(policy, model, criterion, writer)


def get_init(comment):
    writer = SummaryWriter(comment=comment)

    assert args.model in ['resnet50', 'augmix']
    if args.model == 'resnet50':
        model = resnet50_raw().to(device)
    else:
        model = resnet50_augmix().to(device)

    model = nn.DataParallel(model)
    model.eval()

    assert args.policy in ['resnet18', 'efficientnet']
    if args.policy == 'resnet18':
        policy = resnet18_policy(num_levels).to(device)
        policy.fc.weight.data.fill_(0)
        policy.fc.bias.data.fill_(0)
    else:
        policy = effnet_b0(num_levels).to(device)
        policy._fc.weight.data.fill_(0)
        policy._fc.bias.data.fill_(0)

    optimizer = torch.optim.RMSprop(policy.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    return writer, model, policy, optimizer, criterion


# main proc
assert args.traintype in ['oracle', 'vanilla', 'guided']

if args.traintype == 'oracle':
    writer, model, policy, optimizer, criterion = get_init(comment='gradient_oracle')
    whitebox(writer, model, policy, optimizer, criterion)
elif args.traintype == 'vanilla':
    writer, model, policy, optimizer, criterion = get_init(comment='vanilla_train')
    blackbox(writer, model, policy, optimizer, criterion, num_samples=args.samples, guided=False)
else:
    writer, model, policy, optimizer, criterion = get_init(comment='guided_train')
    blackbox(writer, model, policy, optimizer, criterion, num_samples=args.samples, guided=True)

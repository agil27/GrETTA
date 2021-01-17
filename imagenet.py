import torch.nn as nn
from tensorboardX import SummaryWriter
from network.blackbox import *
from network.policy import *
from gretta.optimizer import *
from gretta.metric import *


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_levels = len(transform_list)
num_epochs = 50
lr = 1e-8


def schedule(optimizer, epoch):
    b = batch_size / 256.0
    k = num_epochs // 3
    if epoch < k:
        m = 1
    elif epoch < 2 * k:
        m = 0.1
    else:
        m = 0.01
    lr_modified = lr * m * b
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_modified


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


def whitebox():
    writer = SummaryWriter(comment='gradient_oracle')
    model = resnet50_raw().to(device)
    model = nn.DataParallel(model)
    model.eval()
    policy = resnet18_policy(num_levels).to(device)
    policy.fc.weight.data.fill_(0)
    policy.fc.bias.data.fill_(0)
    optimizer = torch.optim.RMSprop(policy.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_weight = policy.state_dict()
    for epoch in range(num_epochs):
        # train phase
        schedule(optimizer, epoch)
        whitebox_train(policy, model, criterion, optimizer)
        # validation phase
        val_acc, val_loss = val(policy, model, criterion)
        if val_acc > best_val_acc:
            best_weight = policy.state_dict()
            best_val_acc = val_acc
            torch.save('checkpoints/whitebox.pth.tar', best_weight)
        writer.add_scalar(tag='val acc', scalar_value=val_acc, global_step=epoch + 1)
        writer.add_scalar(tag='val loss', scalar_value=val_loss, global_step=epoch + 1)

    # test on clean set
    policy.load_state_dict(best_weight)
    test_loss, test_acc = test_clean(policy, model, criterion)
    writer.add_text(tag='test acc', text_string='%.6f%%' % (test_acc * 100.0, ))
    writer.add_text(tag='test loss', text_string='%.8f' % test_loss)


# black_box
def blackbox_train(policy, model, criterion, optimizer, es, num_samples, student=None):
    policy.train()
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        levels = policy(inputs)
        levels = torch.sigmoid(levels)
        inputs_rep = inputs.repeat((num_samples, 1, 1, 1))
        targets_rep = targets.repeat((num_samples, ))
        func = TTATarget(model, inputs_rep, targets_rep, criterion, normalize)
        if student is None:
            grad = es.step(levels, func)
        else:
            surr = StudentSurrogate(student, inputs, targets, criterion, normalize)
            grad = es.step(levels, func, surr)
        optimizer.zero_grad()
        levels.backward(grad)
        optimizer.step()


def blackbox(guided=False):
    writer = SummaryWriter(comment='vanilla_train')
    model = resnet50_raw().to(device)
    model = nn.DataParallel(model)
    model.eval()
    policy = resnet18_policy(num_levels).to(device)
    policy.fc.weight.data.fill_(0)
    policy.fc.bias.data.fill_(0)
    optimizer = torch.optim.RMSprop(policy.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    num_samples = 12
    if guided:
        es = SurrogateES(n=num_levels, num_samples=num_samples)
        student = resnet18_raw()
    else:
        es = VanillaES(n=num_levels, num_samples=num_samples)
        student = None

    best_val_acc = 0.0
    best_weight = policy.state_dict()
    for epoch in range(num_epochs):
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

    # test on clean set
    policy.load_state_dict(best_weight)
    test_loss, test_acc = test_clean(policy, model, criterion)
    writer.add_text(tag='test acc', text_string='%.6f%%' % (test_acc * 100.0,))
    writer.add_text(tag='test loss', text_string='%.8f' % test_loss)









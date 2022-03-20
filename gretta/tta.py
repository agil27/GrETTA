import torch
from gretta.transformations import apply_transform, get_tlist
from gretta.utils import Target, Surrogate
from gretta.optimization import VanillaES, SurrogateES


class GrETTA(object):
    def __init__(self, model, policy, opt, normalize, student=None, num_samples=12, reg=False, alpha=0.1,
                 transform='full'):
        self.model = model
        self.policy = policy
        self.reg = reg
        assert opt in ['vanilla', 'surrogate', 'whitebox']
        self.opt = opt
        self.tlist = get_tlist(transform)
        self.num_levels = len(self.tlist)
        if self.opt == 'vanilla':
            self.estimator = VanillaES(n=self.num_levels, num_samples=num_samples)
        elif self.opt == 'surrogate':
            self.estimator = SurrogateES(n=self.num_levels, num_samples=num_samples)
        self.normalize = normalize
        self.student = student
        self.num_samples = num_samples
        self.alpha = alpha

    def update_with_reg(self, inputs, labels, levels, criterion, optimizer, reg_inputs):
        random_levels = torch.sigmoid(torch.rand(reg_inputs.size(0), self.num_levels)).to(inputs.device)
        modified_inputs = apply_transform(reg_inputs, random_levels, self.tlist)
        test_levels = torch.sigmoid(self.policy(reg_inputs))
        aug_test_levels = torch.sigmoid(self.policy(modified_inputs))
        loss_reg = torch.mean((test_levels - aug_test_levels - random_levels) ** 2)
        augmented = apply_transform(inputs, levels, self.tlist)
        augmented = self.normalize(augmented)
        outputs = self.model(augmented)
        loss = criterion(outputs, labels)
        loss_total = loss + self.alpha * loss_reg
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

    def update(self, inputs, labels, criterion, optimizer, reg_inputs):
        self.policy.train()
        levels = self.policy(inputs)

        if self.opt == 'whitebox':
            if self.reg:
                self.update_with_reg(inputs, labels, levels, criterion, optimizer, reg_inputs)
            else:
                self.update_whitebox(inputs, labels, levels, criterion, optimizer)
        elif self.opt == 'vanilla':
            self.update_vanilla(inputs, labels, levels, criterion, optimizer)
        else:  # surrogate
            self.update_surrogate(inputs, labels, levels, criterion, optimizer)

    def update_whitebox(self, inputs, labels, levels, criterion, optimizer):
        augmented = apply_transform(inputs, levels, self.tlist)
        augmented = self.normalize(augmented)
        outputs = self.model(augmented)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def update_vanilla(self, inputs, labels, levels, criterion, optimizer):
        inputs_repeated = inputs.repeat((self.num_samples, 1, 1, 1))
        labels_repeated = labels.repeat((self.num_samples,))
        target_train = Target(self.model, inputs_repeated, labels_repeated,
                              criterion, self.normalize, self.tlist)
        grad = self.estimator.step(levels, target_train)
        optimizer.zero_grad()
        levels.backward(grad)
        optimizer.step()

    def update_surrogate(self, inputs, labels, levels, criterion, optimizer):
        inputs_repeated = inputs.repeat((self.num_samples, 1, 1, 1))
        labels_repeated = labels.repeat((self.num_samples,))
        target_train = Target(self.model, inputs_repeated, labels_repeated, criterion, self.normalize, self.tlist)
        surrogate = Surrogate(self.student, inputs, labels, criterion, self.normalize, self.tlist)
        grad = self.estimator.step(levels, target_train, surrogate)
        optimizer.zero_grad()
        levels.backward(grad)
        optimizer.step()

    def __call__(self, inputs):
        with torch.no_grad():
            levels = self.policy(inputs)
            augmented = apply_transform(inputs, levels, self.tlist)
            augmented = self.normalize(augmented)
        return augmented

    def test(self, inputs):
        with torch.no_grad():
            levels = self.policy(inputs)
            augmented = apply_transform(inputs, levels, self.tlist)
            augmented = self.normalize(augmented)
        return augmented, levels

    def test_converge(self, inputs, steps):
        augmented = inputs
        norm = []
        aug = []
        with torch.no_grad():
            for i in range(steps):
                levels = self.policy(augmented)
                norm.append((levels ** 2).mean().item())
                augmented = apply_transform(augmented, levels, self.tlist)
                aug.append(augmented)
        aug = [self.normalize(a) for a in aug]
        return aug, norm

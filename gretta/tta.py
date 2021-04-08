import torch
from gretta.transformations import apply_transform, num_levels
from gretta.utils import Target, Surrogate
from gretta.optimization import VanillaES, SurrogateES


class GrETTA(object):
    def __init__(self, model, policy, opt, normalize, student=None, num_samples=12):
        self.model = model
        self.policy = policy
        assert opt in ['vanilla', 'surrogate', 'whitebox']
        self.opt = opt
        if self.opt == 'vanilla':
            self.estimator = VanillaES(n=num_levels, num_samples=num_samples)
        elif self.opt == 'surrogate':
            self.estimator = SurrogateES(n=num_levels, num_samples=num_samples)
        self.normalize = normalize
        self.student = student
        self.num_samples = num_samples

    def update(self, inputs, labels, criterion, optimizer):
        self.policy.train()
        levels = self.policy(inputs)
        levels = torch.sigmoid(levels)

        if self.opt == 'whitebox':
            self.update_whitebox(inputs, labels, levels, criterion, optimizer)
        elif self.opt == 'vanilla':
            self.update_vanilla(inputs, labels, levels, criterion, optimizer)
        else:  # surrogate
            self.update_surrogate(inputs, labels, levels, criterion, optimizer)

    def update_whitebox(self, inputs, labels, levels, criterion, optimizer):
        augmented = apply_transform(inputs, levels)
        augmented = self.normalize(augmented)
        outputs = self.model(augmented)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def update_vanilla(self, inputs, labels, levels, criterion, optimizer):
        inputs_repeated = inputs.repeat((self.num_samples, 1, 1, 1))
        labels_repeated = labels.repeat((self.num_samples, ))
        target_train = Target(self.model, inputs_repeated, labels_repeated,
                              criterion, self.normalize)
        grad = self.estimator.step(levels, target_train)
        optimizer.zero_grad()
        levels.backward(grad)
        optimizer.step()

    def update_surrogate(self, inputs, labels, levels, criterion, optimizer):
        inputs_repeated = inputs.repeat((self.num_samples, 1, 1, 1))
        labels_repeated = labels.repeat((self.num_samples, ))
        target_train = Target(self.model, inputs_repeated, labels_repeated, criterion, self.normalize)
        surrogate = Surrogate(self.student, inputs, labels, criterion, self.normalize)
        grad = self.estimator.step(levels, target_train, surrogate)
        optimizer.zero_grad()
        levels.backward(grad)
        optimizer.step()

    def __call__(self, inputs):
        with torch.no_grad():
            levels = self.policy(inputs)
            levels = torch.sigmoid(levels)
            augmented = apply_transform(inputs, levels)
            augmented = self.normalize(augmented)
        return augmented
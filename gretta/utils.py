from gretta.transformations import apply_transform
import torch


class Target(object):
    def __init__(self, model, inputs, labels, criterion, normalize):
        super(Target, self).__init__()
        self.model = model
        self.inputs = inputs
        self.labels = labels
        self.criterion = criterion
        self.normalize = normalize

    def __call__(self, levels):
        augmented = apply_transform(self.inputs, levels)
        augmented = self.normalize(augmented)
        with torch.no_grad():
            outputs = self.model(augmented)
            loss = self.criterion(outputs, self.labels)
        return loss


class Surrogate(object):
    def __init__(self, student, inputs, labels, criterion, normalize):
        super(Surrogate, self).__init__()
        self.student = student
        self.inputs = inputs
        self.labels = labels
        self.criterion = criterion
        self.normalize = normalize

    def __call__(self, levels):
        levels_copy = levels.detach()
        levels_copy.requires_grad_(True)
        augmented = apply_transform(self.inputs, levels_copy)
        augmented = self.normalize(augmented)
        biased_outputs = self.student(augmented)
        biased_error = self.criterion(biased_outputs, self.labels)
        biased_error.backward()
        biased_grad = levels_copy.grad.data
        return biased_grad
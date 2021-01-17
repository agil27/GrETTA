import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import os


def generate_val_dataset():
    # Define Imagenet Dataset
    imagenet_dir = '/n/pfister_lab2/Lab/vcg_natural/'
    original_trainset = datasets.ImageFolder(
        os.path.join(imagenet_dir, 'train')
    )
    ratio = 0.9
    train_size = int(ratio * len(original_trainset))
    val_size = len(original_trainset) - train_size
    _, val_set = random_split(original_trainset, [train_size, val_size])

    transform_aug = transforms.Compose([
        transforms.RandomAffine(degrees=[-15.0, 15.0], translate=[0.0, 0.2], scale=[0.8, 1.2]),
        transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.2, hue=0.2),
        transforms.RandomResizedCrop(size=224, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
    ])

    val_set.transform = transform_aug

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=8
    )

    val_dir = '/n/pfister_lab2/Lab/vcg_natural/imagenet_val'
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    to_pil = transforms.ToPILImage()
    for i in range(1000):
        class_dir = os.path.join(val_dir, str(i))
        os.makedirs(class_dir)

    for i, (inputs, targets) in enumerate(val_loader):
        class_name = targets[0].item()
        class_dir = os.path.join(val_dir, str(class_name))
        img = to_pil(inputs[0])
        img.save(os.path.join(class_dir, 'val_%d.png' % i))


if __name__ == '__main__':
    generate_val_dataset()

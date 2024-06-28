import torchvision
import torchvision.transforms as transforms
import os
from PIL import Image
import zipfile

def save_cifar100_images_to_zip(dataset, zip_filename, num_samples=100):
    # Create a temporary directory to save images
    os.makedirs('cifar100_temp', exist_ok=True)
    
    # Save images with filenames including labels
    for i in range(num_samples):
        img, label = dataset[i]
        img = transforms.ToPILImage()(img)
        img.save(f'cifar100_temp/{label}_{i}.png')

    # Create a ZIP file
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for root, dirs, files in os.walk('cifar100_temp'):
            for file in files:
                zipf.write(os.path.join(root, file), file)

    # Clean up the temporary directory
    for file in os.listdir('cifar100_temp'):
        os.remove(os.path.join('cifar100_temp', file))
    os.rmdir('cifar100_temp')

# Load CIFAR-100 data
transform = transforms.Compose([transforms.ToTensor()])
cifar100_train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

# Save a subset of the CIFAR-100 dataset to a ZIP file
save_cifar100_images_to_zip(cifar100_train, 'cifar100_subset.zip', num_samples=100)

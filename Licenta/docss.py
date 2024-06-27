# from docx import Document

# # Create a new Document
# doc = Document()

# # Title
# doc.add_heading('Implementarea și Evaluarea unei Rețele Convoluționale de 5 Straturi Ascunse pentru Clasificarea Imaginilor CIFAR-10 și Atacuri Adversariale folosind FGSM', 0)

# # Introduction
# doc.add_heading('Introducere', level=1)
# doc.add_paragraph(
#     "În această lucrare de licență, vom explora implementarea și evaluarea unei rețele neuronale convoluționale (CNN) cu 5 straturi ascunse pentru clasificarea imaginilor din setul de date CIFAR-10. De asemenea, vom investiga eficiența atacurilor adversariale utilizând metoda Fast Gradient Sign Method (FGSM)."
# )

# # Code Explanation
# doc.add_heading('Codul Explicat', level=1)

# # Installing and Importing Libraries
# doc.add_heading('Instalarea și Importarea Bibliotecilor', level=2)
# doc.add_paragraph(
#     "!pip install matplotlib\n"
#     "!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\n"
#     "!pip install numpy \n\n"
#     "Aceste comenzi instalează bibliotecile necesare pentru implementarea și evaluarea modelului. `matplotlib` este utilizat pentru vizualizarea datelor, `torch` și `torchvision` pentru implementarea și manipularea rețelelor neuronale, iar `numpy` pentru operațiuni numerice.\n\n"
#     "import torchvision.models as models\n\n"
#     "# Load a pre-trained ResNet-34\n"
#     "resnet18 = models.resnet18(pretrained=True)\n\n"
#     "# Load an untrained ResNet-34\n"
#     "resnet18_untrained = models.resnet18(pretrained=False)\n\n"
#     "print(resnet18)\n\n"
#     "În acest segment de cod, importăm modelele din `torchvision` și încărcăm o rețea pre-antrenată ResNet-34 și una neantrenată. Aceste modele sunt utilizate pentru comparație."
# )

# # Preparing CIFAR-10 Dataset
# doc.add_heading('Pregătirea Setului de Date CIFAR-10', level=2)
# doc.add_paragraph(
#     "import torch\n"
#     "import torch.nn as nn\n"
#     "import torch.optim as optim\n"
#     "import torchvision\n"
#     "import torchvision.transforms as transforms\n"
#     "from sklearn.metrics import roc_auc_score\n"
#     "import numpy as np\n"
#     "import matplotlib.pyplot as plt\n\n"
#     "Importăm bibliotecile necesare pentru implementarea modelului și pentru evaluarea performanței acestuia.\n\n"
#     "transform = transforms.Compose([\n"
#     "    transforms.ToTensor()\n"
#     "])\n\n"
#     "# Load the CIFAR-10 dataset\n"
#     "trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)\n"
#     "trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)\n\n"
#     "testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)\n"
#     "testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)\n"
#     "adversarial_testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)\n\n"
#     "Transformăm imaginile din setul de date CIFAR-10 în tensori și le încărcăm utilizând `DataLoader` pentru antrenament și testare."
# )

# # Defining the CNN
# doc.add_heading('Definirea Rețelei Convoluționale', level=2)
# doc.add_paragraph(
#     "class CNNWith5HiddenLayers(nn.Module):\n"
#     "    def __init__(self):\n"
#     "        super(CNNWith5HiddenLayers, self).__init__()\n"
#     "        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)     # Hidden layer 1\n"
#     "        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)    # Hidden layer 2\n"
#     "        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)   # Hidden layer 3\n"
#     "        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)  # Hidden layer 4\n"
#     "        self.pool = nn.MaxPool2d(2, 2)                  # Auxiliary layer (Pooling)\n"
#     "        self.fc1 = nn.Linear(256 * 2 * 2, 512)          # Hidden layer 5 (Fully connected)\n"
#     "        self.fc2 = nn.Linear(512, 10)                   # Output layer\n"
#     "        self.relu = nn.ReLU()                           # Activation function\n"
#     "        self.dropout = nn.Dropout(0.5)                  # Auxiliary layer (Dropout)\n\n"
#     "    def forward(self, x):\n"
#     "        x = self.pool(self.relu(self.conv1(x)))\n"
#     "        x = self.pool(self.relu(self.conv2(x)))\n"
#     "        x = self.pool(self.relu(self.conv3(x)))\n"
#     "        x = self.pool(self.relu(self.conv4(x)))\n"
#     "        x = x.view(x.size(0), -1)  # Flatten the tensor\n"
#     "        x = self.relu(self.fc1(x))\n"
#     "        x = self.dropout(x)\n"
#     "        x = self.fc2(x)\n"
#     "        return x\n\n"
#     "Definim arhitectura CNN cu 5 straturi ascunse, folosind straturi convoluționale, straturi de pooling, activare ReLU și dropout pentru regularizare."
# )

# # Training the Model
# doc.add_heading('Antrenarea Modelului', level=2)
# doc.add_paragraph(
#     "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
#     "print(f'Using device: {device}')\n\n"
#     "def train_model(model, trainloader, criterion, optimizer, device, epochs=25):\n"
#     "    train_loss = []\n"
#     "    train_accuracy = []\n\n"
#     "    for epoch in range(epochs):\n"
#     "        model.train()  # Set the model to training mode\n"
#     "        running_loss = 0.0\n"
#     "        correct = 0\n"
#     "        total = 0\n"
#     "        for i, (inputs, labels) in enumerate(trainloader):\n"
#     "            inputs, labels = inputs.to(device), labels.to(device)\n\n"
#     "            optimizer.zero_grad()\n"
#     "            outputs = model(inputs)\n"
#     "            loss = criterion(outputs, labels)\n"
#     "            loss.backward()\n"
#     "            optimizer.step()\n"
#     "            running_loss += loss.item()\n\n"
#     "            # Calculate accuracy\n"
#     "            _, predicted = torch.max(outputs.data, 1)\n"
#     "            total += labels.size(0)\n"
#     "            correct += (predicted == labels).sum().item()\n\n"
#     "            if i % 100 == 99:  # Print every 100 batches\n"
#     "                print(f'Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(trainloader)}], Loss: {loss.item():.4f}')\n\n"
#     "        accuracy = 100 * correct / total\n"
#     "        train_loss.append(running_loss / len(trainloader))\n"
#     "        train_accuracy.append(accuracy)\n"
#     "        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {running_loss/len(trainloader):.4f}, Accuracy: {accuracy:.2f}%')\n\n"
#     "    return train_loss, train_accuracy\n\n"
#     "Antrenăm modelul pe setul de date CIFAR-10 folosind `CrossEntropyLoss` ca funcție de pierdere și `Adam` ca optimizer."
# )

# # Evaluating the Model
# doc.add_heading('Evaluarea Modelului', level=2)
# doc.add_paragraph(
#     "def evaluate_model(model, testloader, device):\n"
#     "    model.eval()  # Set the model to evaluation mode\n"
#     "    correct = 0\n"
#     "    total = 0\n"
#     "    all_labels = []\n"
#     "    all_outputs = []\n"
#     "    with torch.inference_mode():\n"
#     "        for i, (inputs, labels) in enumerate(testloader):\n"
#     "            inputs, labels = inputs.to(device), labels.to(device)\n"
#     "            outputs = model(inputs)\n"
#     "            _, predicted = torch.max(outputs.data, 1)\n"
#     "            total += labels.size(0)\n"
#     "            correct += (predicted == labels).sum().item()\n\n"
#     "            all_labels.extend(labels.cpu().numpy())\n"
#     "            all_outputs.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())\n\n"
#     "    all_labels = np.array(all_labels)\n"
#     "    all_outputs = np.array(all_outputs)\n\n"
#     "    # Compute AUC for each class and average them\n"
#     "    auc_scores = []\n"
#     "    for i in range(10):  # Assuming 10 classes for CIFAR-10\n"
#     "        auc = roc_auc_score(all_labels == i, all_outputs[:, i])\n"
#     "        auc_scores.append(auc)\n\n"
#     "    mean_auc = np.mean(auc_scores)\n"
#     "    accuracy = 100 * correct / total\n"
#     "    print(f'AUC scores for each class: {auc_scores}')\n"
#     "    print(f'Accuracy: {accuracy:.2f}%')\n"
#     "    return mean_auc\n\n"
#     "Evaluăm performanța modelului pe setul de testare și calculăm scorul AUC pentru fiecare clasă, precum și acuratețea generală."
# )

# # Visualizing Results
# doc.add_heading('Vizualizarea Rezultatelor', level=2)
# doc.add_paragraph(
#     "# Plot the training loss and accuracy\n"
#     "epochs = range(1, 26)\n"
#     "plt.figure(figsize=(12, 5))\n\n"
#     "plt.subplot(1, 2, 1)\n"
#     "plt.plot(epochs, train_loss, 'b', label='Training Loss')\n"
#     "plt.title('Training Loss')\n"
#     "plt.xlabel('Epochs')\n"
#     "plt.ylabel('Loss')\n"
#     "plt.legend()\n\n"
#     "plt.subplot(1, 2, 2)\n"
#     "plt.plot(epochs, train_accuracy, 'b', label='Training Accuracy')\n"
#     "plt.title('Training Accuracy')\n"
#     "plt.xlabel('Epochs')\n"
#     "plt.ylabel('Accuracy')\n"
#     "plt.legend()\n\n"
#     "plt.tight_layout()\n"
#     "plt.show()\n\n"
#     "Plotăm pierderea și acuratețea pe parcursul epocilor de antrenament pentru a vizualiza performanța modelului."
# )

# # Adversarial Attacks using FGSM
# doc.add_heading('Atacuri Adversariale folosind FGSM', level=2)
# doc.add_paragraph(
#     "import os\n"
#     "import torch\n"
#     "import torch.nn.functional as F\n"
#     "from torchvision.utils import save_image\n\n"
#     "# Define the FGSM attack function\n"
#     "def fgsm_attack(model, image, label, epsilon):\n"
#     "    # Set requires_grad attribute of tensor. Important for Attack\n"
#     "    image.requires_grad = True\n\n"
#     "    # Forward pass the image through the model\n"
#     "    output = model(image)\n"
#     "    loss = criterion(output, label)\n\n"
#     "    # Zero all existing gradients\n"
#     "    model.zero_grad()\n\n"
#     "    # Calculate gradients of model in backward pass\n"
#     "    loss.backward()\n\n"
#     "    # Collect the element-wise sign of the data gradient\n"
#     "    sign_data_grad = image.grad.data.sign()\n\n"
#     "    # Create the perturbed image by adjusting each pixel of the input image\n"
#     "    perturbed_image = image + epsilon * sign_data_grad\n\n"
#     "    # Adding clipping to maintain [0,1] range\n"
#     "    perturbed_image = torch.clamp(perturbed_image, 0, 1)\n\n"
#     "    return perturbed_image\n\n"
#     "# Load a sample image from the test dataset\n"
#     "dataiter = iter(testloader)\n"
#     "images, labels = next(dataiter)\n"
#     "images, labels = images.to(device), labels.to(device)\n\n"
#     "# Use the first image in the batch for the attack\n"
#     "image = images[0].unsqueeze(0)\n"
#     "label = labels[0].unsqueeze(0)\n\n"
#     "# Define epsilon values for the FGSM attack\n"
#     "epsilons = [0.001, 0.002, 0.005]\n\n"
#     "# Create a directory to save the perturbed images if it doesn't exist\n"
#     "os.makedirs('perturbed', exist_ok=True)\n\n"
#     "# Apply the FGSM attack for each epsilon and save the perturbed image\n"
#     "for epsilon in epsilons:\n"
#     "    perturbed_image = fgsm_attack(model, image, label, epsilon)\n"
#     "    filename = f'perturbed/perturbed_image_epsilon_{epsilon}.png'\n"
#     "    save_image(perturbed_image, filename)\n"
#     "    print(f'Perturbed image with epsilon {epsilon} saved in '{filename}')\n\n"
#     "print('All perturbed images saved.')\n\n"
#     "#Save also the mask\n\n"
#     "from PIL import Image\n\n"
#     "image = Image.open('./perturbed/perturbed_image_epsilon_0.001.png')\n"
#     "new_image = image.resize((500, 500))\n"
#     "new_image.save('myimage_500.jpg')\n\n"
#     "Definim și aplicăm atacul adversarial FGSM pe imagini din setul de testare, salvăm imaginile perturbate și vizualizăm rezultatele."
# )

# # Loading Data into MinIO
# doc.add_heading('Încărcarea Datelor în MinIO', level=2)
# doc.add_paragraph(
#     "import os\n"
#     "from minio import Minio\n"
#     "from minio.error import S3Error\n\n"
#     "minio_client = Minio('localhost:9000',\n"
#     "        access_key='N18PI0XRzRbLKB8il7Uk',\n"
#     "        secret_key='8EdhMimLnfe4mYVVYw3BVWPgP7Z5jVagoz79LqEs',\n"
#     "        secure=False\n"
#     "    )\n\n"
#     "def save_and_upload(data_tensor, filename, bucket_name):\n"
#     "    # Convert tensor to PIL Image\n"
#     "    image = transforms.ToPILImage()(data_tensor)\n"
#     "    img_byte_arr = io.BytesIO()\n"
#     "    image.save(img_byte_arr, format='PNG')\n"
#     "    img_byte_arr = img_byte_arr.getvalue()\n\n"
#     "    try:\n"
#     "        minio_client.put_object(\n"
#     "            bucket_name,\n"
#     "            filename,\n"
#     "            io.BytesIO(img_byte_arr),\n"
#     "            len(img_byte_arr),\n"
#     "            content_type='image/png'\n"
#     "        )\n"
#     "        print(f'Uploaded {filename} successfully.')\n"
#     "    except S3Error as e:\n"
#     "        print(f'Failed to upload {filename}: {e}')\n\n"
#     "Configurăm un client MinIO pentru a încărca imagini perturbate pe un server de stocare obiecte."
# )

# # Testing the Model with Adversarial Attacks
# doc.add_heading('Testarea Modelului cu Atacuri Adversariale', level=2)
# doc.add_paragraph(
#     "def test( model, device, test_loader, epsilon ):\n\n"
#     "    # Accuracy counter\n"
#     "    correct = 0\n"
#     "    adv_examples = []\n\n"
#     "    # Loop over all examples in test set\n"
#     "    for data, target in test_loader:\n\n"
#     "        # Send the data and label to the device\n"
#     "        data, target = data.to(device), target.to(device)\n\n"
#     "        # Set requires_grad attribute of tensor. Important for Attack\n"
#     "        data.requires_grad = True\n\n"
#     "        # Forward pass the data through the model\n"
#     "        output = model(data)\n"
#     "        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability\n\n"
#     "        # If the initial prediction is wrong, don't bother attacking, just move on\n"
#     "        if init_pred.item() != target.item():\n"
#     "            image_denorm = denorm(data)\n"
#     "            # Implement the saving function here, for example:\n"
#     "            torch.save(image_denorm, f'./perturbed/init_pred_{init_pred.item()}_target_{target.item()}.pt')\n\n"
#     "            # in loc de continue,save image init pred.item.target,target\n"
#     "            # target.item este pentru a nu imi da tensor\n\n"
#     "        # Calculate the loss\n"
#     "        loss = F.nll_loss(output, target)\n\n"
#     "        # Zero all existing gradients\n"
#     "        model.zero_grad()\n\n"
#     "        # Calculate gradients of model in backward pass\n"
#     "        loss.backward()\n\n"
#     "        # Collect ``datagrad``\n"
#     "        data_grad = data.grad.data\n\n"
#     "        # Restore the data to its original scale\n"
#     "        data_denorm = denorm(data)\n\n"
#     "        # Call FGSM Attack\n"
#     "        perturbed_data = fgsm_attack(data_denorm, epsilon, data_grad)\n\n"
#     "        # Reapply normalization\n"
#     "        perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)\n\n"
#     "        #Test regarding perturbed_data_normalized. scoate ulterior\n\n"
#     "        #save the mask and the perturbed data normalize\n\n"
#     "        # Re-classify the perturbed image\n"
#     "        output = model(perturbed_data_normalized)\n\n"
#     "        # Check for success\n"
#     "        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability\n\n"
#     "        #salveaza final_pred si target\n"
#     "        #pt sql imi trebuie de salvat pe hard path,imaginile si epsilon(doar daca se perturbeaza)\n\n"
#     "    #streamlit sa aiba un input pentru epsilon\n\n"
#     "Definim o funcție pentru testarea modelului cu imagini adversariale, salvarea imaginilor perturbate și evaluarea performanței modelului sub atac adversarial."
# )

# # Conclusion
# doc.add_heading('Concluzie', level=1)
# doc.add_paragraph(
#     "Această lucrare demonstrează cum putem implementa și evalua o rețea neuronală convoluțională pentru clasificarea imaginilor, precum și cum putem folosi atacuri adversariale pentru a testa robustețea modelului. Codul prezentat acoperă întregul proces de la preprocesarea datelor, antrenarea și evaluarea modelului, până la aplicarea și evaluarea atacurilor adversariale."
# )

# # Save the document
# doc.save('lucrare_licenta_CNN_FGSM.docx')
from graphviz import Digraph

dot = Digraph()

# Add nodes
dot.node('Input', 'Input Layer: 32x32x3')
dot.node('Conv1', 'Conv2D: 32x32x32\nkernel=3x3, padding=1')
dot.node('ReLU1', 'ReLU Activation')
dot.node('Conv2', 'Conv2D: 32x32x64\nkernel=3x3, padding=1')
dot.node('ReLU2', 'ReLU Activation')
dot.node('Conv3', 'Conv2D: 32x32x128\nkernel=3x3, padding=1')
dot.node('ReLU3', 'ReLU Activation')
dot.node('Conv4', 'Conv2D: 32x32x256\nkernel=3x3, padding=1')
dot.node('ReLU4', 'ReLU Activation')
dot.node('Pool', 'Max Pooling: 16x16x256\nwindow=2x2')
dot.node('Flatten', 'Flatten: 65536')
dot.node('Dense1', 'Dense: 512')
dot.node('ReLU5', 'ReLU Activation')
dot.node('Dropout', 'Dropout: 0.5')
dot.node('Dense2', 'Dense: 10')
dot.node('Output', 'Output Layer: 10')

# Add edges
dot.edges([('Input', 'Conv1'), ('Conv1', 'ReLU1'), ('ReLU1', 'Conv2'), ('Conv2', 'ReLU2'), 
           ('ReLU2', 'Conv3'), ('Conv3', 'ReLU3'), ('ReLU3', 'Conv4'), ('Conv4', 'ReLU4'), 
           ('ReLU4', 'Pool'), ('Pool', 'Flatten'), ('Flatten', 'Dense1'), ('Dense1', 'ReLU5'), 
           ('ReLU5', 'Dropout'), ('Dropout', 'Dense2'), ('Dense2', 'Output')])

# Save and render the graph
dot.render('cnn_architecture', format='png', view=True)

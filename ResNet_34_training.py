import torch.nn as nn
import os
import torch
from torchvision import datasets, transforms
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from torch.optim import Adam
from utils import progress_bar
# To enable system configuration
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# Define ResNet34 overall structure
class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels, momentum=0.9)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels, momentum=0.9)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion, momentum=0.9)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

# Define Residual Unit
class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.9)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)

# Traning Procedure
def train(epoch):
            
    for j, (x_train, y_train) in enumerate(train_loader):
        model.train()
        x_train, y_train = Variable(x_train), Variable(y_train)
        
        if torch.cuda.is_available():
           x_train = x_train.cuda()
           y_train = y_train.cuda()
        
        output_train = model(x_train)
        loss_train = critic(output_train, y_train)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        
        if (j == len(train_loader)-1):
            print('Epoch : ', epoch)
            print('loss :', loss_train)    
        
    if (epoch % 10 == 0):
        print("=> Saving Checkpoint_Loss")
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint, filename="./res34_2/checkpoint_loss"+"_"+str(epoch)+".pth")
            

# Testing Procedure
def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()
                
            outputs = model(inputs)
            loss = critic(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
            print("Accuracy: ", 100*correct/total, "%")

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('=> Saving Checkpoint_Acc')
        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        save_name = "./res34_2/checkpoint_acc"+"_"+str(epoch)+".pth"
        torch.save(state, save_name)
        best_acc = acc
     
def ResNet34(img_channel=3, num_classes=18):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)

 
def save_checkpoint(state, filename):
    print("=> Saving Chekcpoint")
    torch.save(state, filename)
    
img_path = "./Trail_dataset/train_data"
img_path_test = "./Trail_dataset/test_data"
epochs = 30
batch_size = 10
load_model = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data = datasets.ImageFolder(
    img_path,
    transform = transforms.Compose([transforms.ToTensor()])
)

test_data = datasets.ImageFolder(
    img_path_test,
    transform = transforms.Compose([transforms.ToTensor()])
)

model = ResNet34(img_channel=3, num_classes=18)

optimizer = Adam(model.parameters(), lr = 0.125)

if (load_model):
    print("=> Loading Checkpoint")
    checkpoint = torch.load("./res34/checkpoint_acc_33.pth")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
    
critic = CrossEntropyLoss()
train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
best_acc = 0
best_loss = 0


if torch.cuda.is_available():
    model = model.cuda()
    critic = critic.cuda()

for epoch in range(1, epochs+1):
    print('=> Training')
    train(epoch)
    print('=> Testing')
    test(epoch)
    
torch.save(model, "model_res34.h5")


import torch
from model.simplenet import SimpleNet
from utils.dataloader import SimpleDatasetDetect
from utils.loss import bounding_box_loss

lr = 1e-5
train_dataset = SimpleDatasetDetect()
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
print('Dataset loading complete.')
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SimpleNet()
model.load_state_dict(torch.load('resnet18.pth'))
print('Network loading complete.')
optimizer = torch.optim.SGD(model.parameters(), lr)
loss_function = bounding_box_loss
model.eval()
train_loss = 0
for param in model.parameters():
    param.requires_grad = False
epoch = 0
aver_loss = 0
while True:
    for step, data in enumerate(train_dataloader, start=0):
        images, labels = data
        preds_train = model(images)
        train_loss = loss_function(preds_train[:, 1:5], labels)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        aver_loss = train_loss.item()/32
        print(f'\rloss={aver_loss}', end='')
    print(f'\nEpoch{epoch} finished. Press y to continue, or press other key to finish')
    epoch += 1
    presskey = input()
    if presskey == 'y' or presskey == 'Y':
        continue
    else:
        break
print('Press y to save weights file, or press other key to finish')
presskey = input()
if presskey == 'y' or presskey == 'Y':
    torch.save(model.state_dict(), 'resnet18.pth')

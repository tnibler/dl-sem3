from torch.nn import MSELoss
import data
import torch
from torch import Tensor
from model import Model
from torch.optim.adam import Adam
import datetime
from pathlib import Path
from ddpm import *
import signal
import sys
import matplotlib.pyplot as plt

device = 'cuda'

batch_size = 64
width, height = 32, 32
loaders, info = data.load_dataset_and_make_dataloaders('FashionMNIST', batch_size, num_workers=4)

loss = MSELoss()

dataset_info = {
    'sigma_data': info.sigma_data
}
model = Model()
model.to(device)

num_epochs = 500
opt = Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)

running_loss = []

snapshot_dir = Path('snapshots/')
snapshot_dir.mkdir(exist_ok=True)
save_increment = 5

epoch = 0

def save_snapshot():
    snapshot_name = snapshot_dir.joinpath(Path('model-' + datetime.datetime.now().isoformat() + '.pth'))
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'sched_state_dict': sched.state_dict(),
            'loss': loss.state_dict(),
            'epoch': epoch,
            'dataset_info': dataset_info,
            'running_loss': running_loss
        }, str(snapshot_name))

def finish(_signal, _frame):
    save_snapshot()
    fig, ax = plt.subplots()
    ax.plot(range(len(running_loss)), running_loss)
    fig.savefig('loss.png')
    sys.exit()
signal.signal(signal.SIGINT, finish)

# load_snapshot = sorted(list(Path('./snapshots').glob('*.pth')))[-1]
load_snapshot = None
if load_snapshot:
    print(f"Resuming from checkpoint {load_snapshot}")
    s = torch.load(load_snapshot)
    opt.load_state_dict(s['optimizer_state_dict'])
    loss.load_state_dict(s['loss'])
    sched.load_state_dict(s['sched_state_dict'])
    model.load_state_dict(s['model_state_dict'])
    epoch = s['epoch']
    running_loss = s['running_loss'] if 'running_loss' in s else []

while epoch < num_epochs:
    y: Tensor
    for y, _lbl in loaders.train:
        y.to(device)
        opt.zero_grad()
        bs = y.shape[0]
        sigma = sample_sigma(bs).reshape((bs, 1, 1, 1))
        cin = c_in(sigma, info.sigma_data)
        cout = c_out(sigma, info.sigma_data)
        cskip = c_skip(sigma, info.sigma_data)
        cnoise = c_noise(sigma, info.sigma_data)
        x = y + sigma * torch.randn((bs, 1, width, height))

        pred = model((cin * x).to(device), cnoise.squeeze().to(device)) 
        target = (y - cskip * x) / cout

        l: Tensor = loss(pred, target.to(device))
        l.backward()
        opt.step()
        running_loss.append(l.item())
    sched.step(l.item())
    if epoch > 0 and epoch % save_increment == 0:
        print('Saving checkpoint')
        save_snapshot()
    print(f'Epoch {epoch:03}, LR={sched.get_last_lr()[0]:.05f}\tL={sum(running_loss[-10:])/10:.04f}')
    epoch += 1

finish(None, None)

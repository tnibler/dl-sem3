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


batch_size = 128
width, height = 32, 32
loaders, info = data.load_dataset_and_make_dataloaders('FashionMNIST', batch_size)

loss = MSELoss()

dataset_info = {
    'sigma_data': info.sigma_data
}
model_params = {
    'image_channels': 1,
    'nb_channels': 16,
    'num_blocks': 4,
    'cond_channels': 2
}
model = Model(**model_params)

num_epochs = 30
opt = Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.LinearLR(opt, total_iters=num_epochs)

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
            'loss': loss,
            'epoch': epoch,
            'model_params': model_params,
            'dataset_info': dataset_info,
        }, str(snapshot_name))

def finish(_signal, _frame):
    save_snapshot()
    fig, ax = plt.subplots()
    ax.plot(range(len(running_loss)), running_loss)
    fig.savefig('loss.png')
    sys.exit()
signal.signal(signal.SIGINT, finish)

while epoch < num_epochs:
    img: Tensor
    for img, _lbl in loaders.train:
        opt.zero_grad()
        bs = img.shape[0]
        sigma = sample_sigma(bs).reshape((bs, 1, 1, 1))
        cin = c_in(sigma, info.sigma_data)
        cout = c_out(sigma, info.sigma_data)
        cskip = c_skip(sigma, info.sigma_data)
        cnoise = c_noise(sigma, info.sigma_data)
        noisy = img + sigma * torch.randn((bs, 1, width, height))

        pred = model(noisy, cnoise.squeeze()) 
        target = (img - cskip * img) / cout

        l: Tensor = loss(pred, target)
        l.backward()
        opt.step()
        running_loss.append(l.item())
    sched.step()
    if epoch > 0 and epoch % save_increment == 0:
        save_snapshot()
    print(sum(running_loss[-10:]) / 10)
    epoch += 1

finish(None, None)

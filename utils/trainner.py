import numpy as np
import torch.optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tqdm import tqdm
# from torch.nn import CrossEntropyLoss

from datasets import classes_per_task
from utils.loss import MiBLoss, CrossEntropyLoss

losses = {
    'CE': CrossEntropyLoss,
    'MiB': MiBLoss,
}


class Trainner:
    def __init__(self, params, new_model, old_model, train, valid, device):
        lr = params['lr']  # learning rate
        self.params = params
        self.bs = params['batch_size']  # batch size
        self.device = device
        self.new_model = new_model.to(device)
        self.old_model = old_model.to(device) if old_model is not None else None
        self.train_ds = train
        self.valid_ds = valid
        self.epochs = params['epochs']
        self.optimizer = torch.optim.SGD(new_model.parameters(),
                                         lr, momentum=0.9, nesterov=True)
        self.scheduler = StepLR(self.optimizer, 5000, 0.1)
        if params['lr_policy'] == 'cos':
            self.scheduler = CosineAnnealingLR(self.optimizer, self.epochs)
        old_cls = 0
        if params['stage'] > 0:
            old_cls = classes_per_task(params['dataset'], params['task'], params['stage'] - 1)[-1]
        self.loss = losses[params['loss']](old_cls)

    def train(self):
        self.new_model.train()
        if self.old_model is not None:
            self.old_model.eval()
        # start training
        print("Training start...")
        for epoch in range(1, self.epochs + 1):
            train_losses = []
            for img, msk in tqdm(self.train_ds):
                img = img.to(self.device)
                # print(msk.shape)
                msk = msk.to(self.device)
                # print(img.shape, msk.shape)
                y_new = self.new_model(img)['out']
                y_old = None if self.old_model is None else self.old_model(img)
                # print(y_new.shape)

                l = self.loss(y_new, msk, y_old)
                # l = self.loss(y_new, msk)
                self.optimizer.zero_grad()
                # print(l)
                l.backward(retain_graph=True)
                self.optimizer.step()
                train_losses.append(l.item())
            self.scheduler.step()
            if train_losses:
                train_loss = np.sum(train_losses) / len(train_losses)
                print(f"current loss: {train_loss:.5f}")
        print("train step finished, start saving model..")
        params = self.params
        path = f"./states/{params['dataset']}/{params['task']}/{params['backbone']}_{params['stage']}.pth"
        torch.save(self.new_model.state_dict(), path)

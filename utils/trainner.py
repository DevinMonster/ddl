import os

import numpy as np
import torch.optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tqdm import tqdm

from datasets import classes_per_task
from utils import CSSMetrics
from utils.loss import MiBLoss, CrossEntropyLoss

losses = {
    'CE': CrossEntropyLoss,
    'MiB': MiBLoss,
}


class Trainner:
    def __init__(self, params, new_model, old_model, train, valid, test, device):
        lr = params['lr']  # learning rate
        self.params = params
        self.bs = params['batch_size']  # batch size
        self.device = device
        self.new_model = new_model.to(device)
        self.old_model = old_model.to(device) if old_model is not None else None
        self.train_ds = train
        self.valid_ds = valid
        self.test_ds = test
        self.epochs = params['epochs']
        self.optimizer = torch.optim.SGD(new_model.parameters(),
                                         lr, momentum=0.9, nesterov=True)
        self.scheduler = StepLR(self.optimizer, 5000, 0.1)
        if params['lr_policy'] == 'cos':
            self.scheduler = CosineAnnealingLR(self.optimizer, self.epochs)
        cls = classes_per_task(params['dataset'], params['task'], params['stage'])
        n_classes = sum(cls)
        old_cls = 0
        if params['stage'] > 0: old_cls = n_classes - cls[-1]
        self.loss = losses[params['loss']](old_cls)
        self.metrics = CSSMetrics(n_classes)

    def train(self):
        self.new_model.train()
        if self.old_model is not None:
            self.old_model.eval()

        best_model_dict, mIOU = self.new_model.state_dict(), 0
        # start training
        print("Training start...")
        metrics = ""
        for epoch in range(1, self.epochs + 1):
            train_losses = []
            for img, msk in tqdm(self.train_ds):
                img = img.to(self.device)
                msk = msk.to(self.device)
                y_new = self.new_model(img)['out']
                y_old = None if self.old_model is None else self.old_model(img)['out']
                l = self.loss(y_new, msk, y_old)
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                train_losses.append(l.item())
            self.scheduler.step()
            if train_losses:
                train_loss = np.sum(train_losses) / len(train_losses)
                print(f"current loss: {train_loss:.5f}")

            cur_res = self.valid()
            # update best model
            if float(cur_res['Mean IoU']) > mIOU:
                mIOU = float(cur_res['Mean IoU'])
                best_model_dict = self.new_model.state_dict()
            metrics += f"epoch: {epoch} \n" + str(cur_res) + "\n"

        print("train step finished, start saving best model..")
        params = self.params
        model_dict_path = f"./states/{params['dataset']}/{params['task']}/"
        model_dict_name = f"{params['backbone']}_{params['stage']}.pth"
        log_path = f"./log/{params['dataset']}/{params['task']}/"
        log_name = f"{params['backbone']}_{params['stage']}.txt"
        os.makedirs(model_dict_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
        torch.save(best_model_dict, os.path.join(model_dict_path, model_dict_name))
        print(f"best model state saved to: {os.path.join(model_dict_path, model_dict_name)}")

        metrics += f"Test result:\n {self.test_ds()}\n"

        print("Saving logs...")
        with open(os.path.join(log_path, log_name), "w") as f:
            f.write(metrics)
        print("Log saved!")

    def valid(self):
        return self._test_model(self.valid_ds)

    def test(self):
        print("Start testing...")
        return self._test_model(self.test_ds)

    def _test_model(self, dataset):
        loss_item = []
        with torch.no_grad():
            for img, msk in tqdm(dataset):
                img = img.to(self.device)
                msk = msk.to(self.device)
                y_new = self.new_model(img)['out']
                y_old = None if self.old_model is None else self.old_model(img)['out']
                y_pred = torch.argmax(y_new, dim=1)
                l = self.loss(y_new, msk, y_old)
                loss_item.append(l.item())
                self.metrics.update(msk.cpu().numpy(), y_pred.cpu().numpy())
        res = self.metrics.get_results()
        if len(loss_item) > 0:
            res['Avg Loss'] = np.sum(loss_item) / len(loss_item)
        self.metrics.reset()
        return res

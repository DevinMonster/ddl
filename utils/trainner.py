import os

import numpy as np
import torch.optim
from utils.scheduler import PolyLR, StepLR, CosineAnnealingLR
from tqdm import tqdm

from datasets import classes_per_task
from utils import CSSMetrics
from utils.loss import UnbiasedCrossEntropyLoss, CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torch import autocast

losses = {
    'CE': CrossEntropyLoss,
    'UCE': UnbiasedCrossEntropyLoss,
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
        elif params['lr_policy'] == 'poly':
            length = len(train) if train is not None else 1
            self.scheduler = PolyLR(self.optimizer, params['epochs'] * length, params['lr_power'])

        cls = classes_per_task(params['dataset'], params['task'], params['stage'])
        n_classes = sum(cls)
        old_cls = n_classes - cls[-1]
        self.loss = losses[params['loss']](old_cls)
        self.metrics = CSSMetrics(n_classes)
        # 日志记录相关
        self.writer = SummaryWriter(params['path_tb'])
        self.log_path = f"./log/{params['dataset']}/{params['task']}/"
        self.log_name = f"{params['backbone']}_{params['stage']}_{params['classifier_init_method']}.txt"
        self.model_dict_path = f"./states/{params['dataset']}/{params['task']}/"
        self.model_dict_name = f"{params['backbone']}_{params['stage']}_{params['classifier_init_method']}.pth"
        self.model_pth = os.path.join(self.model_dict_path, self.model_dict_name)
        self.log_pth = os.path.join(self.log_path, self.log_name)

    def train(self):
        if self.old_model is not None:
            self.old_model.eval()

        best_model_dict, mIOU_best = self.new_model.state_dict(), 0
        # start training
        print("Training start...")
        metrics = f"hyper-parameters:\n {self.params}\n"
        for epoch in range(self.epochs):
            self.new_model.train()
            train_losses = []
            for i, (img, msk) in enumerate(tqdm(self.train_ds)):
                img = img.to(self.device)
                msk = msk.to(self.device)
                with autocast(self.device.type):
                    y_new = self.new_model(img)['out']
                    # with torch.no_grad():
                    #     y_old = None if self.old_model is None else self.old_model(img)['out']
                    l = self.loss(y_new, msk)
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                train_losses.append(l.item())
            self.scheduler.step()
            if train_losses:
                train_loss = np.sum(train_losses) / len(train_losses)
                print(f"epoch {epoch + 1} loss: {train_loss:.4f}")

            cur_res = self.valid()
            # update best model
            if float(cur_res['Mean IoU']) > mIOU_best:
                mIOU_best = float(cur_res['Mean IoU'])
                best_model_dict = self.new_model.state_dict()
            metrics += f"epoch: {epoch + 1} \n" + str(cur_res) + "\n"

        print("train step finished, start saving best model..")
        os.makedirs(self.model_dict_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        torch.save(best_model_dict, self.model_pth)
        print(f"best model state saved to: {self.model_pth}")
        print("Saving logs...")
        with open(self.log_pth, "w") as f:
            f.write(metrics)
        print("Log saved!")

    def valid(self):
        return self._test_model(self.valid_ds)

    def test(self):
        print("Start testing...")
        res = self._test_model(self.test_ds)
        print(res)
        return res

    def _test_model(self, dataset):
        self.new_model.eval()
        with torch.no_grad():
            for i, (img, msk) in enumerate(tqdm(dataset)):
                img = img.to(self.device)
                msk = msk.to(self.device)
                with autocast(self.device.type):
                    y_new = self.new_model(img)['out']
                y_pred = torch.argmax(y_new, dim=1)
                self.metrics.update(msk.cpu().numpy(), y_pred.cpu().numpy())
        res = self.metrics.get_results()
        self.metrics.reset()
        torch.cuda.empty_cache()
        return res

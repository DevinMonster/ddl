import os

import numpy as np
import torch.optim
from tqdm import tqdm

from datasets import classes_per_task
from utils import StreamSegMetrics
from utils.loss import UnbiasedKDLoss, CrossEntropyLoss, UnbiasedCrossEntropyLoss
from utils.scheduler import PolyLR, StepLR, CosineAnnealingLR

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
        self.optimizer = torch.optim.SGD(new_model.parameters(), lr, momentum=0.9,
                                         weight_decay=params['weight_decay'], nesterov=True)
        if params['lr_policy'] == 'cos':
            self.scheduler = CosineAnnealingLR(self.optimizer, self.epochs)
        elif params['lr_policy'] == 'poly':
            self.scheduler = PolyLR(self.optimizer, params['epochs'] * len(train), params['lr_power'])
        elif params['lr_policy'] == 'step':
            self.scheduler = StepLR(self.optimizer, 5000, 0.1)
        else:
            raise NotImplementedError(f"policy: {params['lr_policy']} have not implemented!")

        cls = classes_per_task(params['dataset'], params['task'], params['stage'])
        n_classes = sum(cls)
        old_cls = n_classes - cls[-1]
        self.uce = UnbiasedCrossEntropyLoss(old_cls)
        self.ukd = UnbiasedKDLoss(alpha=params['alpha'])
        self.metrics = StreamSegMetrics(n_classes, params['dataset'])
        # 日志记录相关
        self.log_path = f"./log/{params['dataset']}/{params['task']}/"
        self.log_name = f"{params['backbone']}_{params['stage']}_{params['classifier_init_method']}.txt"
        self.model_dict_path = f"./states/{params['dataset']}/{params['task']}/"
        self.model_dict_name = f"{params['backbone']}_{params['stage']}_{params['classifier_init_method']}.pth"
        self.model_pth = os.path.join(self.model_dict_path, self.model_dict_name)
        self.log_pth = os.path.join(self.log_path, self.log_name)

    def train(self):
        os.makedirs(self.model_dict_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        if self.old_model is not None:
            self.old_model.eval()

        best_model_dict, mIOU_best = self.new_model.state_dict(), 0
        # start training
        print("Training start...")
        print(self.params)
        with open(self.log_pth, "w") as f:
            f.write(f"hyper-parameters:\n {self.params}\n")
            for epoch in range(self.epochs):
                self._train_epoch(epoch)
                cur_res = self._valid()
                self.scheduler.step()
                # update best model
                f.write(f"epoch:{epoch}\n {cur_res}\n")
                if float(cur_res['Mean IoU']) > mIOU_best:
                    mIOU_best = float(cur_res['Mean IoU'])
                    print(f"current best mIoU: {mIOU_best}, saving model...")
                    best_model_dict = self.new_model.state_dict()
                    torch.save(best_model_dict, self.model_pth)
            f.write(f"Test result:\n {self.test()}\n")
        print("train step finished!")
        print(f"best mIoU: {mIOU_best}")
        print(f"best model state saved to: {self.model_pth}")

    def _valid(self):
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
                y_new = self.new_model(img)['out']
                y_pred = torch.argmax(y_new, dim=1)
                self.metrics.update(msk.cpu().numpy(), y_pred.cpu().numpy())
        res = self.metrics.get_results()
        self.metrics.reset()
        torch.cuda.empty_cache()
        return res

    def _train_epoch(self, epoch):
        self.new_model.train()
        train_losses = []
        for i, (img, msk) in enumerate(tqdm(self.train_ds)):
            img = img.to(self.device)
            msk = msk.to(self.device)
            y_new = self.new_model(img)['out']
            y_old = self.old_model(img)['out'] if self.old_model is not None else None
            l = self.uce(y_new, msk) + self.ukd(y_new, y_old)
            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()
            train_losses.append(l.item())
        if train_losses:
            train_loss = np.sum(train_losses) / len(train_losses)
            print(f"epoch {epoch + 1} current loss: {train_loss:.4f}")

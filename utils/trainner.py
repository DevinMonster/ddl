import os

import numpy as np
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

from datasets import classes_per_task
from utils import CSSMetrics
from utils.loss import CrossEntropyLoss, LocalPODLoss, UnbiasedCrossEntropyLoss
from utils.scheduler import PolyLR, StepLR, CosineAnnealingLR
from torch import autocast

features_name = {
    'deeplabv3_resnet50': ['backbone.layer1', 'backbone.layer2', 'backbone.layer3', 'backbone.layer4',
                           'classifier.0', 'classifier.1', 'classifier.2', 'classifier.3', 'classifier.4'],
    'deeplabv3_resnet101': ['backbone.layer1', 'backbone.layer2', 'backbone.layer3', 'backbone.layer4',
                            'classifier.0', 'classifier.1', 'classifier.2', 'classifier.3', 'classifier.4'],
    'fcn_resnet50': ['backbone.layer1', 'backbone.layer2', 'backbone.layer3', 'backbone.layer4',
                     'classifier.0', 'classifier.1', 'classifier.2', 'classifier.3', 'classifier.4'],
    'fcn_resnet101': ['backbone.layer1', 'backbone.layer2', 'backbone.layer3', 'backbone.layer4',
                      'classifier.0', 'classifier.1', 'classifier.2', 'classifier.3', 'classifier.4'],
    'deeplabv3_mobilenet_v3_large': ['backbone.0', 'backbone.1', 'backbone.10', 'backbone.11', 'backbone.12',
                                     'backbone.13', 'backbone.14',
                                     'backbone.15', 'backbone.16', 'backbone.2', 'backbone.3', 'backbone.4',
                                     'backbone.5', 'backbone.6',
                                     'backbone.7', 'backbone.8', 'backbone.9', 'classifier.0', 'classifier.1',
                                     'classifier.2', 'classifier.3',
                                     'classifier.4'],
    'lraspp_mobilenet_v3_large': ['backbone.0', 'backbone.1', 'backbone.10', 'backbone.11', 'backbone.12',
                                  'backbone.13', 'backbone.14', 'backbone.15', 'backbone.16', 'backbone.2',
                                  'backbone.3', 'backbone.4', 'backbone.5', 'backbone.6', 'backbone.7', 'backbone.8',
                                  'backbone.9', 'classifier.add', 'classifier.cbr', 'classifier.getattr',
                                  'classifier.getitem', 'classifier.high_classifier', 'classifier.interpolate',
                                  'classifier.low_classifier', 'classifier.mul', 'classifier.scale'],

}


def pseudo_label(msk, y_old):
    idx = msk == 0
    y_pred = torch.argmax(y_old, dim=1)
    msk[idx] = y_pred[idx]
    return msk


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
            self.scheduler = PolyLR(self.optimizer, params['epochs'] * len(train), params['lr_power'])

        cls = classes_per_task(params['dataset'], params['task'], params['stage'])
        n_classes = sum(cls)
        self.distil = LocalPODLoss(params['scale'], params['alpha'])
        self.ce = UnbiasedCrossEntropyLoss(n_classes - cls[-1])
        self.metrics = CSSMetrics(n_classes)
        self.writer = SummaryWriter(params['path_tb'])
        self.log_path = f"./log/{params['dataset']}/{params['task']}/"
        self.log_name = f"{params['backbone']}_{params['stage']}_{params['classifier_init_method']}.txt"
        self.model_dict_path = f"./states/{params['dataset']}/{params['task']}/"
        self.model_dict_name = f"{params['backbone']}_{params['stage']}_{params['classifier_init_method']}.pth"
        self.model_pth = os.path.join(self.model_dict_path, self.model_dict_name)
        self.log_pth = os.path.join(self.log_path, self.log_name)
        # 特征POD使用
        self.feature_extractor_new = create_feature_extractor(self.new_model, features_name[params['backbone']])
        if self.old_model is not None:
            self.feature_extractor_old = create_feature_extractor(self.old_model, features_name[params['backbone']])

    def train(self):
        if self.old_model is not None:
            self.old_model.eval()

        # start training
        print("Training start...")
        best_model_dict, mIOU_best = self.new_model.state_dict(), 0
        metrics = f"hyper-parameters:\n {self.params}\n"
        for epoch in range(0, self.epochs):
            self.new_model.train()
            train_losses = []
            for i, (img, msk) in enumerate(tqdm(self.train_ds)):
                img = img.to(self.device)
                msk = msk.to(self.device)
                # amp混合精度
                with autocast(self.device.type):
                    y_new = self.new_model(img)['out']
                    y_old = None if self.old_model is None else self.old_model(img)['out']

                    # PLOP改进
                    if self.old_model is not None:
                        # 伪标签技术
                        msk = pseudo_label(msk, y_old)
                        # 特征POD技术
                        new_f, old_f = self.calc_pod(img)
                        uce, dis = self.ce(y_new, msk), self.distil(new_f, old_f)
                        l = uce + dis
                        print(uce, dis)
                    else:
                        l = self.ce(y_new, msk)

                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                train_losses.append(l.item())
                self.writer.add_scalar("Loss/train", l.item())
            self.scheduler.step()
            if train_losses:
                train_loss = np.sum(train_losses) / len(train_losses)
                print(f"epoch {epoch + 1} loss: {train_loss:.5f}")

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

        metrics += f"Test result:\n {str(self.test())}\n"

        print("Saving logs...")
        with open(self.log_pth, "w") as f:
            f.write(metrics)
        print("Log saved!")

    def calc_pod(self, imgs):
        if self.old_model is None: return None
        new_f = self.feature_extractor_new(imgs)
        old_f = self.feature_extractor_old(imgs)
        return new_f, old_f

    def valid(self):
        return self._test_model(self.valid_ds)

    def test(self):
        print("Start testing...")
        res = self._test_model(self.test_ds, False)
        print(res)
        return res

    def _test_model(self, dataset, valid=True):
        self.new_model.eval()
        if self.old_model is not None:
            self.old_model.eval()
        loss_item = []
        with torch.no_grad():
            for i, (img, msk) in enumerate(tqdm(dataset)):
                img = img.to(self.device)
                msk = msk.to(self.device)
                y_new = self.new_model(img)['out']
                y_old = None if self.old_model is None else self.old_model(img)['out']
                y_pred = torch.argmax(y_new, dim=1)
                if self.old_model is not None:
                    # 伪标签技术
                    msk = pseudo_label(msk, y_old)
                    # 特征POD技术
                    new_f, old_f = self.calc_pod(img)
                    l = self.ce(y_new, msk) + self.distil(new_f, old_f)
                else:
                    l = self.ce(y_new, msk)
                loss_item.append(l.item())
                self.metrics.update(msk.cpu().numpy(), y_pred.cpu().numpy())
                res = self.metrics.get_results()
                s1, s2 = "Loss/valid", "mIOU/valid"
                if not valid:
                    s1, s2 = "Loss/tests", "mIOU/tests"
                self.writer.add_scalar(s1, l.item())
                self.writer.add_scalar(s2, res['Mean IoU'])
        res = self.metrics.get_results()
        if loss_item:
            res['Avg Loss'] = np.sum(loss_item) / len(loss_item)
        self.metrics.reset()
        torch.cuda.empty_cache()
        return res

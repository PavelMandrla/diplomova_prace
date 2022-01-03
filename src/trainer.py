import os.path
import time

import numpy as np
import torch
from torch import optim, nn

from losses.ot_loss import OT_Loss
from utils import AverageMeter
from models.model import MyModel
from torch.utils.data import DataLoader
from datasets.fdst import FDST

class Trainer(object):

    def __init__(self, args):
        self.args = args

        self.save_dir = self.get_save_dir()

        self.dataloaders = {  # TODO -> move constants to args
            'train': DataLoader(dataset=FDST("../datasets/our_dataset", training=True, sequence_len=1),
                                batch_size=1,
                                shuffle=True,
                                num_workers=1),
            'test': DataLoader(dataset=FDST("../datasets/our_dataset", training=True, sequence_len=1),
                               batch_size=1,
                               shuffle=True,
                               num_workers=1)
        }

        self.model = MyModel()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        downsample_ratio = 2 # TODO -> CHANGE??? IS IT OK??
        self.ot_loss = OT_Loss(args.crop_size, downsample_ratio, args.norm_cood, self.device, args.num_of_iter_in_ot, args.reg)
        self.tv_loss = nn.L1Loss(reduction='none').to(self.device)
        self.mse = nn.MSELoss().to(self.device)
        self.mae = nn.L1Loss().to(self.device)
        # self.save_list = Save_Handle(max_num=1)
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_count = 0

    def get_save_dir(self):
        # TODO -> maybe generate new directory for each new model depending on its params
        save_dir = 'save_dir'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        return save_dir

    def train(self):
        for epoch in range(self.args.max_epoch + 1):       # TODO -> maybe add checkpoint mechanism, like in DM-Count
            self.train_epoch(epoch)
            #if epoch % self.args.val_epoch == 0 and epoch >= self.args.val_start:
            #    self.val_epoch()

    def train_epoch(self, epoch):
        #region Setup for measuring
        epoch_ot_loss = AverageMeter()
        epoch_ot_obj_value = AverageMeter()
        epoch_wd = AverageMeter()
        epoch_count_loss = AverageMeter()
        epoch_tv_loss = AverageMeter()
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        #endregion
        print(self.model)

        self.model.train()  # Set model to training mode

        for step, (inputs, points, gt_discrete) in enumerate(self.dataloaders['train']):
            inputs = inputs.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            gt_discrete = gt_discrete.to(self.device)
            N = inputs.size(0)

            with torch.set_grad_enabled(True):
                outputs, outputs_normed = self.model(inputs) #'tuple' object has no attribute 'dim'

                # region CALCULATE EPOCH LOSSES
                ot_loss, ot_obj_value, wd = self.get_OT_loss(outputs, outputs_normed, points)
                count_loss = self.get_count_loss(outputs, gd_count)
                tv_loss = self.get_TV_loss(gd_count, gt_discrete, outputs_normed)

                # UPDATE EPOCH LOSSES
                epoch_ot_loss.update(ot_loss.item(), N)
                epoch_ot_obj_value.update(ot_obj_value.item(), N)
                epoch_wd.update(wd, N)
                epoch_count_loss.update(count_loss.item(), N)
                epoch_tv_loss.update(tv_loss.item(), N)

                loss = ot_loss + count_loss + tv_loss
                #print(loss)
                # endregion

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                #region CALCULATE ERROR
                pred_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                pred_err = pred_count - gd_count

                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(pred_err * pred_err), N)
                epoch_mae.update(np.mean(abs(pred_err)), N)
                #endregion

        print("avg_epoch_loss: ", epoch_loss.avg)
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(epoch))
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)
        #self.save_list.append(save_path) # TODO -> IS THIS NEEDED???s

    def get_OT_loss(self, outputs, outputs_normed, points):
        ot_loss, wd, ot_obj_value = self.ot_loss(outputs_normed, outputs, points)
        ot_loss = ot_loss * self.args.wot
        ot_obj_value = ot_obj_value * self.args.wot
        return ot_loss, ot_obj_value, wd

    def get_count_loss(self, outputs, gd_count):
        return self.mae(outputs.sum(1).sum(1).sum(1), torch.from_numpy(gd_count).float().to(self.device))

    def get_TV_loss(self, gd_count, gt_discrete, outputs_normed):
        gd_count_tensor = torch.from_numpy(gd_count).float().to(self.device).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        gt_discrete_normed = gt_discrete / (gd_count_tensor + 1e-6)
        #print(outputs_normed.shape, gt_discrete_normed.shape, gt_discrete.shape)
        left = self.tv_loss(outputs_normed, gt_discrete_normed).sum(1).sum(1).sum(1)
        right = torch.from_numpy(gd_count).float().to(self.device)
        return (left * right).mean(0) * self.args.wtv




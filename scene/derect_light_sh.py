import torch
import torch.nn as nn
from arguments import OptimizationParams


class DirectLightEnv:

    def __init__(self, sh_degree):
        self.sh_degree = sh_degree
        env_shs = torch.zeros((1, 3, (self.sh_degree + 1) ** 2)).float().cuda()
        self.env_shs_dc = nn.Parameter(env_shs[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self.env_shs_rest = nn.Parameter(env_shs[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

    @property
    def get_env_shs(self):
        shs_dc = self.env_shs_dc
        shs_rest = self.env_shs_rest
        return torch.cat((shs_dc, shs_rest), dim=1)

    def training_setup(self, training_args: OptimizationParams):
        if training_args.env_rest_lr < 0:
            training_args.env_rest_lr = training_args.env_lr / 20.0
        l = [
            {'params': [self.env_shs_dc], 'lr': training_args.env_lr, "name": "env_dc"},
            {'params': [self.env_shs_rest], 'lr': training_args.env_rest_lr, "name": "env_rest"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def capture(self):
        captured_list = [
            self.sh_degree,
            self.env_shs_dc,
            self.env_shs_rest,
            self.optimizer.state_dict(),
        ]

        return captured_list

    def restore(self, model_args, training_args,
                is_training=False, restore_optimizer=True):
        pass

    def create_from_ckpt(self, checkpoint_path, restore_optimizer=False):
        (model_args, first_iter) = torch.load(checkpoint_path)
        (self.sh_degree,
         self.env_shs_dc,
         self.env_shs_rest,
         opt_dict) = model_args[:4]

        if restore_optimizer:
            try:
                self.optimizer.load_state_dict(opt_dict)
            except:
                print("Not loading optimizer state_dict!")

        return first_iter

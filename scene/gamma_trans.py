import torch
import torch.nn as nn
from arguments import OptimizationParams


class LearningGammaTransform:

    def __init__(self, use_ldr_image):
        self.use_ldr_image = use_ldr_image
        self.gamma = nn.Parameter(torch.ones(1).float().cuda()).requires_grad_(True)

    def training_setup(self, training_args: OptimizationParams):
        l = [{'name': 'gamma', 'params': self.gamma, 'lr': training_args.gamma_lr}]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def capture(self):
        captured_list = [
            self.gamma,
            self.optimizer.state_dict(),
        ]

        return captured_list

    def restore(self, model_args, training_args,
                is_training=False, restore_optimizer=True):
        pass

    def create_from_ckpt(self, checkpoint_path, restore_optimizer=False):
        (model_args, first_iter) = torch.load(checkpoint_path)
        (self.gamma,
         opt_dict) = model_args[:2]

        if restore_optimizer:
            try:
                self.optimizer.load_state_dict(opt_dict)
            except:
                print("Not loading optimizer state_dict!")

        return first_iter

    def hdr2ldr(self, hdr_img):
        if self.use_ldr_image:
            hdr_img = hdr_img.clamp(1e-9, 1)
            ldr_img = hdr_img ** self.gamma
            return ldr_img
        else:
            return hdr_img

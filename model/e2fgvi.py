''' Towards An End-to-End Framework for Video Inpainting
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.flow_comp import SPyNet
from model.modules.feat_prop import BidirectionalPropagation, SecondOrderDeformableAlignment
from model.modules.tfocal_transformer import TemporalFocalTransformerBlock, SoftSplit, SoftComp
from model.modules.spectral_norm import spectral_norm as _spectral_norm

import argparse
from raft.raft import RAFT


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(
            'Network [%s] was created. Total number of parameters: %.1f million. '
            'To see the architecture, do print(network).' %
            (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1
                                           or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' %
                        init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.group = [1, 2, 4, 8, 1]
        self.layers = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 512, kernel_size=3, stride=1, padding=1, groups=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(768, 384, kernel_size=3, stride=1, padding=1, groups=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 256, kernel_size=3, stride=1, padding=1, groups=8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True)
        ])

    def forward(self, x):
        bt, c, h, w = x.size()
        h, w = h // 4, w // 4
        out = x
        for i, layer in enumerate(self.layers):
            if i == 8:
                x0 = out
            if i > 8 and i % 2 == 0:
                g = self.group[(i - 8) // 2]
                x = x0.view(bt, g, -1, h, w)
                o = out.view(bt, g, -1, h, w)
                out = torch.cat([x, o], 2).view(bt, -1, h, w)
            out = layer(out)
        return out


class deconv(nn.Module):
    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size=3,
                 padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel,
                              output_channel,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=padding)

    def forward(self, x):
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        return self.conv(x)


class InpaintGenerator(BaseNetwork):
    def __init__(self, init_weights=True, raft_iters=20, raft_path='weights/raft-sintel.pth'):
        super(InpaintGenerator, self).__init__()
        channel = 256
        hidden = 512
        self.raft_iters = raft_iters

        # 1. Encoder Initialization
        self.encoder = Encoder()

        # 2. Decoder Initialization
        self.decoder = nn.Sequential(
            deconv(channel // 2, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1))

        # 3. Feature Propagation Module
        self.feat_prop_module = BidirectionalPropagation(channel // 2)

        # 4. Soft Split / Composition and Transformer setup
        kernel_size = (7, 7)
        padding = (3, 3)
        stride = (3, 3)
        output_size = (60, 108)
        t2t_params = {
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'output_size': output_size
        }
        self.ss = SoftSplit(channel // 2, hidden, kernel_size, stride, padding, t2t_param=t2t_params)
        self.sc = SoftComp(channel // 2, hidden, output_size, kernel_size, stride, padding)

        n_vecs = 1
        for i, d in enumerate(kernel_size):
            n_vecs *= int((output_size[i] + 2 * padding[i] - (d - 1) - 1) / stride[i] + 1)

        blocks = []
        depths = 8
        for i in range(depths):
            blocks.append(
                TemporalFocalTransformerBlock(dim=hidden, num_heads=4, window_size=(5, 9), 
                                              focal_level=2, focal_window=(5, 9), 
                                              n_vecs=n_vecs, t2t_params=t2t_params, pool_method="fc"))
        self.transformer = nn.Sequential(*blocks)

        # 5. Weight Initialization
        if init_weights:
            self.init_weights()
            for m in self.modules():
                if isinstance(m, SecondOrderDeformableAlignment):
                    m.init_offset()
            
        # 6. RAFT Flow Completion Initialization
        args = argparse.Namespace(
            small=False, 
            mixed_precision=False, 
            alternate_corr=False
        )
        self.update_spynet2 = RAFT(args) # Keeping the name update_spynet to minimize changes elsewhere
        self.update_spynet = SPyNet()

        # 7. Loading RAFT Sintel Weights
        #self.load_raft_weights("./release_model/raft-sintel.pth")

    def load_raft_weights(self, path):
        try:
            checkpoint = torch.load(path, map_location='cpu')
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            
            # Remove 'module.' prefix from DataParallel saving
            new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
            
            self.update_spynet2.load_state_dict(new_state_dict)
            self.update_spynet2.eval()
            print(f"Successfully loaded RAFT weights from {path}")
        except Exception as e:
            print(f"Warning: Could not load RAFT weights from {path}. Error: {e}")

    def forward_bidirect_flow(self, masked_local_frames):
        b, l_t, c, h, w = masked_local_frames.size()

        # RAFT requirements: Downsample to 1/4 (per paper) and ensure multiple of 8
        h_down, w_down = h // 4, w // 4
        pad_h = (8 - h_down % 8) % 8
        pad_w = (8 - w_down % 8) % 8

        frames = F.interpolate(masked_local_frames.view(-1, c, h, w), 
                               size=(h_down, w_down), mode='bilinear', align_corners=True)
        
        if pad_h > 0 or pad_w > 0:
            frames = F.pad(frames, (0, pad_w, 0, pad_h))

        frames = frames.view(b, l_t, c, h_down + pad_h, w_down + pad_w)
        f1 = frames[:, :-1, ...].reshape(-1, c, h_down + pad_h, w_down + pad_w)
        f2 = frames[:, 1:, ...].reshape(-1, c, h_down + pad_h, w_down + pad_w)

        # Inference
        _, flow_fwd = self.update_spynet(f1, f2, iters=self.raft_iters, test_mode=True)
        _, flow_bwd = self.update_spynet(f2, f1, iters=self.raft_iters, test_mode=True)

        # Unpad
        if pad_h > 0 or pad_w > 0:
            flow_fwd = flow_fwd[:, :, :h_down, :w_down]
            flow_bwd = flow_bwd[:, :, :h_down, :w_down]

        return (flow_fwd.view(b, l_t - 1, 2, h_down, w_down), 
                flow_bwd.view(b, l_t - 1, 2, h_down, w_down))

    def forward(self, masked_frames, num_local_frames):
        l_t = num_local_frames
        b, t, ori_c, ori_h, ori_w = masked_frames.size()

        # 1. Flow Completion
        masked_local_frames = (masked_frames[:, :l_t, ...] + 1) / 2
        pred_flows = self.forward_bidirect_flow(masked_local_frames)

        # 2. Feature Extraction & Propagation
        enc_feat = self.encoder(masked_frames.view(b * t, ori_c, ori_h, ori_w))
        _, c, h, w = enc_feat.size()
        
        local_feat = enc_feat.view(b, t, c, h, w)[:, :l_t, ...]
        ref_feat = enc_feat.view(b, t, c, h, w)[:, l_t:, ...]
        
        local_feat = self.feat_prop_module(local_feat, pred_flows[0], pred_flows[1])
        enc_feat = torch.cat((local_feat, ref_feat), dim=1)

        # 3. Content Hallucination (Transformer)
        trans_feat = self.ss(enc_feat.view(-1, c, h, w), b)
        trans_feat = self.transformer(trans_feat)
        trans_feat = self.sc(trans_feat, t)
        enc_feat = enc_feat + trans_feat.view(b, t, -1, h, w)

        # 4. Decoding
        output = torch.tanh(self.decoder(enc_feat.view(b * t, c, h, w)))
        return output, pred_flows


# ######################################################################
#  Discriminator for Temporal Patch GAN
# ######################################################################


class Discriminator(BaseNetwork):
    def __init__(self,
                 in_channels=3,
                 use_sigmoid=False,
                 use_spectral_norm=True,
                 init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 32

        self.conv = nn.Sequential(
            spectral_norm(
                nn.Conv3d(in_channels=in_channels,
                          out_channels=nf * 1,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=1,
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 1,
                          nf * 2,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 2,
                          nf * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 4,
                          nf * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 4,
                          nf * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf * 4,
                      nf * 4,
                      kernel_size=(3, 5, 5),
                      stride=(1, 2, 2),
                      padding=(1, 2, 2)))

        if init_weights:
            self.init_weights()

    def forward(self, xs):
        # T, C, H, W = xs.shape (old)
        # B, T, C, H, W (new)
        xs_t = torch.transpose(xs, 1, 2)
        feat = self.conv(xs_t)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        out = torch.transpose(feat, 1, 2)  # B, T, C, H, W
        return out


def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)
    return module

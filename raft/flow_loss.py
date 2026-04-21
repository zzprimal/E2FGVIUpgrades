import torch
import torch.nn as nn
import torch.nn.functional as F

class RAFTFlowCompletionLoss(nn.Module):
    """Flow completion loss using RAFT as the teacher model"""
    def __init__(self, raft_model):
        super().__init__()
        # Use the provided RAFT instance (should be pre-trained)
        self.fix_raft = raft_model
        for p in self.fix_raft.parameters():
            p.requires_grad = False
        self.fix_raft.eval() 

        self.l1_criterion = nn.L1Loss()

    def forward(self, pred_flows, gt_local_frames):
        """
        pred_flows: tuple (flow_fwd, flow_bwd) from generator
        gt_local_frames: original uncorrupted video [b, l_t, c, h, w]
        """
        b, l_t, c, h, w = gt_local_frames.size()
        h_down, w_down = h // 4, w // 4

        with torch.no_grad():
            # 1. Downsample GT frames
            gt_frames = F.interpolate(gt_local_frames.view(-1, c, h, w),
                                      size=(h_down, w_down),
                                      mode='bilinear',
                                      align_corners=True)

            # 2. Apply the same padding logic used in the generator (multiple of 128)
            pad_h = (128 - h_down % 128) % 128
            pad_w = (128 - w_down % 128) % 128
            
            if pad_h > 0 or pad_w > 0:
                gt_frames = F.pad(gt_frames, (0, pad_w, 0, pad_h))

            # 3. Reshape for RAFT pairs
            cur_h, cur_w = h_down + pad_h, w_down + pad_w
            gt_frames = gt_frames.view(b, l_t, c, cur_h, cur_w)
            
            gtlf_1 = gt_frames[:, :-1, ...].reshape(-1, c, cur_h, cur_w)
            gtlf_2 = gt_frames[:, 1:, ...].reshape(-1, c, cur_h, cur_w)

            # 4. Compute GT flows (Teacher)
            # Standard RAFT returns a list of flows; we take the final one [-1]
            _, gt_flows_forward = self.fix_raft(gtlf_1, gtlf_2, iters=20, test_mode=True)
            _, gt_flows_backward = self.fix_raft(gtlf_2, gtlf_1, iters=20, test_mode=True)

            # 5. Unpad GT flows to match pred_flows size
            if pad_h > 0 or pad_w > 0:
                gt_flows_forward = gt_flows_forward[:, :, :h_down, :w_down]
                gt_flows_backward = gt_flows_backward[:, :, :h_down, :w_down]

        # 6. Calculate L1 Loss
        # pred_flows[0/1] are already [b, l_t-1, 2, h_down, w_down]
        forward_loss = self.l1_criterion(pred_flows[0].reshape(-1, 2, h_down, w_down), 
                                         gt_flows_forward)
        backward_loss = self.l1_criterion(pred_flows[1].reshape(-1, 2, h_down, w_down), 
                                          gt_flows_backward)
        
        return forward_loss + backward_loss

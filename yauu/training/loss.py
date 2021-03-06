from fastai.vision.all import *
import torch


def gram_matrix(x):
    n,c,h,w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1,2))/(c*h*w)


class FeatureLoss(Module):
    def __init__(self, base_loss, m_feat, layer_ids, layer_wgts):

        self.base_loss = base_loss
        self.m_feat = m_feat
        self.loss_features = [m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel', ] + [f'feat_{i}' for i in range(len(layer_ids))
                                           ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target, reduction='mean'):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [self.base_loss(input, target, reduction=reduction)]
        self.feat_losses += [self.base_loss(f_in, f_out, reduction=reduction) * w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [self.base_loss(gram_matrix(f_in), gram_matrix(f_out), reduction=reduction) * w ** 2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        if reduction == 'none':
            self.feat_losses = [f.mean(dim=[1, 2, 3]) for f in self.feat_losses[:4]] + [f.mean(dim=[1, 2]) for f in
                                                                                        self.feat_losses[4:]]
        for n, l in zip(self.metric_names, self.feat_losses): setattr(self, n, l)
        return sum(self.feat_losses)

    def __del__(self):
        self.hooks.remove()


def get_loss(model_path):
    loaded = torch.load(model_path)
    print(loaded)
    vgg_m = loaded['model'].eval()

    blocks = [i - 1 for i, o in enumerate(vgg_m.children()) if isinstance(o, nn.MaxPool2d)]

    return FeatureLoss(F.mse_loss, vgg_m, blocks[2:5], [5,15,2])


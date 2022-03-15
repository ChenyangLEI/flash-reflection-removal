from torch.optim.lr_scheduler import CosineAnnealingLR


def get_schedulers(cfg,opt):
    schedulers=[]
    if cfg.TRAIN.COS_LR:
        schedulers.append(CosineAnnealingLR(opt, cfg.TRAIN.MAX_EPOCH))
    return schedulers
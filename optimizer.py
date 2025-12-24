from torch import optim

def Adam(model, learning_rate, weight_decay):
    return optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)

def SequentialLR(optimizer):
    identity = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, total_iters=0.0)
    main_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=40,
        eta_min=0.00005,
    )
    return optim.lr_scheduler.SequentialLR(optimizer, schedulers=[identity, main_lr_scheduler], milestones=[0])

def MultiStepLR(optimizer):
    return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.3)

import torch


def SGD_Momentum_Optimizer(model, learn_rate=1e-2):
    return torch.optim.SGD(model, lr=learn_rate, momentum=0.9, dampening=0, weight_decay=0, nesterov=False)


def SGD_Momentum_Nesterov_Optimizer(model, learn_rate=1e-2):
    return torch.optim.SGD(model, lr=learn_rate, momentum=0.9, dampening=0, weight_decay=0, nesterov=True)


def SGD_Optimizer(model, learn_rate=1e-2):
    return torch.optim.SGD(model, lr=learn_rate, momentum=0, dampening=0, weight_decay=0, nesterov=False)


def ASGD_Optimizer(model, learn_rate=1e-2):
    return torch.optim.ASGD(model, lr=learn_rate, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)


def Rprop_Optimizer(model, learn_rate=1e-2):
    return torch.optim.Rprop(model, lr=learn_rate, etas=(0.5, 1.2), step_sizes=(1e-06, 50))


def Adam_Oprimizer(model, learn_rate=1e-4):
    return torch.optim.Adam(model, lr=learn_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)


def Adamax_Oprimizer(model, learn_rate=1e-2):
    return torch.optim.Adamax(model, lr=learn_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)


def SparseAdam_Oprimizer(model, learn_rate=1e-2):
    return torch.optim.SparseAdam(model, lr=learn_rate, betas=(0.9, 0.999), eps=1e-08)


def AdamW_Oprimizer(model, learn_rate=1e-3):
    return torch.optim.AdamW(model, lr=learn_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01,
                             amsgrad=False)


def Adagrad_Oprimizer(model, learn_rate=1e-2):
    return torch.optim.Adagrad(model, lr=learn_rate, lr_decay=0, weight_decay=0,
                               initial_accumulator_value=0)


def Adadelta_Oprimizer(model, learn_rate=1e-2):
    return torch.optim.Adadelta(model, lr=learn_rate, rho=0.9, eps=1e-06, weight_decay=0)


def RMSprop_Optimizer(model, learn_rate=1e-2):
    return torch.optim.RMSprop(model, lr=learn_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0,
                               centered=False)


def LBFGS_Oprimizer(model, learn_rate=1):
    return torch.optim.LBFGS(model, lr=learn_rate, max_iter=20, max_eval=None, tolerance_grad=1e-05,
                             tolerance_change=1e-09, history_size=100, line_search_fn=None)


def get_optimizer(name, model, learn_rate):
    if name == 'SGD_ND':
        return SGD_Momentum_Nesterov_Optimizer(model=model, learn_rate=learn_rate)
    elif name == 'SGD_M':
        return SGD_Momentum_Optimizer(model=model, learn_rate=learn_rate)
    elif name == 'SGD':
        return SGD_Optimizer(model=model, learn_rate=learn_rate)
    elif name == 'ASGD':
        return ASGD_Optimizer(model=model, learn_rate=learn_rate)
    elif name == 'RPROP':
        return Rprop_Optimizer(model=model, learn_rate=learn_rate)
    elif name == 'ADAM':
        return Adam_Oprimizer(model=model, learn_rate=learn_rate)
    elif name == 'ADAMX':
        return Adamax_Oprimizer(model=model, learn_rate=learn_rate)
    elif name == 'SPARSEADAM':
        return SparseAdam_Oprimizer(model=model, learn_rate=learn_rate)
    elif name == 'ADAMW':
        return AdamW_Oprimizer(model=model, learn_rate=learn_rate)
    elif name == 'ADAGRAD':
        return Adagrad_Oprimizer(model=model, learn_rate=learn_rate)
    elif name == 'ADADELTA':
        return Adadelta_Oprimizer(model=model, learn_rate=learn_rate)
    elif name == 'RMSPROP':
        return RMSprop_Optimizer(model=model, learn_rate=learn_rate)
    elif name == 'LBFGS':
        return LBFGS_Oprimizer(model=model, learn_rate=learn_rate)

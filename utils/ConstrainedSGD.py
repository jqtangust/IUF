import torch


class ConstrainedSGD(torch.optim.SGD):
    def __init__(self, named_params, lr=1e-3, weight_decay=0, momentum=0, dampening=0, nesterov=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not isinstance(nesterov, bool):
            raise ValueError("Invalid nesterov value: {}".format(nesterov))

        self.feature_matrix = None
        self.logger = None
        # separate names and parameters
        names, params = zip(*named_params)
        self.param_names = dict(zip(params, names))
        
        alpha = 1
        channel_weights = torch.log(alpha * torch.arange(0, 256) + 1)
        channel_weights = channel_weights / torch.max(channel_weights)
        self.channel_weights_revise = channel_weights.unsqueeze(1)
        
        super(ConstrainedSGD, self).__init__(params, lr=lr, weight_decay=weight_decay, momentum=momentum, dampening=dampening, nesterov=nesterov)

    def add_feature_matrix(self, feature_matrix, logger):
        self.feature_matrix = feature_matrix
        self.logger = logger

    def step(self, closure=None):
        """Performs a single optimization step."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if group['weight_decay'] != 0:
                    d_p.add_(group['weight_decay'], p.data)

                # apply momentum
                if group['momentum'] != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(group['momentum']).add_(1 - group['dampening'], d_p)
                    if group['nesterov']:
                        d_p = d_p.add(group['momentum'], buf)
                    else:
                        d_p = buf

                # incorporate svd information
                param_name = self.param_names[p]
                param_name_clean = param_name.replace('.weight', '').replace('.bias', '')
                # self.logger.info(f"param_name: {param_name}")
                
                if self.feature_matrix is not None and param_name_clean in self.feature_matrix:
                    svd_space = self.feature_matrix[param_name_clean]
                    
                    # self.logger.info(f"param_name: {param_name}")
                    # self.logger.info(f"self.feature_matrix[param_name] size: {self.feature_matrix[param_name_clean].size()}")
                    # self.logger.info(f"d_p size: {d_p.size()}")
                    
                    if '.weight' in param_name:
                        
                        grad_projected = torch.matmul(svd_space, d_p)  # project to SVD space
                        
                        grad_projected = torch.mul(self.channel_weights_revise.to(grad_projected.device), grad_projected) # constraint strategy
                        
                        d_p = torch.matmul(svd_space.T, grad_projected)  # project back to original space
                    elif '.bias' in param_name:
                        grad_projected = torch.matmul(svd_space, d_p.unsqueeze(-1))  
                        
                        grad_projected = torch.mul(self.channel_weights_revise.to(grad_projected.device), grad_projected)

                        d_p = torch.matmul(svd_space.T, grad_projected).squeeze() 

                p.data.add_(-group['lr'], d_p)


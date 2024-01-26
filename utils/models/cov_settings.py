import torch


class CovMatrix_IRW:
    def __init__(self, relax_denom=2.0):
        super(CovMatrix_IRW, self).__init__()
        self.relax_denom = relax_denom

    def __call__(self, feats):

        dim = feats.shape[1]
        self.dim = dim
        self.i = torch.eye(dim, dim).cuda()
        self.reversal_i = torch.ones(dim, dim).triu(diagonal=1).cuda()

        self.num_off_diagonal = torch.sum(self.reversal_i)
        if self.relax_denom == 0:
            print("relax_denom == 0!!!!!")
            self.margin = 0
        else:
            self.margin = self.num_off_diagonal // self.relax_denom

        return self.i, self.reversal_i, self.margin, self.num_off_diagonal


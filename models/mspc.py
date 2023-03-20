import torch 
from torch import nn 
from torch.nn import functional as F
import itertools
import numpy as np

def grid_sample(input, grid, canvas = None, mode='bilinear'):
    output = F.grid_sample(input, grid, mode=mode, align_corners=True)
    if canvas is None:
        return output
    else:
        input_mask = input.data.new(input.size().fill_(1))
        output_mask = F.grid_sample(input_mask, grid, mode=mode)
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output

def compute_partial_repr(input_points, control_points):
    N = input_points.size(1)
    M = control_points.size(1)
    pairwise_diff = input_points.view(input_points.shape[0], N, 1, 2) - control_points.view(input_points.shape[0], 1, M, 2)
    # original implementation, very slow
    # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[:, :, :, 0] + pairwise_diff_square[:, :, :, 1]
    repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
    # fix numerical error for 0 * log(0), substitute all nan with 0
    mask = repr_matrix != repr_matrix
    repr_matrix.masked_fill_(mask, 0)
    return repr_matrix

def scale_constraint(source_control_points, target_control_points, pert_threshold=0.1):
    constraint_index = np.random.choice(source_control_points.shape[1], 3, replace=False)
    constraint_source_points1 = source_control_points[:, constraint_index[:2], :]
    constraint_target_points1 = target_control_points[:, constraint_index[:2], :]
    constraint_dis1 = ((constraint_source_points1[:, 0, :] - constraint_source_points1[:, 1, :]) ** 2).sum(1).sqrt()
    constraint_dis_t1 = ((constraint_target_points1[:, 0, :] - constraint_target_points1[:, 1, :]) ** 2).sum(1).sqrt()

    constraint_source_points2 = source_control_points[:, constraint_index[1:], :]
    constraint_target_points2 = target_control_points[:, constraint_index[1:], :]
    constraint_dis2 = ((constraint_source_points2[:, 0, :] - constraint_source_points2[:, 1, :]) ** 2).sum(1).sqrt()
    constraint_dis_t2 = ((constraint_target_points2[:, 0, :] - constraint_target_points2[:, 1, :]) ** 2).sum(1).sqrt()

    z = pert_threshold
    a = -(z + 1 / z) / 2.0
    b = ((z - 1 / z) / 2.0)

    z = 2.0
    c = -(z + 1 / z) / 2.0
    d = ((z - 1 / z) / 2.0)

    constraint = ((((constraint_dis1 + constraint_dis2) / (constraint_dis_t1 + constraint_dis_t2)) + a).abs()).clamp(min=b).mean() \
                + ((constraint_dis1 / constraint_dis_t1) / (constraint_dis2 / constraint_dis_t2) + c).abs().clamp(min=d).mean()

    return constraint

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None

class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class TPSGridGen(nn.Module):
    def __init__(self,):
        super(TPSGridGen, self).__init__()

        self.register_buffer('padding_matrix', torch.zeros(3, 2))

    def cal_matrix(self, target_height, target_width, target_control_points):
        assert target_control_points.ndimension() == 3
        assert target_control_points.size(2) == 2
        N = target_control_points.size(1)
        target_control_points = target_control_points.float()
        device = torch.device(target_control_points.get_device())

        # create padded kernel matrix
        forward_kernel = torch.zeros(target_control_points.shape[0], N + 3, N + 3).to(device)
        target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points)
        forward_kernel[:, :N, :N] = (target_control_partial_repr)
        forward_kernel[:, :N, -3].fill_(1)
        forward_kernel[:, -3, :N].fill_(1)
        forward_kernel[:, :N, -2:] = target_control_points
        forward_kernel[:, -2:, :N] = (target_control_points.transpose(1, 2))
        # compute inverse matrix
        inverse_kernel = torch.linalg.inv(forward_kernel)

        # create target cordinate matrix
        HW = target_height * target_width
        target_coordinate = list(itertools.product(range(target_height), range(target_width)))
        target_coordinate = torch.Tensor(target_coordinate).to(device) # HW x 2
        Y, X = target_coordinate.split(1, dim=1)
        Y = Y * 2 / (target_height - 1) - 1
        X = X * 2 / (target_width - 1) - 1
        target_coordinate = torch.cat([X, Y], dim=1)  # convert from (y, x) to (x, y)
        target_coordinate = torch.cat([target_coordinate.unsqueeze(dim=0)]*target_control_points.shape[0], dim=0)
        target_coordinate_partial_repr = compute_partial_repr(target_coordinate, target_control_points)
        target_coordinate_repr = torch.cat([
            target_coordinate_partial_repr, torch.ones(target_control_points.shape[0],HW, 1).to(device), target_coordinate
        ], dim=2)
        return inverse_kernel, target_coordinate_repr

    def forward(self, source_control_points, target_control_points, target_height, target_width):
        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == target_control_points.size(1)
        assert source_control_points.size(2) == 2

        inverse_kernel, target_coordinate_repr = self.cal_matrix(target_height, target_width, target_control_points)
        batch_size = source_control_points.size(0)
        Y = torch.cat([source_control_points, self.padding_matrix.expand(batch_size, 3, 2)], 1)
        mapping_matrix = torch.matmul(inverse_kernel, Y)
        source_coordinate = torch.matmul(target_coordinate_repr, mapping_matrix)
        return source_coordinate

class CNN(nn.Module):
    def __init__(self, num_output,in_c=3):
        super(CNN, self).__init__()
        self.net = \
            nn.Sequential(
            nn.Conv2d(in_c, 64, kernel_size=5),
            nn.InstanceNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=5),
            nn.InstanceNorm2d(256),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(256, 20, kernel_size=5),
            nn.InstanceNorm2d(20),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        # resnet18(pretrained=False, num_classes=320, norm_layer=nn.InstanceNorm2d)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_output)

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class UnBoundedGridLocNet(nn.Module):
    def __init__(self, grid_height, grid_width, target_control_point, in_c=3):
        super(UnBoundedGridLocNet, self).__init__()

        self.cnn = CNN(grid_height * grid_width * 2, in_c=in_c)

        bias = target_control_point.view(-1)
        self.reverse = GradientReversal()
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = self.cnn(x)
        points = points.view(batch_size, -1, 2)
        points = self.reverse(points)
        return points

class PerturbationNetwork(nn.Module):
    def __init__(self, grid_size, crop_size, r1, r2, pert_threshold, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.grid_size = grid_size
        self.crop_size = crop_size
        self.pert_threshold = pert_threshold

        self.target_control_points = torch.Tensor(list(itertools.product(
            np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (grid_size - 1)),
            np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (grid_size - 1)),
        )))
        Y, X = self.target_control_points.split(1, dim=1)
        self.target_control_point = torch.cat([X, Y], dim=1).to(device)

        self.loc = UnBoundedGridLocNet(grid_size, grid_size, self.target_control_point)
        self.tps = TPSGridGen()
    
    def forward(self, x):
        b, _, _, _ = x.shape
        self.target_control_points = torch.cat([self.target_control_point.unsqueeze(0)]*b, dim=0)

        downsampled = F.interpolate(x, [64, 64], mode='bilinear', align_corners=True)
        source_control_points = self.loc(downsampled)
        source_coordinate = self.tps(source_control_points, self.target_control_points, self.crop_size, self.crop_size)
        grid = source_coordinate.view(b, self.crop_size, self.crop_size, 2)
        pert = grid_sample(x, grid)
        constraint = scale_constraint(source_control_points, self.target_control_points, self.pert_threshold)
        cordinate_contraint = ((source_coordinate.mean(dim=1).abs()).clamp(min=0.25)).mean()

        return grid, pert, constraint, cordinate_contraint

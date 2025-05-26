import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import abc




class MetaKANConvNDLayer(nn.Module):
    @staticmethod
    def _to_tuple(val, N, name):
        if isinstance(val, (list, tuple)):
            if len(val) == N:
                return tuple(val)
            else:
                raise ValueError(f"{name} must be a tuple of length {N}, but got {len(val)}")
        elif isinstance(val, int):
            return (val,) * N
        else:
            raise TypeError(f"{name} must be an int or a list/tuple of length {N}, but got {type(val)}")

    def __init__(self, conv_class, norm_class, input_dim, output_dim, spline_order, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1,
                 ndim: int = 2, grid_size=5, base_activation=nn.GELU, grid_range=[-1, 1], dropout=0.0,
                 o_batch_size=32, **norm_kwargs):
        super(MetaKANConvNDLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.spline_order = spline_order
        
        self.groups = groups
        self.ndim = ndim
        self.grid_size = grid_size
        self.base_activation = base_activation() if base_activation is not None else nn.Identity()
        self.grid_range = grid_range
        
        # 确保卷积参数是元组形式
        self.kernel_size_tuple = self._to_tuple(kernel_size, ndim, "kernel_size")
        self.stride_tuple = self._to_tuple(stride, ndim, "stride")
        self.padding_tuple = self._to_tuple(padding, ndim, "padding")
        self.dilation_tuple = self._to_tuple(dilation, ndim, "dilation")

        self.grid_k = grid_size + spline_order # 样条基函数的数量 (不含base)
        self.params_per_input_channel = self.grid_k + 1 # 每个输入通道连接的总参数数量 (样条 + 基础)

        self.dropout = None
        if dropout > 0:
            if ndim == 1:
                self.dropout = nn.Dropout1d(p=dropout)
            elif ndim == 2:
                self.dropout = nn.Dropout2d(p=dropout)
            elif ndim == 3:
                self.dropout = nn.Dropout3d(p=dropout)
            else: # 作为后备或针对其他情况
                self.dropout = nn.Dropout(p=dropout)


        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if input_dim % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output_dim must be divisible by groups')

        self.input_dim_per_group = input_dim // groups
        self.output_dim_per_group = output_dim // groups


        self.o_batch_size = o_batch_size
        self.layer_norm = nn.ModuleList([norm_class(output_dim // groups, **norm_kwargs) for _ in range(groups)])

        self.prelus = nn.ModuleList([nn.PReLU() for _ in range(groups)])

        h = (self.grid_range[1] - self.grid_range[0]) / grid_size

        self.grid = torch.linspace(
            self.grid_range[0] - h * spline_order,
            self.grid_range[1] + h * spline_order,
            grid_size + 2 * spline_order + 1, # 总共 grid_size + 2*k + 1 个点
            dtype=torch.float32
        )


    def forward(self, x: torch.Tensor, layer_weight: torch.Tensor):


        split_x = torch.split(x, self.input_dim_per_group, dim=1)
        output_chunks = []

        for group_ind, x_group in enumerate(split_x):

            start_o_idx = group_ind * self.output_dim_per_group
            end_o_idx = (group_ind + 1) * self.output_dim_per_group
            
            # weights_for_output_group 的形状: (O_group, I_total * params_per_input_channel, K_dims...)
            weights_for_output_group = layer_weight[start_o_idx:end_o_idx, :, ...]

            # 2. 从这些权重中，选择对应于当前输入组的部分
            #    每个输入通道贡献 params_per_input_channel 个参数
            start_i_param_idx = group_ind * self.input_dim_per_group * self.params_per_input_channel
            end_i_param_idx = (group_ind + 1) * self.input_dim_per_group * self.params_per_input_channel
            
            # current_group_layer_weight 的形状: (O_group, I_group * params_per_input_channel, K_dims...)
            current_group_layer_weight = weights_for_output_group[:, start_i_param_idx:end_i_param_idx, ...]
            
            y_group = self.forward_kan(x_group, group_ind, current_group_layer_weight)
            output_chunks.append(y_group)
        
        y = torch.cat(output_chunks, dim=1)
        return y

    def calculate_spline_basis_maps(self, x_group: torch.Tensor) -> torch.Tensor:
        # x_group: (N, I_group, Input_Spatial_Dims...)
        # 输出: (N, I_group * self.grid_k, Input_Spatial_Dims...)

        N_batch = x_group.shape[0]
        I_group = self.input_dim_per_group # x_group.shape[1]
        input_spatial_dims = x_group.shape[2:]


        grid_view_dims = [1] * (self.ndim + 1) + [-1] # e.g., [1,1,1,-1] for ndim=2
        # target_expand_shape: (I_group, *input_spatial_dims, G_total_points)
        target_expand_shape = list(x_group.shape[1:]) #获取 (I_group, *input_spatial_dims)
        target_expand_shape.append(self.grid.shape[0])

        grid_ready_for_broadcast = self.grid.view(*grid_view_dims).expand(target_expand_shape).contiguous().to(x_group.device)
        grid_ready_for_broadcast = grid_ready_for_broadcast.unsqueeze(0) 


        x_uns = x_group.unsqueeze(-1) 

        bases = ((x_uns >= grid_ready_for_broadcast[..., :-1]) & (x_uns < grid_ready_for_broadcast[..., 1:])).to(x_group.dtype)

        epsilon = 1e-8 
        for k_order in range(1, self.spline_order + 1):
            # grid_... slicing needs to align with bases_... slicing
            # All these are (1, I_group, *spatial_dims, relevant_grid_pts)
            g_left = grid_ready_for_broadcast[..., :-(k_order + 1)]
            g_right = grid_ready_for_broadcast[..., k_order:-1]
            
            delta_prev = g_right - g_left
            delta_prev = torch.where(delta_prev == 0, torch.ones_like(delta_prev) * epsilon, delta_prev) # Avoid division by zero

            g_k_plus_1 = grid_ready_for_broadcast[..., k_order + 1:]
            g_1_minus_k = grid_ready_for_broadcast[..., 1:(-k_order)]
            delta_next = g_k_plus_1 - g_1_minus_k
            delta_next = torch.where(delta_next == 0, torch.ones_like(delta_next) * epsilon, delta_next) # Avoid division by zero

            term1_num = x_uns - g_left
            term1 = (term1_num / delta_prev) * bases[..., :-1]
            
            term2_num = g_k_plus_1 - x_uns
            term2 = (term2_num / delta_next) * bases[..., 1:]
            bases = term1 + term2
        
        bases = bases.contiguous() # Shape: (N, I_group, *input_spatial_dims, self.grid_k)
        
        permute_dims = [0, 1, self.ndim + 2] + list(range(2, self.ndim + 2))
        bases_permuted = bases.permute(*permute_dims).contiguous()
        
        final_shape_channels = I_group * self.grid_k
        bases_reshaped = bases_permuted.view(N_batch, final_shape_channels, *input_spatial_dims)
        
        return bases_reshaped


    def forward_kan(self, x_group: torch.Tensor, group_index: int, current_group_layer_weight: torch.Tensor):
        # x_group shape: (N, I_group, Input_Spatial_Dims...)
        # current_group_layer_weight shape: (O_group, I_group * params_per_input_channel, Kernel_dims...)

        N_batch = x_group.shape[0]
        I_group = self.input_dim_per_group
        O_group = self.output_dim_per_group

        # 1. Select convolution function based on ndim
        if self.ndim == 1:
            conv_fn = F.conv1d
        elif self.ndim == 2:
            conv_fn = F.conv2d
        elif self.ndim == 3:
            conv_fn = F.conv3d
        else:
            raise ValueError(f"Unsupported ndim: {self.ndim}")

        reshaped_weights = current_group_layer_weight.view(
            O_group, 
            I_group, 
            self.params_per_input_channel, # grid_k + 1
            *self.kernel_size_tuple
        )

        # Base weights: params_per_input_channel index 0
        base_weight_params = reshaped_weights[:, :, 0, ...] 
        # Shape: (O_group, I_group, *kernel_size_tuple) - ready for conv_fn

        # Spline weights: params_per_input_channel index 1 onwards
        spline_weight_params = reshaped_weights[:, :, 1:, ...]
        # Shape: (O_group, I_group, grid_k, *kernel_size_tuple)
        # Reshape for conv_fn: (O_group, I_group * grid_k, *kernel_size_tuple)
        spline_weight_full_group = spline_weight_params.reshape(
            O_group,
            I_group * self.grid_k,
            *self.kernel_size_tuple
        )

        # 3. Precompute features for convolutions
        x_activated = self.base_activation(x_group) # For base convolution
        spline_basis_as_input_maps = self.calculate_spline_basis_maps(x_group) # For spline convolution
                                                                          # Shape: (N, I_group * grid_k, *Input_Spatial_Dims)

        # 4. Calculate output spatial shape (once)
        output_spatial_shape = []
        input_spatial_dims = x_group.shape[2:]
        for d in range(self.ndim):
            L_in = input_spatial_dims[d]
            P = self.padding_tuple[d]
            DIL = self.dilation_tuple[d] # Corrected from D
            K = self.kernel_size_tuple[d]
            S = self.stride_tuple[d]
            L_out = (L_in + 2 * P - DIL * (K - 1) - 1) // S + 1
            output_spatial_shape.append(L_out)
        output_spatial_shape = tuple(output_spatial_shape)

        # Initialize full output tensor for this group's computation
        group_conv_output_accumulator = torch.zeros(
            N_batch, O_group, *output_spatial_shape,
            device=x_group.device, dtype=x_group.dtype
        )

        # 5. Loop over output channels in batches
        for o_start in range(0, O_group, self.o_batch_size):
            o_end = min(o_start + self.o_batch_size, O_group)
            # current_o_actual_batch_size = o_end - o_start # Not strictly needed for slicing

            # Slice weights for the current batch of output channels
            # base_weight_params shape: (O_group, I_group, *kernel_size_tuple)
            base_weight_batch = base_weight_params[o_start:o_end, ...]
            # spline_weight_full_group shape: (O_group, I_group * grid_k, *kernel_size_tuple)
            spline_weight_batch = spline_weight_full_group[o_start:o_end, ...]
            
            # Perform convolutions for this batch (groups=1 as input x_group/spline_basis is for this group only)
            current_base_conv_out = conv_fn(
                x_activated, base_weight_batch,
                stride=self.stride_tuple, padding=self.padding_tuple,
                dilation=self.dilation_tuple, groups=1 
            )
            current_spline_conv_out = conv_fn(
                spline_basis_as_input_maps, spline_weight_batch,
                stride=self.stride_tuple, padding=self.padding_tuple,
                dilation=self.dilation_tuple, groups=1
            )
            
            # Accumulate results
            group_conv_output_accumulator[:, o_start:o_end, ...] = current_base_conv_out + current_spline_conv_out
        

        normalized_output = self.layer_norm[group_index](group_conv_output_accumulator)
        activated_output = self.prelus[group_index](normalized_output)


        if self.dropout is not None:
            final_group_output = self.dropout(activated_output)
        else:
            final_group_output = activated_output

        return final_group_output



# class MetaKANConvNDLayer(nn.Module):
#     def __init__(self, conv_class, norm_class, input_dim, output_dim, spline_order, kernel_size,
#                  groups=1, padding=0, stride=1, dilation=1,
#                  ndim: int = 2, grid_size=5, base_activation=nn.GELU, grid_range=[-1, 1], dropout=0.0,
#                  **norm_kwargs):
#         super(MetaKANConvNDLayer, self).__init__()
#         self.inputdim = input_dim
#         self.outdim = output_dim
#         self.spline_order = spline_order
#         self.kernel_size = kernel_size
#         self.padding = padding
#         self.stride = stride
#         self.dilation = dilation
#         self.groups = groups
#         self.ndim = ndim
#         self.grid_size = grid_size
#         self.base_activation = base_activation()
#         self.grid_range = grid_range
#         self.norm_kwargs = norm_kwargs
#         self.grid_k = grid_size + spline_order
#         self.dropout = None
#         if dropout > 0:
#             if ndim == 1:
#                 self.dropout = nn.Dropout1d(p=dropout)
#             if ndim == 2:
#                 self.dropout = nn.Dropout2d(p=dropout)
#             if ndim == 3:
#                 self.dropout = nn.Dropout3d(p=dropout)
#         if groups <= 0:
#             raise ValueError('groups must be a positive integer')
#         if input_dim % groups != 0:
#             raise ValueError('input_dim must be divisible by groups')
#         if output_dim % groups != 0:
#             raise ValueError('output_dim must be divisible by groups')


#         self.layer_norm = nn.ModuleList([norm_class(output_dim // groups, **norm_kwargs) for _ in range(groups)])

#         self.prelus = nn.ModuleList([nn.PReLU() for _ in range(groups)])

#         h = (self.grid_range[1] - self.grid_range[0]) / grid_size
#         self.grid = torch.linspace(
#             self.grid_range[0] - h * spline_order,
#             self.grid_range[1] + h * spline_order,
#             grid_size + 2 * spline_order + 1,
#             dtype=torch.float32
#         )


#     def forward_kan(self, x, group_index, layer_weight):

#         # Apply base activation to input and then linear transform with base weights
#         base_weight = layer_weight[:, :self.inputdim // self.groups, :, :].view(self.outdim // self.groups, self.inputdim // self.groups, self.kernel_size, self.kernel_size)
#         base_output = F.conv2d(self.base_activation(x), base_weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

#         x_uns = x.unsqueeze(-1)  # Expand dimensions for spline operations.
#         # Compute the basis for the spline using intervals and input values.
#         target = x.shape[1:] + self.grid.shape
#         grid = self.grid.view(*list([1 for _ in range(self.ndim + 1)] + [-1, ])).expand(target).contiguous().to(
#             x.device)

#         bases = ((x_uns >= grid[..., :-1]) & (x_uns < grid[..., 1:])).to(x.dtype)

#         # Compute the spline basis over multiple orders.
#         for k in range(1, self.spline_order + 1):
#             left_intervals = grid[..., :-(k + 1)]
#             right_intervals = grid[..., k:-1]
#             delta = torch.where(right_intervals == left_intervals, torch.ones_like(right_intervals),
#                                 right_intervals - left_intervals)
#             bases = ((x_uns - left_intervals) / delta * bases[..., :-1]) + \
#                     ((grid[..., k + 1:] - x_uns) / (grid[..., k + 1:] - grid[..., 1:(-k)]) * bases[..., 1:])
#         bases = bases.contiguous()
#         bases = bases.moveaxis(-1, 2).flatten(1, 2)
#         spline_weight = layer_weight[:, self.inputdim // self.groups:, :, :].reshape(self.outdim // self.groups, self.inputdim // self.groups * self.grid_k, self.kernel_size, self.kernel_size) 
#         spline_output = F.conv2d(bases, spline_weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

#         x = self.prelus[group_index](self.layer_norm[group_index](base_output + spline_output))

#         if self.dropout is not None:
#             x = self.dropout(x)

#         return x

#     def forward(self, x, layer_weight):
#         split_x = torch.split(x, self.inputdim // self.groups, dim=1)
#         output = []
#         for group_ind, _x in enumerate(split_x):
#             y = self.forward_kan(_x, group_ind, layer_weight)
#             output.append(y.clone())
#         y = torch.cat(output, dim=1)
#         return y


class MetaKANConv3DLayer(MetaKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, spline_order=3, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=5, base_activation=nn.GELU, grid_range=[-1, 1], dropout=0.0, norm_layer=nn.InstanceNorm3d,
                 **norm_kwargs):
        super(MetaKANConv3DLayer, self).__init__(nn.Conv3d, norm_layer,
                                             input_dim, output_dim,
                                             spline_order, kernel_size,
                                             groups=groups, padding=padding, stride=stride, dilation=dilation,
                                             ndim=3,
                                             grid_size=grid_size, base_activation=base_activation,
                                             grid_range=grid_range, dropout=dropout, **norm_kwargs)


class MetaKANConv2DLayer(MetaKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, spline_order=3, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=5, base_activation=nn.GELU, grid_range=[-1, 1], dropout=0.0, norm_layer=nn.InstanceNorm2d,
                 **norm_kwargs):
        super(MetaKANConv2DLayer, self).__init__(nn.Conv2d, norm_layer,
                                             input_dim, output_dim,
                                             spline_order, kernel_size,
                                             groups=groups, padding=padding, stride=stride, dilation=dilation,
                                             ndim=2,
                                             grid_size=grid_size, base_activation=base_activation,
                                             grid_range=grid_range, dropout=dropout, **norm_kwargs)


class MetaKANConv1DLayer(MetaKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, spline_order=3, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=5, base_activation=nn.GELU, grid_range=[-1, 1], dropout=0.0, norm_layer=nn.InstanceNorm1d,
                 **norm_kwargs):
        super(MetaKANConv1DLayer, self).__init__(nn.Conv1d, norm_layer,
                                             input_dim, output_dim,
                                             spline_order, kernel_size,
                                             groups=groups, padding=padding, stride=stride, dilation=dilation,
                                             ndim=1,
                                             grid_size=grid_size, base_activation=base_activation,
                                             grid_range=grid_range, dropout=dropout, **norm_kwargs)


class WeightDecay(nn.Module):
    def __init__(self, module, weight_decay, name: str = None):
        if weight_decay < 0.0:
            raise ValueError(
                "Regularization's weight_decay should be greater than 0.0, got {}".format(
                    weight_decay
                )
            )

        super().__init__()
        self.module = module
        self.weight_decay = weight_decay
        self.name = name

        self.hook = self.module.register_full_backward_hook(self._weight_decay_hook)

    def remove(self):
        self.hook.remove()

    def _weight_decay_hook(self, *_):
        if self.name is None:
            for param in self.module.parameters():
                if param.grad is None or torch.all(param.grad == 0.0):
                    param.grad = self.regularize(param)
        else:
            for name, param in self.module.named_parameters():
                if self.name in name and (
                    param.grad is None or torch.all(param.grad == 0.0)
                ):
                    param.grad = self.regularize(param)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def extra_repr(self) -> str:
        representation = "weight_decay={}".format(self.weight_decay)
        if self.name is not None:
            representation += ", name={}".format(self.name)
        return representation

    @abc.abstractmethod
    def regularize(self, parameter):
        pass


class L2(WeightDecay):
    r"""Regularize module's parameters using L2 weight decay.

    Example::

        import torchlayers as tl

        # Regularize only weights of Linear module
        regularized_layer = tl.L2(tl.Linear(30), weight_decay=1e-5, name="weight")

    .. note::
            Backward hook will be registered on `module`. If you wish
            to remove `L2` regularization use `remove()` method.

    Parameters
    ----------
    module : torch.nn.Module
        Module whose parameters will be regularized.
    weight_decay : float
        Strength of regularization (has to be greater than `0.0`).
    name : str, optional
        Name of parameter to be regularized (if any).
        Default: all parameters will be regularized (including "bias").

    """

    def regularize(self, parameter):
        return self.weight_decay * parameter.data


class L1(WeightDecay):
    """Regularize module's parameters using L1 weight decay.

    Example::

        import torchlayers as tl

        # Regularize all parameters of Linear module
        regularized_layer = tl.L1(tl.Linear(30), weight_decay=1e-5)

    .. note::
            Backward hook will be registered on `module`. If you wish
            to remove `L1` regularization use `remove()` method.

    Parameters
    ----------
    module : torch.nn.Module
        Module whose parameters will be regularized.
    weight_decay : float
        Strength of regularization (has to be greater than `0.0`).
    name : str, optional
        Name of parameter to be regularized (if any).
        Default: all parameters will be regularized (including "bias").

    """

    def regularize(self, parameter):
        return self.weight_decay * torch.sign(parameter.data)


class KANLayer(nn.Module):
    def __init__(self, input_features, output_features, grid_size=5, spline_order=3, base_activation=nn.GELU,
                 grid_range=[-1, 1]):
        super(KANLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # The number of points in the grid for the spline interpolation.
        self.grid_size = grid_size
        # The order of the spline used in the interpolation.
        self.spline_order = spline_order
        # Activation function used for the initial transformation of the input.
        self.base_activation = base_activation()
        # The range of values over which the grid for spline interpolation is defined.
        self.grid_range = grid_range

        # Initialize the base weights with random values for the linear transformation.
        self.base_weight = nn.Parameter(torch.randn(output_features, input_features))
        # Initialize the spline weights with random values for the spline transformation.
        self.spline_weight = nn.Parameter(torch.randn(output_features, input_features, grid_size + spline_order))
        # Add a layer normalization for stabilizing the output of this layer.
        self.layer_norm = nn.LayerNorm(output_features)
        # Add a PReLU activation for this layer to provide a learnable non-linearity.
        self.prelu = nn.PReLU()

        # Compute the grid values based on the specified range and grid size.
        h = (self.grid_range[1] - self.grid_range[0]) / grid_size
        self.grid = torch.linspace(
            self.grid_range[0] - h * spline_order,
            self.grid_range[1] + h * spline_order,
            grid_size + 2 * spline_order + 1,
            dtype=torch.float32
        ).expand(input_features, -1).contiguous()

        # Initialize the weights using Kaiming uniform distribution for better initial values.
        nn.init.kaiming_uniform_(self.base_weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.spline_weight, nonlinearity='linear')

    def forward(self, x):
        # Process each layer using the defined base weights, spline weights, norms, and activations.
        grid = self.grid.to(x.device)
        # Move the input tensor to the device where the weights are located.

        # Perform the base linear transformation followed by the activation function.
        base_output = F.linear(self.base_activation(x), self.base_weight)
        x_uns = x.unsqueeze(-1)  # Expand dimensions for spline operations.
        # Compute the basis for the spline using intervals and input values.
        bases = ((x_uns >= grid[:, :-1]) & (x_uns < grid[:, 1:])).to(x.dtype).to(x.device)

        # Compute the spline basis over multiple orders.
        for k in range(1, self.spline_order + 1):
            left_intervals = grid[:, :-(k + 1)]
            right_intervals = grid[:, k:-1]
            delta = torch.where(right_intervals == left_intervals, torch.ones_like(right_intervals),
                                right_intervals - left_intervals)
            bases = ((x_uns - left_intervals) / delta * bases[:, :, :-1]) + \
                    ((grid[:, k + 1:] - x_uns) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:])
        bases = bases.contiguous()

        # Compute the spline transformation and combine it with the base transformation.
        spline_output = F.linear(bases.view(x.size(0), -1), self.spline_weight.view(self.spline_weight.size(0), -1))
        # Apply layer normalization and PReLU activation to the combined output.
        x = self.prelu(self.layer_norm(base_output + spline_output))

        return x

class KAN(nn.Module):  # Kolmogorov Arnold Legendre Network (KAL-Net)
    def __init__(self, layers_hidden, dropout: float = 0.0, grid_size=5, spline_order=3, base_activation=nn.GELU,
                 grid_range: List = [-1, 1], l1_decay: float = 0.0, first_dropout: bool = True, **kwargs):
        super(KAN, self).__init__()  # Initialize the parent nn.Module class

        # layers_hidden: A list of integers specifying the number of neurons in each layer
        self.layers_hidden = layers_hidden
        # polynomial_order: Order up to which Legendre polynomials are calculated
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation = base_activation
        self.grid_range = grid_range

        # list of layers
        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.base_activation = base_activation
        self.num_layers = len(layers_hidden[:-1])

        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            layer = KANLayer(in_features, out_features, grid_size=grid_size, spline_order=spline_order,
                             base_activation=base_activation, grid_range=grid_range)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)
            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def linear_layer(input_dim, output_dim):
    linear = nn.Linear(input_dim, output_dim)
    nn.init.xavier_normal_(linear.weight)
    nn.init.constant_(linear.bias, 0)
    return linear

class MetaLearner(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MetaLearner, self).__init__()
        self.layers = nn.Sequential(
            linear_layer(input_dim, hidden_dim),
            nn.ReLU(),
            linear_layer(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.layers(x)

class SimpleMetaConvKAN(nn.Module):
    def __init__(self, args):
        super(SimpleMetaConvKAN, self).__init__()        
        layer_sizes = args.layer_sizes
        num_classes = args.num_classes
        input_channels = args.input_channel
        spline_order = args.spline_order
        degree_out = args.degree_out
        groups = args.groups
        dropout = args.dropout
        dropout_linear = args.dropout_linear
        affine = args.affine
        embedding_dim = args.embedding_dim
        hidden_dim = args.hidden_dim
        grid_size = args.grid_size

        if args.norm_layer == 'batch':
            norm_layer = nn.BatchNorm2d
        elif args.norm_layer == 'instance':
            norm_layer = nn.InstanceNorm2d
        elif args.norm_layer == 'layer':
            norm_layer = nn.LayerNorm
        else:
            NotImplementedError('Norm layer not implemented')        

        self.layer_sizes = [input_channels]+layer_sizes
        self.grid_k = grid_size + spline_order

        self.metanet = MetaLearner(input_dim=embedding_dim, output_dim=self.grid_k+1, hidden_dim=hidden_dim)

        # 初始化每一层的嵌入向量列表 nn.ParameterList
        self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(input_channels * layer_sizes[0]*3*3, embedding_dim))])
        for i in range(len(layer_sizes) - 1):
            self.embeddings.append(nn.Parameter(torch.randn(layer_sizes[i] * layer_sizes[i + 1]*3*3, embedding_dim)))
        self.layers = nn.Sequential(
            MetaKANConv2DLayer(input_channels, layer_sizes[0], kernel_size=3, grid_size = grid_size,spline_order=spline_order, groups=1,
                           padding=1, stride=1, dilation=1, affine=affine, norm_layer=norm_layer),
            MetaKANConv2DLayer(layer_sizes[0], layer_sizes[1], kernel_size=3, grid_size = grid_size,spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKANConv2DLayer(layer_sizes[1], layer_sizes[2], kernel_size=3, grid_size = grid_size,spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
            
            MetaKANConv2DLayer(layer_sizes[2], layer_sizes[3], kernel_size=3, grid_size = grid_size,spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
            

        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))        
        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[3], num_classes))
        else:

            self.output = KAN([layer_sizes[3], num_classes], dropout=dropout_linear,
                              first_dropout=True, spline_order=spline_order)

    def _initialize_embeddings(self):
        """ 对嵌入向量进行 Xavier 初始化 """
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding)


    def forward(self, x):
        for i, layer in enumerate(self.layers):
            embedding = self.embeddings[i]
            layer_weight = self.metanet(embedding).reshape(self.layer_sizes[i+1], self.layer_sizes[i]*(self.grid_k+1), 3, 3)
            x = layer(x, layer_weight)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x

class EightSimpleMetaConvKAN(nn.Module):
    def __init__(self, args):
        super(EightSimpleMetaConvKAN, self).__init__()
        
        layer_sizes = args.layer_sizes
        num_classes = args.num_classes
        input_channels = args.input_channels
        spline_order = args.spline_order
        degree_out = args.degree_out
        groups = args.groups
        grid_size = args.grid_size
        dropout = args.dropout
        dropout_linear = args.dropout_linear
        l1_penalty = args.l1_penalty
        affine = args.affine
        embedding_dim = args.embedding_dim
        hidden_dim = args.hidden_dim

        
        if args.norm_layer == 'batch':
            norm_layer = nn.BatchNorm2d
        elif args.norm_layer == 'instance':
            norm_layer = nn.InstanceNorm2d
        elif args.norm_layer == 'layer':
            norm_layer = nn.LayerNorm
        else:
            raise NotImplementedError('Norm layer not implemented')
        self.layer_sizes = [input_channels]+layer_sizes
        self.grid_k = grid_size + spline_order

        self.metanet = MetaLearner(input_dim=embedding_dim, output_dim=self.grid_k+1, hidden_dim=hidden_dim)

        # 初始化每一层的嵌入向量列表 nn.ParameterList
        self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(input_channels * layer_sizes[0]*3*3, embedding_dim))])
        for i in range(len(layer_sizes) - 1):
            self.embeddings.append(nn.Parameter(torch.randn(layer_sizes[i] * layer_sizes[i + 1]*3*3, embedding_dim)))
        self.layers = nn.Sequential(
            MetaKANConv2DLayer(input_channels, layer_sizes[0], kernel_size=3, spline_order=spline_order, groups=1,
                           padding=1, stride=1, dilation=1, affine=affine, norm_layer=norm_layer),
            MetaKANConv2DLayer(layer_sizes[0], layer_sizes[1], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
            
            MetaKANConv2DLayer(layer_sizes[1], layer_sizes[2], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
            
            MetaKANConv2DLayer(layer_sizes[2], layer_sizes[3], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
            
            MetaKANConv2DLayer(layer_sizes[3], layer_sizes[4], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKANConv2DLayer(layer_sizes[4], layer_sizes[5], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKANConv2DLayer(layer_sizes[5], layer_sizes[6], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKANConv2DLayer(layer_sizes[6], layer_sizes[7], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               

        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[7], num_classes))
        else:
            self.output = KAN([layer_sizes[7], num_classes], dropout=dropout_linear,
                              first_dropout=True, spline_order=spline_order)

    def _initialize_embeddings(self):
        """ 对嵌入向量进行 Xavier 初始化 """
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding)


    def forward(self, x):
        for i, layer in enumerate(self.layers):
            embedding = self.embeddings[i]
            layer_weight = self.metanet(embedding).reshape(self.layer_sizes[i+1], self.layer_sizes[i]*(self.grid_k+1), 3, 3)
            x = layer(x, layer_weight)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x
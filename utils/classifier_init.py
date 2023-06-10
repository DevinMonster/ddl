import torch


def kaiming_init(params, state_dict, num_classes):
    cf_name = params['classifier_name']
    tot_cls = sum(num_classes)
    weights_shape = list(state_dict[f'{cf_name}.weight'].data.shape)
    weights_shape[0] = tot_cls
    weights = torch.zeros(weights_shape)
    torch.nn.init.kaiming_normal_(weights)
    state_dict[f'{cf_name}.weight'].data = weights
    state_dict[f'{cf_name}.bias'].data = torch.rand(tot_cls)
    return state_dict


def xavier_init(params, state_dict, num_classes):
    cf_name = params['classifier_name']
    tot_cls = sum(num_classes)
    weights_shape = list(state_dict[f'{cf_name}.weight'].data.shape)
    weights_shape[0] = tot_cls
    weights = torch.zeros(weights_shape)
    torch.nn.init.xavier_normal_(weights)
    state_dict[f'{cf_name}.weight'].data = weights
    state_dict[f'{cf_name}.bias'].data = torch.rand(tot_cls)
    return state_dict


def rand_new_init(params, state_dict, num_classes):
    cf_name = params['classifier_name']
    weight_shape = [num_classes[-1]] + list(state_dict[f'{cf_name}.weight'][0].shape)
    bias_shape = [num_classes[-1]] + list(state_dict[f'{cf_name}.bias'][0].shape)
    new_cls_weights = torch.randn(weight_shape)
    new_cls_bias = torch.randn(bias_shape)
    state_dict[f'{cf_name}.weight'].data = torch.cat((state_dict[f'{cf_name}.weight'].data, new_cls_weights),
                                                     dim=0)
    state_dict[f'{cf_name}.bias'].data = torch.cat((state_dict[f'{cf_name}.bias'].data, new_cls_bias), dim=0)
    return state_dict
import network


def GetNet(model_name, in_channel=3, num_class=6, pretrained_path=None):
    """ 
    in_channel -> 输入通道数
    num_class -> 输出分类数
    pretrain_path -> 预训练权重文件
    """
    return getattr(network, model_name)(in_channel, num_class, pretrained_path=pretrained_path)

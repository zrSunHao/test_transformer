class DefaultConfig(object):

    device = 'cuda'                         # 设备
    print_every = 200                       # 进度输出频率
    epoch_max = 40                          # 训练轮次
    epoch_current = 1                       # 当前轮次

    data_root = 'D:/WorkSpace/DataSet/translation/'      # 训练集根目录
    dict_zh_path = 'dict.zh-cn'                         # 中文词编码
    dict_en_path = 'dict.en'                            # 英文词编码
    train_zh_path = 'train.zh-cn.bped'                  # ted 中文平行语料
    train_en_path = 'train.en.bped'                     # ted 英文平行语料

    net_save_root = './checkpoints'         # 模型参数保存目录
    net_path = 'transformer_.pth'           # 训练好的模型

    image_size = 256                        # 图像尺寸
    batch_size = 4                          # 每批次图像数量
    num_workers = 0                         # 进程数
    lr = 5e-5                               # 学习率

    d_model = 512                           # 字 Embedding 的维度
    d_ff = 2048                             # 前向传播隐藏层维度
    d_k = d_v = 64                          # K(=Q), V的维度 
    n_layers = 6                            # 有多少个encoder和decoder
    n_heads = 8                             # Multi-Head Attention设置为8                        
    

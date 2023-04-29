from models import make_model
from dataprovider import real_data_gen
from tools import run_epoch, get_std_opt, LabelSmoothing, SimpleLossCompute
from config import DefaultConfig

cfg = DefaultConfig()

# wc -l dict.en dict.zh-cn
V_zh = 12203 + 2    # 中文词表的长度，dict.zh-cn 的行数
V_en = 30684 + 2    # 英文词表的长度，dict.en 的行数

criterion = LabelSmoothing(size = V_zh, 
                           padding_idx = 0, 
                           smoothing = 0.0)
model = make_model(src_vocab = V_en, 
                   tgt_vocab = V_zh, 
                   N = 2)
optim = get_std_opt(model=model)

for epoch in range(10):
    model.train()
    data_iter = real_data_gen(nbatches = cfg.batch_size,
                              dict_zh_path = cfg.data_root + cfg.dict_zh_path,
                              dict_en_path = cfg.data_root + cfg.dict_en_path,
                              train_zh_path = cfg.data_root + cfg.train_zh_path,
                              train_en_path = cfg.data_root + cfg.train_en_path,)
    loss_compute = SimpleLossCompute(model.generator, criterion, optim)
    run_epoch(data_iter = data_iter, 
              model = model,
              loss_compute = loss_compute)


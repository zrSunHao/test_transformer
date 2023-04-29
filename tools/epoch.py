import time

'''
Standard Training and Logging Function
    data_iter:  数据加载器
    model:      模型
    loss_compute:   损失计算
'''
def run_epoch(data_iter, model, loss_compute):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0

    for i, batch in enumerate(data_iter):
        src = batch.src
        tgt = batch.tgt
        src_mask = batch.src_mask
        tgt_mask = batch.tgt_mask
        out = model.forward(src, tgt, src_mask, tgt_mask)

        tgt_y = batch.tgt_y
        ntokens = batch.ntokens
        loss = loss_compute(out, tgt_y, ntokens)
        total_loss += loss
        total_tokens += ntokens
        tokens += ntokens

        if i%50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                   (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
            
    return total_loss / total_tokens        

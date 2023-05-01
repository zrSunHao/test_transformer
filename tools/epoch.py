import time

'''
Standard Training and Logging Function
    data_iter:  数据加载器
    model:      模型
    loss_compute:   损失计算
'''
def run_epoch(data_iter, model, loss_compute, device='cpu'):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0

    for i, batch in enumerate(data_iter):
        src = batch.src.to(device)
        tgt = batch.tgt.to(device)
        src_mask = batch.src_mask.to(device)
        tgt_mask = batch.tgt_mask.to(device)

        out = model.forward(src, tgt, src_mask, tgt_mask)

        tgt_y = batch.tgt_y
        ntokens = batch.ntokens     # 当前batch目标语言的token总数
        
        loss = loss_compute(out, tgt_y, ntokens)
        total_loss += loss
        total_tokens += ntokens
        tokens += ntokens

        print(i+1)
        if i!=0 and i%50 == 0:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                   (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
            
    return total_loss / total_tokens        

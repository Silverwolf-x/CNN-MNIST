import torch
from set_config import config,save_model
from tqdm import tqdm
import logging
def trainer(train_loader, valid_loader, model):
    #===prepare===
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.NAdam(model.parameters())# 变化的学习率
    early_stop_count = 0
    record = {
        'train_loss': [],
        'valid_loss': [],
        'valid_acc': [],
        'best_loss': 1e5,
        'best_epoch': 0
    }

    for epoch in range(config.n_epoches):
        #===train mode===
        model.train()
        train_loss = 0
        train_loop = tqdm(train_loader, leave=0, mininterval=1,maxinterval=float('inf'),dynamic_ncols=True)
        for x, y in train_loop:
            x, y = x.to(config.device), y.to(config.device)
            y_pred = model(x)
            # targets的类型是要求long(int64)，这里对齐
            loss = criterion(y_pred, y.long())
            # 清零梯度，反向传播，更新权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 进度条设置
            L  = loss.item()
            train_loop.set_description(f'Epoch [{epoch}/{config.n_epoches}]')
            train_loop.set_postfix({'loss':L},refresh=False)
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader.dataset)
        record['train_loss'].append(train_loss)

        #===evaluate mode===
        model.eval()
        valid_loss = 0
        correct = 0
        for x, y in valid_loader:
            x, y = x.to(config.device), y.to(config.device)
            with torch.no_grad():  # 减少内存损耗
                output = model(x)
                loss = criterion(output, y.long())
                pred = output.argmax(dim=1)

                correct += pred.eq(y).sum()
                valid_loss += loss.item()
        valid_accuracy = correct / len(valid_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)
        record['valid_loss'].append(valid_loss)
        record['valid_acc'].append(valid_accuracy)

        #===early stopping===
        if record['valid_loss'][-1] < record['best_loss']:
            record['best_loss'] = record['valid_loss'][-1]
            record['best_epoch'] = epoch
            logging.info(
                f"Now model with loss {record['best_loss']:.2e}, valid accuracy {record['valid_acc'][-1]:.4f}... from epoch {epoch}"
            )
            early_stop_count = 0
        else:
            early_stop_count += 1
        if early_stop_count >= config.early_stop:
            logging.info(
                f'Model is not improving for {config.early_stop} epoches. The last epoch is {epoch}.'
            )
            break
    torch.save(model.state_dict(), save_model(record['best_loss']))
    logging.info(
        f"Saving model with loss {record['best_loss']:.2e}... from epoch {record['best_epoch']}"
    )
    return record['train_loss'], record['valid_loss'], record['best_loss']


def predict(test_data, model):
    '''注意这里载入data不是loader一批批载入
    返回pred的值，错误率，错误的坐标'''
    model.eval()
    preds = []
    incorrect_index = []
    for i, (x, y) in tqdm(enumerate(test_data), position=0, ncols=100):
        # (B, 28, 28)-->(B, 1, 28, 28)
        x = torch.unsqueeze(x, dim=1).to(config.device)
        with torch.no_grad():
            output = model(x)
            y_pred = output.argmax(dim=1).cpu().numpy().squeeze()
            preds.append(y_pred)
            if (y_pred != y):
                incorrect_index.append(i)
    return preds, 1 - len(incorrect_index) / len(test_data), incorrect_index
# -*- coding: utf-8 -*-
import sys, os
import argparse
from time import time
from xml.sax.handler import feature_string_interning

sys.path.append(os.getcwd())
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Process.process import *
import torch
import torch.nn.functional as F
import numpy as np
from tools.earlystopping import EarlyStopping
from Process.rand5fold import *
from tools.evaluate import *
from Bert import data_loader
import logging
from label_smooth_loss import LabelSmoothingLoss
from origin_multi import MultiModalBERTClass##########需要改回来
from transformers import get_linear_schedule_with_warmup

def train_model(x_test, x_train, args, iter):
    train_list_tweet_source, test_list_tweet_source, num_columns = loadData(x_test, x_train)

    num_training_steps = len(train_list_tweet_source) * args.n_epochs // args.batchsize
    num_warmup_steps = num_training_steps // 10

    num_estimators = 5
    models = [MultiModalBERTClass(args,pca_feature_size=num_columns).to(args.device) for _ in range(num_estimators)]#model实例
    #criterion = LabelSmoothingLoss(smoothing=0.1).to(args.device)#损失函数
    criterion = torch.nn.CrossEntropyLoss()
    optimizers = [torch.optim.AdamW(model.parameters(), lr=1e-4 , weight_decay=args.l2) for model in models]#优化器
    schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=5) for optimizer in optimizers]#学习率调整器
    weights = torch.nn.Parameter(torch.tensor([0.15, 0.15, 0.15, 0.15, 0.4]))
    optimizer_w = torch.optim.Adam([weights], lr=1e-3)


    train_losses = []  # 存放训练损失的列表
    val_losses = []  # 存放评估损失的列表
    train_accs = []  # 存放训练精确率的列表
    val_accs = []  # 存放评估精确率的列表
    # verbose：日志显示函数
    #
    # verbose = 0 为不在标准输出流输出日志信息
    # verbose = 1 为输出进度条记录
    # verbose = 2 为每一个epoch输出一行记录
    early_stopping = EarlyStopping(patience=args.patience,
                                   verbose=True)  # 采用早停止法， patience表明能容忍几次，delta表明能容忍在训练过程中val_loss 的上升的范围。每一个epoch都把val_loss和模型传入到该实例中,正常来说，随着训练的过程，val_loss应该跟train_loss一起变小，但过拟合时，train_loss 会降低，val_loss会升高，我们设置patience表示容忍几次升高就停止

    ids, attention_masks, token_type_ids = data_loader(train_list_tweet_source)#input_ids对应词汇表、attention_mask有效和填充、token_type_ids对输入句子两句区分
    ids_tensor = torch.LongTensor(ids).to(args.device)
    attention_masks_tensor = torch.LongTensor(attention_masks).to(args.device)#转成张量
    token_type_ids_tensor = torch.LongTensor(token_type_ids).to(args.device)
    # 第一个参数是加载的数据集，batch_size是批次数；shuffle若为true则可以在每个epoch中对数据进行打乱；num_workers用于数据加载的子进程数量，0标识数据将在主进程中加载
    tids, tattention_masks, ttoken_type_ids = data_loader(
        test_list_tweet_source)  # input_ids对应词汇表、attention_mask有效和填充、token_type_ids对输入句子两句区分
    tids_tensor = torch.LongTensor(tids).to(args.device)
    tattention_masks_tensor = torch.LongTensor(tattention_masks).to(args.device)  # 转成张量
    ttoken_type_ids_tensor = torch.LongTensor(ttoken_type_ids).to(args.device)
    best_acc = 0
    student_quality = {}
    for epoch in range(args.n_epochs):



        avg_loss = []  # 存放平均损失
        avg_acc = []  # 存放平均精确率
        batch_idx = 0
        index = 0
        total_len=len(train_list_tweet_source)
        print("strat train")
        y = [item['tag'] for item in train_list_tweet_source.values()]

        while index < total_len:
            output_set = []
            batch_data = list(train_list_tweet_source.values())[index:index + args.batchsize]

            batch_y = y[index:index + args.batchsize]
            batch_y_tensor = torch.tensor(batch_y).to(args.device)
            # train_y_tensor = torch.tensor(train_y[index:index + 128]).to(args.device)
            logging.basicConfig(filename="./logs.txt", level=logging.INFO)
            num_model = 1
            #for model, optimizer in zip(models, optimizers):
            for model, optimizer,scheduler in zip(models, optimizers,schedulers):
                # try:
                model.train()

                sample_lengths = []
                all_ids = []
                all_attention_masks = []
                all_token_type_ids = []
                features = []
                padding_count = 0
                # 收集所有对话数据
                for item in batch_data:
                    if num_model in item['tags']:
                        sample_lengths.append(len(item['tags'][num_model][0]))
                        for i in range(len(item['tags'][num_model][0])):
                            all_ids.append(item['tags'][num_model][0][i].unsqueeze(0))
                            all_attention_masks.append(item['tags'][num_model][1][i].unsqueeze(0))
                            all_token_type_ids.append(item['tags'][num_model][2][i].unsqueeze(0))
                            features.append((item['feature']).clone().detach())
                    elif 5 in item['tags']:
                        sample_lengths.append(len(item['tags'][5][0]))
                        for i in range(len(item['tags'][5][0])):
                            all_ids.append(item['tags'][5][0][i].unsqueeze(0))
                            all_attention_masks.append(item['tags'][5][1][i].unsqueeze(0))
                            all_token_type_ids.append(item['tags'][5][2][i].unsqueeze(0))
                            features.append((item['feature']).clone().detach())
                    elif 4 in item['tags']:
                        sample_lengths.append(len(item['tags'][4][0]))
                        for i in range(len(item['tags'][4][0])):
                            all_ids.append(item['tags'][4][0][i].unsqueeze(0))
                            all_attention_masks.append(item['tags'][4][1][i].unsqueeze(0))
                            all_token_type_ids.append(item['tags'][4][2][i].unsqueeze(0))
                            features.append((item['feature']).clone().detach())
                    else:
                        sample_lengths.append(1)
                        all_ids.append(torch.randint(0, 100, (1, 64), dtype=torch.long))
                        all_attention_masks.append(torch.ones(1, 64, dtype=torch.long))
                        all_token_type_ids.append(torch.zeros(1, 64, dtype=torch.long))
                        features.append((item['feature']).clone().detach())
                        padding_count += 1
                total_items = len(batch_data)
                padding_percentage = (padding_count / total_items) * 100
                print(f'Padding data percentage: {padding_percentage:.2f}%')

                ids_tensor = torch.cat(all_ids).to(args.device)
                attention_masks_tensor = torch.cat(all_attention_masks).to(args.device)
                token_type_ids_tensor = torch.cat(all_token_type_ids).to(args.device)
                feature_tensor = torch.stack(features).to(args.device)

                out_labels = model(ids_tensor, attention_masks_tensor, token_type_ids_tensor,feature_tensor, args).to(args.device)

                # 还原每个样本的输出并平均
                start_idx = 0
                batch_outputs = []
                for length in sample_lengths:
                    if length > 0:
                        sample_output = out_labels[start_idx:start_idx + length].clone()
                        # 计算每个预测的confidence
                        with torch.no_grad():
                            # 1. 计算更有区分度的confidence scores
                            logits = sample_output.detach()
                            probs = F.softmax(logits/0.3, dim=1)
                            max_probs, _ = probs.max(dim=1)

                            # 2. 添加温度参数增加差异
                            temperature = 0.5  # 温度参数(小于1增加差异)
                            confidence_scores = torch.pow(max_probs, 1 / temperature)

                            # 3. 设置confidence阈值
                            threshold = 0.6
                            mask = confidence_scores > threshold
                            if mask.sum() > 0:  # 如果有超过阈值的预测
                                confidence_scores = confidence_scores * mask.float()
                                weight = F.softmax(confidence_scores, dim=0)
                            else:  # 如果没有超过阈值的预测，使用原始scores
                                weight = F.softmax(confidence_scores, dim=0)

                            # 4. 可选：对权重进行幂次放大
                            weight = torch.pow(weight, 2)  # 二次方增加差异
                            weight = weight / weight.sum()  # 重新归一化
                        # 加权平均
                        weighted_avg = (sample_output * weight.unsqueeze(1)).sum(dim=0, keepdim=True)
                        batch_outputs.append(weighted_avg)
                    start_idx += length

                # 合并批次中所有样本的平均输出
                out_labels = torch.cat(batch_outputs).to(args.device)

                output_set.append(out_labels.clone().detach())
                # out_labels = torch.stack(out_labels_list, 0)
                # 此处是利用bert
                # out_labels = model.bert(ids_tensor[index:index+128],attention_masks_tensor[index:index+128],token_type_ids_tensor[index:index+128]).to(args.device)

                # 随机权重
                # out_labels = out_labels1*0.9 + out_labels3*0.1
                loss = criterion(out_labels,batch_y_tensor)  # 求输出的标签值和对应的标签值之间的损失值，详细计算方式见代码解读.txt      tensor(1.3131, device='cuda:0', grad_fn=<NllLossBackward>)
                # logging.info(loss)

                optimizer.zero_grad()  # 将模型中的参数梯度设为0。根据pytorch中的backward()函数的计算，当网络参量进行反馈时，梯度是被积累的而不是被替换掉；但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad 了
                loss.backward()  # 做完运算后使用tensor.backward()，所有的梯度就会自动运算，tensor的梯度将会累加到它的.grad属性里面去。
                avg_loss.append(loss.item())
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Current Learning Rate: {current_lr}')


                _, pred = out_labels.max(
                    dim=1)  # _.shape=[1,128],_获得的是out_lables中128个列表中每一个列表中的4个数据中的最大值,pred.shape=[1,128]，是预测的标签值
                correct = pred.eq(batch_y_tensor).sum().item()  # 计算Batch_data.y和pred中对应位置相同的数据的个数，也就是预测正确的个数

                train_acc = correct / len(batch_y)  # 训练准确率=预测正确个数/标签个数
                avg_acc.append(train_acc)
                logging.info((iter, epoch, batch_idx, loss.item(), train_acc))
                print("Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(iter,
                                                                                                                   epoch,
                                                                                                                   batch_idx,
                                                                                                                   loss.item(),
                                                                                                                   train_acc))
                num_model = num_model + 1
            weighted_output = torch.stack([w * o for w, o in zip(weights, output_set)]).sum(dim=0).to(args.device)
            weighted_output.requires_grad_(True)
            loss2 = criterion(weighted_output, batch_y_tensor)
            optimizer_w.zero_grad()  # 将模型中的参数梯度设为0。根据pytorch中的backward()函数的计算，当网络参量进行反馈时，梯度是被积累的而不是被替换掉；但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad 了
            loss2.backward()  # 做完运算后使用tensor.backward()，所有的梯度就会自动运算，tensor的梯度将会累加到它的.grad属性里面去。
            optimizer_w.step()
            #with torch.no_grad():
            #    weights.data = F.softmax(weights.data/0.5, dim=0)
            print(weights)

            batch_ids = list(train_list_tweet_source.keys())[index:index + args.batchsize]
            for i, student_id in enumerate(batch_ids):
                # 计算该学生的confidence score
                probs = F.softmax(weighted_output[i], dim=0)
                confidence = probs.max().item()

                # 更新学生质量信息
                if student_id not in student_quality:
                    student_quality[student_id] = {
                        'loss': loss2.item(),
                        'confidence': confidence,
                        'acc': pred[i].eq(batch_y_tensor[i]).item()
                    }
                else:
                    # 如果已存在，更新为更好的指标
                    if loss2.item() < student_quality[student_id]['loss']:
                        student_quality[student_id].update({
                            'loss': loss2.item(),
                            'confidence': confidence,
                            'acc': pred[i].eq(batch_y_tensor[i]).item()
                        })

            batch_idx = batch_idx + 1
            index = index + args.batchsize

            #except Exception as e:
            #    print(f"Error encountered: {e},in {iter},{epoch},{batch_idx} ")
            #    continue  # 跳过当前循环，继续下一个批次

        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))

        #######################验证阶段#######################
        print("strat val")
        index = 0
        temp_val_losses = []
        temp_val_accs = []
        temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
        temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
        temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        val_len=len(test_list_tweet_source)
        test_y = [item['tag'] for item in test_list_tweet_source.values()]
        with torch.no_grad():
            while index < val_len:
                output_set = []
                batch_data = list(test_list_tweet_source.values())[index:index + args.batchsize]
                batch_y = test_y[index:index + args.batchsize]
                batch_y_tensor = torch.tensor(batch_y).to(args.device)
                num_model = 1
                #for model, optimizer in zip(models, optimizers):
                for model, optimizer,scheduler in zip(models, optimizers,schedulers):
                    model.eval()
                    sample_lengths = []
                    all_ids = []
                    all_attention_masks = []
                    all_token_type_ids = []
                    features = []

                    for item in batch_data:
                        if num_model in item['tags']:
                            sample_lengths.append(len(item['tags'][num_model][0]))
                            for i in range(len(item['tags'][num_model][0])):
                                all_ids.append(item['tags'][num_model][0][i].unsqueeze(0))
                                all_attention_masks.append(item['tags'][num_model][1][i].unsqueeze(0))
                                all_token_type_ids.append(item['tags'][num_model][2][i].unsqueeze(0))
                                features.append((item['feature']).clone().detach())
                        elif 5 in item['tags']:
                            sample_lengths.append(len(item['tags'][5][0]))
                            for i in range(len(item['tags'][5][0])):
                                all_ids.append(item['tags'][5][0][i].unsqueeze(0))
                                all_attention_masks.append(item['tags'][5][1][i].unsqueeze(0))
                                all_token_type_ids.append(item['tags'][5][2][i].unsqueeze(0))
                                features.append((item['feature']).clone().detach())
                        elif 4 in item['tags']:
                            sample_lengths.append(len(item['tags'][4][0]))
                            for i in range(len(item['tags'][4][0])):
                                all_ids.append(item['tags'][4][0][i].unsqueeze(0))
                                all_attention_masks.append(item['tags'][4][1][i].unsqueeze(0))
                                all_token_type_ids.append(item['tags'][4][2][i].unsqueeze(0))
                                features.append((item['feature']).clone().detach())
                        else:
                            sample_lengths.append(1)
                            all_ids.append(torch.randint(0, 100, (1, 64), dtype=torch.long))
                            all_attention_masks.append(torch.ones(1, 64, dtype=torch.long))
                            all_token_type_ids.append(torch.zeros(1, 64, dtype=torch.long))
                            features.append((item['feature']).clone().detach())

                    ids_tensor = torch.cat(all_ids).to(args.device)
                    attention_masks_tensor = torch.cat(all_attention_masks).to(args.device)
                    token_type_ids_tensor = torch.cat(all_token_type_ids).to(args.device)
                    feature_tensor = torch.stack(features).to(args.device)

                    out_labels = model(ids_tensor, attention_masks_tensor, token_type_ids_tensor, feature_tensor,
                                       args).to(args.device)

                    # 还原每个样本的输出并平均
                    start_idx = 0
                    batch_outputs = []
                    for length in sample_lengths:
                        if length > 0:
                            sample_output = out_labels[start_idx:start_idx + length]
                            # 计算每个预测的confidence
                            confidence_scores = F.softmax(sample_output, dim=1).max(dim=1)[0]  # 获取最大概率值作为confidence
                            # 归一化confidence scores
                            weight = F.softmax(confidence_scores, dim=0)
                            # 加权平均
                            weighted_avg = torch.sum(sample_output * weight.unsqueeze(1), dim=0, keepdim=True)
                            batch_outputs.append(weighted_avg)
                        start_idx += length

                    # 合并批次中所有样本的平均输出
                    out_labels = torch.cat(batch_outputs).to(args.device)
                    output_set.append(out_labels)
                    num_model = num_model + 1

                weighted_output = torch.stack([w.detach() * o.detach() for w, o in zip(weights, output_set)]).sum(dim=0).to(args.device)
                val_loss = F.nll_loss(F.log_softmax(weighted_output, dim=1),
                                  batch_y_tensor)
                for scheduler in schedulers:
                    scheduler.step(val_loss)
                temp_val_losses.append(val_loss.item())
                _, val_pred = weighted_output.max(dim=1)

                for true_class in range(4):
                    true_indices = (batch_y_tensor == true_class).nonzero(as_tuple=True)[0]
                    if len(true_indices) > 0:
                        class_preds = val_pred[true_indices]
                        total = len(true_indices)
                        for pred_class in range(4):
                            if pred_class != true_class:
                                misclass_rate = (class_preds == pred_class).sum().item() / total
                                if misclass_rate > 0:
                                    print(
                                        f"Class {true_class} misclassified as Class {pred_class}: {misclass_rate:.2%}")

                correct = val_pred.eq(batch_y_tensor).sum().item()
                val_acc = correct / len(batch_y_tensor)
                print(val_acc)
                Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
                    val_pred, batch_y_tensor)
                temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                    Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
                temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                    Recll2), temp_val_F2.append(F2), \
                temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
                    Recll3), temp_val_F3.append(F3), \
                temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(
                    Recll4), temp_val_F4.append(F4)
                temp_val_accs.append(val_acc)
                index = index + args.batchsize
        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        '''
        if val_acc > best_acc:
            best_acc = val_acc
            # 保存最佳模型
            save_dict = {
                'epoch': epoch,
                'models_state_dict': [model.state_dict() for model in models],
                'accuracy': val_acc,
                'weights': weights
            }
            torch.save(save_dict, f'best_model_acc_{val_acc:.4f}.pt')
            print(f'保存新的最佳模型, 准确率: {val_acc:.4f}')
        '''

        logging.info((np.mean(temp_val_losses), np.mean(temp_val_accs)))
        print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses),
                                                                           np.mean(temp_val_accs)))

        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
               'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
                                                       np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
               'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
                                                       np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
        logging.info(res)
        print('results:', res)
        scheduler.step(np.mean(temp_val_losses))
        early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_F1), np.mean(temp_val_F2),
                       np.mean(temp_val_F3), np.mean(temp_val_F4), model, 'BiGCN', args.datasetname)
        accs = np.mean(temp_val_accs)
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        F3 = np.mean(temp_val_F3)
        F4 = np.mean(temp_val_F4)
        if early_stopping.early_stop:
            print("Early stopping")
            accs = early_stopping.accs
            F1 = early_stopping.F1
            F2 = early_stopping.F2
            F3 = early_stopping.F3
            F4 = early_stopping.F4
            break

    quality_threshold = {
        'loss': 2,  # loss阈值
        'confidence': 0.25,  # confidence阈值
        'acc': 1  # accuracy阈值
    }

    high_quality_students = []
    for student_id, metrics in student_quality.items():
        if (metrics['loss'] < quality_threshold['loss'] and
                metrics['confidence'] > quality_threshold['confidence'] and
                metrics['acc'] >= quality_threshold['acc']):
            high_quality_students.append((student_id, metrics))

    # 按loss排序
    high_quality_students.sort(key=lambda x: x[1]['loss'])

    # 输出高质量样本信息
    print("\n高质量样本学生信息:")
    print(student_quality)
    print([student_id for student_id, _ in high_quality_students])
    for student_id, metrics in high_quality_students:
        print(f"学号: {student_id}")
        print(f"Loss: {metrics['loss']:.4f}")
        print(f"Confidence: {metrics['confidence']:.4f}")
        print(f"Accuracy: {metrics['acc']:.4f}")
    return accs, F1, F2, F3, F4


# 在需要生成随机数据的实验中，每次实验都需要生成数据。设置随机种子是为了确保每次生成固定的随机数，这就使得每次实验结果显示一致了，有利于实验的比较和改进。使得每次运行该 .py 文件时生成的随机数相同。
def init_seeds(seed=2020):
    torch.manual_seed(seed)  # sets the seed for generating random numbers.为CPU中设置种子，生成随机数
    torch.cuda.manual_seed(  # 为特定GPU设置种子，生成随机数
        seed)  # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(  # 为所有GPU设置种子，生成随机数
        seed)  # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    # 下面两行代码的作用：保证每次运行网络的时候相同输入的输出是固定的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Init_seeds....", seed)  # Init_seeds.... 2020


if __name__ == '__main__':
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    parser = argparse.ArgumentParser()

    parser.add_argument('--datasetname', type=str, default="filtered_balanced_data_stu_fak22.xlsx", metavar='dataname',
                        help='dataset name')
    parser.add_argument('--modelname', type=str, default="EBGCN", metavar='modeltype',
                        help='model type, option: BiGCN/EBGCN')
    parser.add_argument('--input_features', type=int, default=5000, metavar='inputF',
                        help='dimension of input features (TF-IDF)')
    parser.add_argument('--hidden_features', type=int, default=64, metavar='graph_hidden',
                        help='dimension of graph hidden state')
    parser.add_argument('--output_features', type=int, default=64, metavar='output_features',
                        help='dimension of output features')
    parser.add_argument('--num_class', type=int, default=4, metavar='numclass',
                        help='number of classes')
    parser.add_argument('--num_workers', type=int, default=30, metavar='num_workers',
                        help='number of workers for training')############################

    # textcnn parameter
    parser.add_argument('--n_vocab', type=int, default=32, metavar='n_vocab',
                        help='n_vocab')
    parser.add_argument('--embed', type=int, default=128, metavar='embed',
                        help='embed')
    parser.add_argument('--num_filters', type=int, default=256, metavar='num_filters',
                        help='num_filters')
    parser.add_argument('--filter_sizes', type=list, default=[2,3,4], metavar='filter_sizes',
                        help='filter_sizes')

    # Parameters for training the model
    parser.add_argument('--seed', type=int, default=2020, help='random state seed')
    parser.add_argument('--no_cuda', action='store_true',
                        help='does not use GPU')
    parser.add_argument('--num_cuda', type=int, default=0,
                        help='index of GPU 0/1')

    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate')
    parser.add_argument('--lr_scale_bu', type=int, default=5, metavar='LRSB',
                        help='learning rate scale for bottom-up direction')
    parser.add_argument('--lr_scale_td', type=int, default=1, metavar='LRST',
                        help='learning rate scale for top-down direction')
    parser.add_argument('--l2', type=float, default=1e-4, metavar='L2',
                        help='L2 regularization weight')

    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout',
                        help='dropout rate')
    parser.add_argument('--patience', type=int, default=10, metavar='patience',
                        help='patience for early stop')
    parser.add_argument('--batchsize', type=int, default=32, metavar='BS',
                        help='batch size')
    parser.add_argument('--n_epochs', type=int, default=200, metavar='E',
                        help='number of max epochs')
    parser.add_argument('--iterations', type=int, default=15, metavar='F',
                        help='number of iterations for 5-fold cross-validation')

    # Parameters for the proposed model
    parser.add_argument('--TDdroprate', type=float, default=0.2, metavar='TDdroprate',
                        help='drop rate for edges in the top-down propagation graph')
    parser.add_argument('--BUdroprate', type=float, default=0.2, metavar='BUdroprate',
                        help='drop rate for edges in the bottom-up dispersion graph')
    parser.add_argument('--edge_infer_td', action='store_true', default=False,
                        help='edge inference in the top-down graph')
    parser.add_argument('--edge_infer_bu', action='store_true', default=False,
                        help='edge inference in the bottom-up graph')
    parser.add_argument('--edge_loss_td', type=float, default=0.2, metavar='edge_loss_td',
                        help='a hyperparameter gamma to weight the unsupervised relation learning loss in the top-down propagation graph')
    parser.add_argument('--edge_loss_bu', type=float, default=0.2, metavar='edge_loss_bu',
                        help='a hyperparameter gamma to weight the unsupervised relation learning loss in the bottom-up dispersion graph')
    parser.add_argument('--edge_num', type=int, default=2, metavar='edgenum',
                        help='latent relation types T in the edge inference')

    args = parser.parse_args()
    print(args)

    vocab_size = 4857
    embed_size = 300
    gat_hidden_dim = 16
    output_dim = 300
    alpha = 0.3

    if not args.no_cuda:
        print('Running on GPU:{}'.format(args.num_cuda))  # Running on GPU:0
        args.device = torch.device('cuda:{}'.format(args.num_cuda) if torch.cuda.is_available() else 'cpu')#############在这里加并行
    else:
        print('Running on CPU')
        args.device = torch.device('cpu')
    # print(
    #     args)  # Namespace(BUdroprate=0.2, TDdroprate=0.2, batchsize=128, datasetname='Twitter15', device=device(type='cuda', index=0), dropout=0.5, edge_infer_bu=False, edge_infer_td=False, edge_loss_bu=0.2, edge_loss_td=0.2, edge_num=2, hidden_features=64, input_features=5000, iterations=50, l2=0.0001, lr=0.0005, lr_scale_bu=5, lr_scale_td=1, modelname='BiGCN', n_epochs=200, no_cuda=False, num_class=4, num_cuda=0, num_workers=30, output_features=64, patience=10, seed=2020)

    init_seeds(seed=args.seed)
    logging.basicConfig(filename="./logs_fak22_compare_RNNLSTM.txt", level=logging.INFO)
    total_accs, total_NR_F1, total_FR_F1, total_TR_F1, total_UR_F1 = [], [], [], [], []  # 存放几个评价指标的列表

    for iter in range(args.iterations):  # args.iterations=50 迭代次数
        iter_timestamp = time()  # 显示运行到此处的时间
        # fold_tests, fold_trains = load5foldData(args.datasetname)
        fold_tests, fold_trains = load5foldData(args.datasetname,shuffle_flag=True,seed=args.seed)  # fold_tests包含5个test列表，fold_trains包含5个train列表

        accs, NR_F1, FR_F1, TR_F1, UR_F1 = [], [], [], [], []
        for fold_idx in range(5):
            fold_timestamp = time()
            acc, F1, F2, F3, F4 = train_model(fold_tests[fold_idx], fold_trains[fold_idx], args, iter)
            accs.append(acc)
            NR_F1.append(F1)
            FR_F1.append(F2)
            TR_F1.append(F3)
            UR_F1.append(F4)

            logging.info((iter, args.iterations, fold_idx, acc, F1, F2, F3, F4))
            print(
                "Iter:{}/{}\tFold:{}/5 - Acc:{:.4f}\tNR_F1:{:.4f}\tFR_F1:{:.4f}\tTR_F1:{:.4f}\tUR_F1:{:.4f}\tTime:{:.4f}s".format(
                    iter, args.iterations,
                    fold_idx,
                    acc, F1, F2, F3, F4,
                    time() - fold_timestamp))

        total_accs.append(np.mean(accs))
        total_NR_F1.append(np.mean(NR_F1))
        total_FR_F1.append(np.mean(FR_F1))
        total_TR_F1.append(np.mean(TR_F1))
        total_UR_F1.append(np.mean(UR_F1))

        print("****  Iteration Result {}/{} Time:{:.4f}s  ****".format(iter, args.iterations, time() - iter_timestamp))
        print(
            "Acc:{:.4f}\tNR_F1:{:.4f}\tFR_F1:{:.4f}\tTR_F1:{:.4f}\tUR_F1:{:.4f}\t\tavg_F1:{:.4f}".format(np.mean(accs),
                                                                                                         np.mean(NR_F1),
                                                                                                         np.mean(FR_F1),
                                                                                                         np.mean(TR_F1),
                                                                                                         np.mean(UR_F1),
                                                                                                         (np.mean(
                                                                                                             NR_F1) + np.mean(
                                                                                                             FR_F1) + np.mean(
                                                                                                             TR_F1) + np.mean(
                                                                                                             UR_F1)) / 4))
        logging.info((np.mean(accs), np.mean(NR_F1), np.mean(FR_F1), np.mean(TR_F1), np.mean(UR_F1), (np.mean(NR_F1) + np.mean(FR_F1) + np.mean(TR_F1) + np.mean(UR_F1)) / 4))
    print("****  Total Result  ****")
    print("Acc:{:.4f}\tNR_F1:{:.4f}\tFR_F1:{:.4f}\tTR_F1:{:.4f}\tUR_F1:{:.4f}".format(np.mean(total_accs),
                                                                                      np.mean(total_NR_F1),
                                                                                      np.mean(total_FR_F1),
                                                                                      np.mean(total_TR_F1),
                                                                                      np.mean(total_UR_F1)))
    logging.info((np.mean(total_accs), np.mean(total_NR_F1), np.mean(total_FR_F1), np.mean(total_TR_F1), np.mean(total_UR_F1)))
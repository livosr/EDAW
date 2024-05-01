import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import random
import scipy
import numpy as np
import math
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
from sklearn.metrics import confusion_matrix
from seqeval.metrics import f1_score # 序列标注评估工具
from transformers import AutoTokenizer

from e_cl.src.config import get_params
from e_cl.src.dataloader import *
from e_cl.src.utils import *

logger = logging.getLogger()
params = get_params()
auto_tokenizer = AutoTokenizer.from_pretrained(params.model_name)
pad_token_label_id = nn.CrossEntropyLoss().ignore_index  # -100
# from matplotlib import font_manager
# font_manager.fontManager.addfont('/usr/local/share/fonts/truetype/times_new_roman/times.ttf')
# plt.rcParams['font.family'] = 'Times New Roman'


class BaseTrainer(object):
    def __init__(self, params, model, label_list):
        self.params = params
        self.model = model 
        self.label_list = label_list
        
        # training
        self.lr = float(params.lr)
        self.mu = 0.9
        self.weight_decay = 5e-4
    

    def batch_forward(self, inputs):
        self.inputs = inputs
        self.all_features = self.model.encoder(self.inputs)
        self.logits = self.model.forward_classifier(self.all_features[1][-1])


    def batch_loss(self, labels):
        self.loss = 0
        assert self.logits!=None, "logits is none!"

        # classification loss
        ce_loss = nn.CrossEntropyLoss()(self.logits.view(-1, self.logits.shape[-1]), 
                                labels.flatten().long())
        self.loss = ce_loss
        return ce_loss.item() 



    def get_entropy_weight(self, probabilities):
        entropy_max = math.log(probabilities.shape[-1] + 1e-8)
        entropy_matrix = -probabilities * torch.log(probabilities + 1e-8)
        entropy = torch.sum(entropy_matrix, dim=-1)
        weight = 1 - entropy / entropy_max
        return weight


    def batch_loss_ede(self, labels):
     
        original_labels = labels.clone()
        self.loss = 0
        refer_dims = self.refer_model.classifier.output_dim
        all_dims = self.model.classifier.output_dim

        assert self.logits!=None, "logits is none!"
        assert self.refer_model!=None, "refer_model is none!"
        assert self.inputs!=None, "inputs is none!"
        assert self.inputs.shape[:2]==labels.shape[:2], "inputs and labels are not matched!"  

        with torch.no_grad():
            self.refer_model.eval()
            refer_all_features = self.refer_model.encoder(self.inputs)
            refer_features = refer_all_features[1][-1]
            refer_logits = self.refer_model.forward_classifier(refer_features)
            assert refer_logits.shape[:2] == self.logits.shape[:2], \
                    "the first 2 dims of refer_logits and logits are not equal!!!"
        

        mask_background = (labels < self.old_classes) & (labels != pad_token_label_id) # 0 的位置
        mask_new = labels >= self.old_classes

  
        probs = torch.softmax(refer_logits, dim=-1)
        _, pseudo_labels = probs.max(dim=-1)


        if params.use_pseudo_label:
            labels[mask_background] = pseudo_labels[mask_background]

        loss = nn.CrossEntropyLoss(reduction='none')(self.logits.permute(0,2,1), labels)

        if params.use_entropy_weight:

            entropy_weights = self.get_entropy_weight(probs)
            entropy_weights = entropy_weights ** params.weight_power
            entropy_weights = torch.where(mask_background, entropy_weights, torch.tensor(0.))
            entropy_weights[mask_new] = torch.tensor(params.new_label_weight)
            loss = entropy_weights * loss


        ignore_mask = (labels != pad_token_label_id)
            
        if torch.sum(ignore_mask.float())==0: 
            ce_loss = torch.tensor(0., requires_grad=True).to(params.device)
        else:
            ce_loss = loss[ignore_mask].mean()  # scalar
        

        
        distill_mask = torch.logical_and(original_labels==0, original_labels!=pad_token_label_id)

        if params.use_distill:
            if torch.sum(distill_mask.float())==0:
                loss_kl = torch.tensor(0., requires_grad=True).to(params.device)
                loss_sim = torch.tensor(0., requires_grad=True).to(params.device)
                distill_loss = torch.tensor(0., requires_grad=True).to(params.device)
            else:

                if params.use_decomposed:
                    student = self.logits[distill_mask][:, :refer_dims].view(-1, refer_dims)
                    teacher = refer_logits[distill_mask].view(-1, refer_dims)

                    positive_mask = student >= 0
                    negative_mask = student <= 0

                    loss_kl_positive = nn.KLDivLoss(reduction='batchmean')(
                        F.log_softmax(student[positive_mask], dim=-1), F.softmax(teacher[positive_mask], dim=-1))
                    loss_kl_negative = nn.KLDivLoss(reduction="batchmean")(
                        F.log_softmax(student[negative_mask], dim=-1), F.softmax(teacher[negative_mask], dim=-1))
                    loss_kl = loss_kl_positive + loss_kl_negative

                else:
                    old_logits_score = F.log_softmax(
                                        self.logits[distill_mask]/self.params.temperature,
                                        dim=-1)[:,:refer_dims].view(-1, refer_dims) #(bsz*seq_len(select out), refer_dims)

                    ref_old_logits_score = F.softmax(
                                        refer_logits[distill_mask]/self.params.ref_temperature,
                                        dim=-1).view(-1, refer_dims)

                    loss_kl = nn.KLDivLoss(reduction='batchmean')(old_logits_score, ref_old_logits_score)

                if params.use_sim:
                    student_logits = self.logits[distill_mask][:, :refer_dims].view(-1, refer_dims)
                    teacher_logits = refer_logits[distill_mask].view(-1, refer_dims)

                    # 对 student_logits 进行 L2 规范化
                    student_logits_normalized = F.normalize(student_logits, p=2, dim=-1)

                    # 对 teacher_logits 进行 L2 规范化
                    teacher_logits_normalized = F.normalize(teacher_logits, p=2, dim=-1)

                    if params.use_sim_norm:

                        # 计算 cos 相似度
                        loss_sim = 1 - F.cosine_similarity(student_logits_normalized, teacher_logits_normalized)
                        # loss_sim = loss_sim * params.sim_weight
                    else:
                        # 计算 cos 相似度
                        loss_sim = 1 - F.cosine_similarity(student_logits, teacher_logits)
                    #
                    # loss_sim = 1 - F.cosine_similarity(self.logits[distill_mask][:, :refer_dims].view(-1, refer_dims),
                    #                                    refer_logits[distill_mask].view(-1, refer_dims))
                    loss_sim = torch.mean(loss_sim)
                    # loss_kd = params.kl_weight * loss_kl + (1 - params.kl_weight) * loss_sim
                    if params.use_kl:
                        loss_kd = loss_kl + loss_sim * params.sim_weight
                    else:
                        loss_kl = torch.tensor(0).to(params.device)
                        loss_kd = loss_sim * params.sim_weight



                else:
                    loss_sim = torch.tensor(0).to(params.device)
                    loss_kd = loss_kl


                distill_loss = params.distill_weight*loss_kd
        else:
            loss_kl = torch.tensor(0., requires_grad=True).to(params.device)
            loss_sim = torch.tensor(0., requires_grad=True).to(params.device)
            distill_loss = torch.tensor(0., requires_grad=True).to(params.device)


        self.loss = ce_loss + distill_loss # 总loss

        return ce_loss.item(), distill_loss.item(), loss_kl.item(), loss_sim.item()

            
    def batch_backward(self):
        self.model.train()
        self.optimizer.zero_grad()        
        self.loss.backward()
        self.optimizer.step()
        
        return self.loss.item()

    def merge_subwords(self, token_list):
        # 合并子词token为完整的单词
        merged_tokens = []
        for token in token_list:
            if token.startswith("##"):
                # 如果是子词，则去掉"##"并连接到上一个token
                merged_tokens[-1] += token[2:]
            else:
                # 否则，直接添加到列表中
                merged_tokens.append(token)
        return merged_tokens

    def evaluate(self, dataloader, each_class=False, entity_order=[], is_plot_hist=False, is_plot_cm=False):
        with torch.no_grad():
            self.model.eval()

            y_list = []
            x_list = []
            logits_list = []

            bar = tqdm(dataloader)

            for x, y in bar:
                x, y = x.to(params.device), y.to(params.device)
                self.batch_forward(x)
                _logits = self.logits.view(-1, self.logits.shape[-1]).detach().cpu()
                logits_list.append(_logits)
                x = x.view(x.size(0)*x.size(1)).detach().cpu() # bs*seq_len
                x_list.append(x)
                bar.set_description("Evaluating")


                y = y.view(y.size(0)*y.size(1)).detach().cpu()
                y_list.append(y)



            
            y_list = torch.cat(y_list)
            x_list = torch.cat(x_list)
            logits_list = torch.cat(logits_list)   
            pred_list = torch.argmax(logits_list, dim=-1)


            # ### Plot the (logits) prob distribution for each class
            # if is_plot_hist: # False
            #     plot_prob_hist_each_class(deepcopy(y_list),
            #                             deepcopy(logits_list),
            #                             ignore_label_lst=[
            #                                 self.label_list.index('O'),
            #                                 pad_token_label_id
            #                             ])
            #
            #
            # ### for confusion matrix visualization
            # if is_plot_cm: # False
            #     plot_confusion_matrix(deepcopy(pred_list),
            #                     deepcopy(y_list),
            #                     label_list=self.label_list,
            #                     pad_token_label_id=pad_token_label_id)

            ### calcuate f1 score
            pred_line = []
            gold_line = []
            word_line = []
            word_li = []
            with open(file=params.exp_name + "_res.txt", mode="w", encoding="utf-8") as f:
                pass
                f.close()

            for i, (pred_index, word_index, gold_index) in enumerate(zip(pred_list, x_list, y_list)):
                gold_index = int(gold_index)
                # print(gold_index)
                if word_index in [102]:
                    with open(file=params.exp_name + "_res.txt", mode="a", encoding="utf-8") as f:
                        f.write("\n")
                if word_index in [101, 102, 100]:
                    continue


                if gold_index != pad_token_label_id: # !=-100

                    pred_token = self.label_list[pred_index] # label索引转label
                    gold_token = self.label_list[gold_index]


                    word_li.append(word_index)
                    num = i + 1
                    while num < len(pred_list) and auto_tokenizer.convert_ids_to_tokens([x_list[num]])[0].startswith("##"):
                        word_li.append(x_list[num])
                        num += 1

                    merged_tokens = self.merge_subwords(auto_tokenizer.convert_ids_to_tokens(word_li))
                    merged_string = " ".join(merged_tokens)
                    with open(file=params.exp_name + "_res.txt", mode="a", encoding="utf-8") as f:
                        f.write("{}\t{}\t{}\n".format(merged_string, pred_token, gold_token))
                    word_li = []
                    # lines.append("w" + " " + pred_token + " " + gold_token)
                    word_line.append(word_index)
                    pred_line.append(pred_token) 
                    gold_line.append(gold_token) 


            # Check whether the label set are the same,
            # ensure that the predict label set is the subset of the gold label set
            gold_label_set, pred_label_set = np.unique(gold_line), np.unique(pred_line)
            if set(gold_label_set)!=set(pred_label_set):
                O_label_set = []
                for e in pred_label_set:
                    if e not in gold_label_set:
                        O_label_set.append(e)
                if len(O_label_set)>0:
                    # map the predicted labels which are not seen in gold label set to 'O'
                    for i, pred in enumerate(pred_line):
                        if pred in O_label_set:
                            pred_line[i] = 'O'

            self.model.train()

            # compute overall f1 score
            # micro f1 (default)




            f1 = f1_score([gold_line], [pred_line])*100
            # macro f1 (average of each class f1)
            ma_f1 = f1_score([gold_line], [pred_line], average='macro')*100
            if not each_class: # 不打印每个类别的f1
                return f1, ma_f1

            # compute f1 score for each class
            f1_list = f1_score([gold_line], [pred_line], average=None)
            f1_list = list(np.array(f1_list)*100)
            gold_entity_set = set()
            for l in gold_label_set:
                if 'B-' in l or 'I-' in l or 'E-' in l or 'S-' in l:
                    gold_entity_set.add(l[2:])
            gold_entity_list = sorted(list(gold_entity_set))
            f1_score_dict = dict()
            for e, s in zip(gold_entity_list,f1_list):
                f1_score_dict[e] = round(s,2)
            # using the default order for f1_score_dict
            if entity_order==[]:
                return f1, ma_f1, f1_score_dict
            # using the pre-defined order for f1_score_dict
            assert set(entity_order)==set(gold_entity_list),\
                "gold_entity_list and entity_order has different entity set!"
            ordered_f1_score_dict = dict()
            for e in entity_order:
                ordered_f1_score_dict[e] = f1_score_dict[e]
            return f1, ma_f1, ordered_f1_score_dict



    def save_model(self, save_model_name, path=''):
        """
        save the best model
        """
        if len(path)>0:
            saved_path = os.path.join(path, str(save_model_name))
        else:
            saved_path = os.path.join(self.params.dump_path, str(save_model_name))
        torch.save({
            "hidden_dim": self.model.hidden_dim,
            "output_dim": self.model.output_dim,
            "encoder": self.model.encoder.state_dict(),
            "classifier": self.model.classifier
        }, saved_path)
        logger.info("Best model has been saved to %s" % saved_path)

    def load_model(self, load_model_name, path=''):
        """
        load the checkpoint
        """
        if len(path)>0:
            load_path = os.path.join(path, str(load_model_name))
        else:
            load_path = os.path.join(self.params.dump_path, str(load_model_name))
        print(load_path)
        ckpt = torch.load(load_path)

        self.model.hidden_dim = ckpt['hidden_dim']
        self.model.output_dim = ckpt['output_dim']
        self.model.encoder.load_state_dict(ckpt['encoder'])
        self.model.classifier = ckpt['classifier']
        logger.info("Model has been load from %s" % load_path)

    def entropy_plot(self, dataloader):
        with torch.no_grad():
            self.model.eval()

            y_list = []
            x_list = []
            logits_list = []
            entropy_list = []
            weight_list = []
            preds_list = []


            for x, y in dataloader:
                x, y = x.to(params.device) , y.to(params.device)
                self.batch_forward(x)
                _logits = self.logits.view(-1, self.logits.shape[-1]).detach().cpu()
                probabilities = F.softmax(_logits, dim=-1)
                preds_list.append(torch.argmax(probabilities, dim=1))
                entropy_weights = self.get_entropy_weight(probabilities)
                entropy_weights = entropy_weights ** 3
                entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=-1)


                entropy_list.append(entropy)
                weight_list.append(entropy_weights)
                x = x.view(x.size(0)*x.size(1)).detach().cpu() # bs*seq_len
                x_list.append(x)
                y = y.view(y.size(0)*y.size(1)).detach().cpu()
                y_list.append(y)

            # 将所有batch的数据合并
            all_y = torch.cat(y_list)
            all_entropy = torch.cat(entropy_list)
            all_weights = torch.cat(weight_list)
            all_preds = torch.cat(preds_list)

            all_acc = []
            all_wei = []

            for pred, y, ent, weight in zip(all_preds, all_y, all_entropy, all_weights):
                y = int(y)
                if y != pad_token_label_id: # !=-100
                    if pred == y:
                        all_acc.append((ent, 1))
                        all_wei.append((weight, 1))
                    else:
                        all_acc.append((ent, 0))
                        all_wei.append((weight, 0))

        correct_samples = [item for item in all_acc if item[1] == 1]
        incorrect_samples = [item for item, y in zip(all_acc, all_y) if item[1] == 0 and int(y) != 0]

        correct_weights = [item for item in all_wei if item[1] == 1]
        incorrect_weights = [item for item, y in zip(all_wei, all_y) if item[1] == 0 and int(y) != 0]

        # 从每个子集中随机选择500个样本
        sampled_correct = random.sample(correct_samples, min(500, len(correct_samples)))
        sampled_incorrect = random.sample(incorrect_samples, min(500, len(incorrect_samples)))
        # sampled_incorrect = np.array(incorrect_samples).sort(key=lambda x: x[0], reverse=True)[:500]

        sampled_correct_weight = random.sample(correct_weights, min(500, len(correct_weights)))
        # sampled_incorrect = random.sample(incorrect_samples, min(500, len(incorrect_samples)))
        sampled_incorrect_weight = incorrect_weights


        # 合并采样的正确和错误样本，并随机打乱顺序
        combined_samples = sampled_correct + sampled_incorrect

        combined_weights = sampled_correct_weight + sampled_incorrect_weight

        random.shuffle(combined_samples)

        random.shuffle(combined_weights)

        # 分别为横轴和纵轴初始化列表，以及颜色
        x_axis = []
        entropies = []
        colors = []

        # 添加数据到列表中
        for idx, (ent, is_correct) in enumerate(combined_samples):
            x_axis.append(idx)
            entropies.append(ent)
            colors.append('blue' if is_correct else 'red')

        print("total = ", len(x_axis))
        print("ent = ", len(entropies))

        # 绘制散点图，设置点的大小为 smaller size e.g. s=10
        plt.scatter(x_axis, entropies, c=colors, s=3)  # s 控制点的大小
        plt.subplots_adjust(left=0.15, right=0.85, top=0.98, bottom=0.35)
        plt.xlabel('Token index', fontdict={"family":'Times New Roman',"fontsize":15})
        plt.ylabel('Entropy', fontdict={"family":'Times New Roman',"fontsize":15})
        # plt.xticks([])  # 隐藏 x 轴刻度
        # plt.yticks([])  # 隐藏 y 轴刻度
        # plt.title('Entropy of Predictions')
        # plt.legend(loc='upper right', fontsize=30)
        plt.xticks([0, 500, 1000], fontsize=10)
        plt.yticks(fontsize=10)
        plt.savefig(params.exp_name + '_entropy.pdf', dpi=400, format='pdf')
        plt.clf()



        x_axis = []
        weights = []
        colors = []

        for idx, (weight, is_correct) in enumerate(combined_weights):
            x_axis.append(idx)
            weights.append(weight)
            colors.append('blue' if is_correct else 'red')

        plt.scatter(x_axis, weights, c=colors, s=2)  # s 控制点的大小
        plt.xlabel('Token Index')
        plt.ylabel('Entropy Weight')
        plt.title('Entropy Weight of Predictions')
        plt.savefig(params.exp_name + '_entropy_weight.png')
        plt.clf()

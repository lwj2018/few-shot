import torch
import numpy as np

class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label_set = list(set(label))
        self.label_set = label_set
        label = np.array(label)
        self.m_ind = {}
        for i in label_set:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind[i] = ind

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = np.random.permutation(self.label_set)[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


class CategoriesSampler_train():
    # Normal sampler, suit for omniglot
    def __init__(self, label, n_batch, n_cls, n_shot,n_query, n_base_class):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_base_class = n_base_class

        label_set = list(set(label))
        self.label_set = label_set
        label = np.array(label)
        self.m_ind = {}
        for i in label_set:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind[i] = ind

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            query_batch = []
            shot_batch = []
            classes = np.random.permutation(self.label_set)[:self.n_cls]
            classes,order =classes.sort()
            for c in classes:
                if c < self.n_base_class:
                    l = self.m_ind[c]
                    tmp = torch.randperm(len(l))
                    batch.append(l[tmp[:self.n_shot+self.n_query]])

                else:
                    # 如果c属于c_novel，只取前n_shot个样本用于训练，然后做数据增强
                    l = self.m_ind[c]
                    if self.n_shot>1:
                        tmp = torch.randperm(self.n_shot)
                        novel_query = torch.randperm(self.n_shot-1)[0]+1
                        a = tmp[:self.n_shot-novel_query]
                        b = tmp[self.n_shot-novel_query:]
                        batch.append(torch.cat((l[a.repeat(15)[:self.n_shot]],l[b.repeat(15)[:self.n_query]])))
                    else:
                        ind = l[0].view(-1)
                        batch.append(torch.cat([ind,ind]))

            # shape of batch is (n_shot+n_query) x train_way
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


class CategoriesSampler_train_100way():

    def __init__(self, label, n_batch, n_cls, n_shot,n_query, n_base_class):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_base_class = n_base_class

        label_set = list(set(label))
        self.label_set = label_set
        label = np.array(label)
        self.m_ind = {}
        for i in label_set:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind[i] = ind

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            query_batch = []
            shot_batch = []
            # 随机选出n_cls个类
            classes = np.random.permutation(self.label_set)[:self.n_cls]
            # 排序以保证基类在前，新类在后
            classes,order =classes.sort()

            for c in classes:
                if c < self.n_base_class:
                    # 如果c属于c_base，在前500个样本中随机采样n_shot+n_query个样本
                    l = self.m_ind[c]
                    tmp = torch.randperm(len(l)-100)
                    batch.append(l[tmp[:self.n_shot+self.n_query]])

                else:
                    # 如果c属于c_novel，只取前n_shot个样本用于训练，然后做数据增强
                    l = self.m_ind[c]
                    if self.n_shot>1:
                        tmp = torch.randperm(self.n_shot)
                        novel_query = torch.randperm(self.n_shot-1)[0]+1
                        a = tmp[:self.n_shot-novel_query]
                        b = tmp[self.n_shot-novel_query:]
                        batch.append(torch.cat((l[a.repeat(15)[:self.n_shot]],l[b.repeat(15)[:self.n_query]])))
                    else:
                        ind = l[0].view(-1)
                        batch.append(torch.cat([ind,ind]))

            batch = torch.stack(batch).t().reshape(-1)
            yield batch

class CategoriesSampler_train_mn():
    # Sample data in MN format
    def __init__(self, label, n_batch, n_cls, n_shot,n_query, n_base_class):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_base_class = n_base_class

        label_set = list(set(label))
        self.label_set = label_set
        label = np.array(label)
        self.m_ind = {}
        for i in label_set:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind[i] = ind

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            query_batch = []
            shot_batch = []
            # 随机选出n_cls个类
            classes = np.random.permutation(self.label_set)[:self.n_cls]
            # 排序以保证基类在前，新类在后
            classes,order =classes.sort()

            for c in classes:
                if c < self.n_base_class:
                    l = self.m_ind[c]
                    tmp = torch.randperm(len(l))
                    batch.append(l[tmp[:self.n_shot+self.n_query]])
                else:
                    # 如果c属于c_novel，只取前n_shot个样本用于训练，然后做数据增强
                    l = self.m_ind[c]
                    if self.n_shot>1:
                        tmp = torch.randperm(self.n_shot)
                        novel_query = torch.randperm(self.n_shot-1)[0]+1
                        a = tmp[:self.n_shot-novel_query]
                        b = tmp[self.n_shot-novel_query:]
                        batch.append(torch.cat((l[a.repeat(15)[:self.n_shot]],l[b.repeat(15)[:self.n_query]])))
                    else:
                        ind = l[0].view(-1)
                        batch.append(torch.cat([ind,ind]))

            batch = torch.stack(batch)
            data_shot = batch[:,:self.n_shot]
            data_query = batch[:,self.n_shot:]
            batch = torch.cat([data_shot.reshape(-1),data_query.reshape(-1)])

            yield batch

class CategoriesSampler_val_100way():
    # Sample data in MN format
    def __init__(self, label, n_batch, n_cls, n_shot,n_query):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_shot = n_shot
        self.n_query = n_query

        label_set = list(set(label))
        self.label_set = label_set
        label = np.array(label)
        self.m_ind = {}
        for i in label_set:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind[i] = ind

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            # 随机选出n_cls个类
            classes = np.random.permutation(self.label_set)[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                #pos = torch.cat([torch.Tensor(range(0,self.n_shot)).type(torch.LongTensor), self.n_shot+torch.randperm(len(l)-self.n_shot)[:self.n_query]])
                # 利用训练时未使用过的最后100个样本
                # shot+query_val = 5+15 = 20
                tmp = torch.randperm(100)[:self.n_shot+self.n_query]+500
                batch.append(l[tmp])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

class CategoriesSampler_val_mn():

    def __init__(self, label, n_batch, n_cls, n_shot,n_query):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_shot = n_shot
        self.n_query = n_query

        label_set = list(set(label))
        self.label_set = label_set
        label = np.array(label)
        self.m_ind = {}
        for i in label_set:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind[i] = ind

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            # 随机选出n_cls个类
            classes = np.random.permutation(self.label_set)[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                # 前n_shot个样本作为支撑集，之后的样本中取n_query个作为query set
                pos = torch.cat([torch.Tensor(range(0,self.n_shot)).type(torch.LongTensor),self.n_shot+torch.randperm(len(l)-self.n_shot)[:self.n_query]])
                batch.append(l[pos])
            # Rearrange the data as the input format of MN
            batch = torch.stack(batch)
            data_shot = batch[:,:self.n_shot]
            data_query = batch[:,self.n_shot:]
            batch = torch.cat([data_shot.reshape(-1),data_query.reshape(-1)])
            yield batch

class CategoriesSampler_val():
    # Normal sampler, suit for omniglot
    def __init__(self, label, n_batch, n_cls, n_shot,n_query):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_shot = n_shot
        self.n_query = n_query

        label_set = list(set(label))
        self.label_set = label_set
        label = np.array(label)
        self.m_ind = {}
        for i in label_set:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind[i] = ind

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            # 随机选出n_cls个类
            classes = np.random.permutation(self.label_set)[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                # 前n_shot个样本作为支撑集，之后的样本中取n_query个作为query set
                pos = torch.cat([torch.Tensor(range(0,self.n_shot)).type(torch.LongTensor),self.n_shot+torch.randperm(len(l)-self.n_shot)[:self.n_query]])
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

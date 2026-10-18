import torch
import torch.nn as nn
import math

class CORAL(nn.Module):
    def __init__(self):
        super(CORAL, self).__init__()

    def metric_cal(self, x):
        mean_x = x.mean(0, keepdim=True)
        cent_x = x - mean_x
        cova_x = torch.mm(cent_x.transpose(0, 1), cent_x) / (x.shape[0] - 1)
        return mean_x[0], cova_x

    def metric_diff(self, means_pairs, covas_pairs):
        mean_diff = (means_pairs[:, 0, :] - means_pairs[:, 1, :]).pow(2).mean(1).sum()
        cova_diff = (covas_pairs[:, 0, :, :] - covas_pairs[:, 1, :, :]).pow(2).mean(1).mean(1).sum()
        # print(mean_diff, cova_diff)
        return mean_diff + cova_diff

    def forward(self, features, domains):
        bz = features.shape[0]
        dimension = features.shape[2]

        means = []
        covas = []
        relations = []
        for i in range(4):
            index = torch.where(domains == i)[0]
            if index.shape[0] != 0:
                features_from_domain = features[index]
                mean_features, cova_features = self.metric_cal(features_from_domain.view(-1, dimension))
                means.append(mean_features)
                covas.append(cova_features)
                mean_relation = self.relation_cal(features_from_domain)
                relations.append(mean_relation)

        means_features = torch.stack(means, dim=0)
        covas_features = torch.stack(covas, dim=0)
        relations_features = torch.stack(relations, dim=0)

        domain_num = relations_features.shape[0]
        domain_ids = torch.arange(0, domain_num, device=features.device)
        domain_pair = torch.combinations(domain_ids)

        means_pairs = means_features[domain_pair]
        covas_pairs = covas_features[domain_pair]

        relations_pairs = relations_features[domain_pair]

        loss1 = self.metric_diff(means_pairs, covas_pairs)
        loss2 = self.relation_diff(relations_pairs)
        # print(loss1, loss2)
        return loss1 + loss2

    def relation_cal(self, x):
        dimension = x.shape[2]
        mean_feature = x.mean(2, keepdim=True)
        cent_feature = x - mean_feature
        var = torch.norm(cent_feature, p=2, dim=2).unsqueeze(2)
        relation = torch.bmm(cent_feature, cent_feature.transpose(1, 2)) / (torch.bmm(var, var.transpose(1, 2)))
        mean_relation = relation.mean(0)
        return mean_relation

    def relation_diff(self, relations_pairs):
        relation_diff = (relations_pairs[:, 0, :, :] - relations_pairs[:, 1, :, :]).pow(2).mean(0).sum()
        return relation_diff


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    torch.cuda.set_device(6)
    features = torch.randn((8, 20, 512)).cuda()
    domains = torch.tensor([0, 1, 0, 1, 2, 1, 3, 2]).cuda()

    coral_loss = CORAL()
    loss = coral_loss(features, domains)
    print(loss)
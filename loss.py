import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np


class MeCoQLoss:
    def __init__(self, T=1.0,
                 mode='simple',
                 pos_prior=0,
                 hp_lambda=0,  # entropy regularization hyper-param
                 hp_gamma=0,  # codeword regularization hyper-param
                 device='cpu',
                 ):
        self.device = device
        self.temperature = T
        self.mode = mode
        self.pos_prior = pos_prior

        self.hp_lambda = hp_lambda
        self.hp_gamma = hp_gamma

    def __call__(self, view1_feats, view2_feats, queue_feats,
                     view1_soft_codes, view2_soft_codes, codebooks, student_features, teacher_features, soft_features, hard_features,topk, beta, gamma,
                     global_step=0, writer=None):

        # compute contrastive loss
        loss = self._simclr_loss(view1_feats, view2_feats, queue_feats=queue_feats)
        if writer is not None:
            writer.add_scalar('loss/simclr_loss', loss.item(), global_step)

        if student_features is not None and teacher_features is not None:
            batch_size = student_features.shape[0]

           # Get sorted teacher similarity and corresponding cross similarity
            teacher_topk_values, sorted_indices = torch.sort(teacher_features, dim=1, descending=True)
            student_topk_values = torch.gather(student_features, 1, sorted_indices)

            # Extract top-k similarities (excluding the highest)
            student_distances = student_topk_values[:, 1:topk]
            teacher_distances = teacher_topk_values[:, 1:topk]


            # Compute pairwise difference matrices
            student_diff_matrix = student_distances.unsqueeze(1) - student_distances.unsqueeze(2)
            teacher_diff_matrix = teacher_distances.unsqueeze(1) - teacher_distances.unsqueeze(2)

            # Flatten the difference matrices
            student_diff_flat = student_diff_matrix.view(batch_size, -1)
            teacher_diff_flat = teacher_diff_matrix.view(batch_size, -1)

            # Avoid division by zero
            teacher_diff_flat[teacher_diff_flat == 0] = 1


            # Compute hard and simple weights
            hard_weights = (student_diff_flat / teacher_diff_flat).detach()
            simple_weights = (student_diff_flat / teacher_diff_flat).detach()

            hard_weights[hard_weights >= 0] = 0
            hard_weights[hard_weights < 0] = 1

            simple_weights[simple_weights <= 0] = 0
            simple_weights[simple_weights > 0] = 1



            # Avoid division by zero in student differences
            student_diff_flat[student_diff_flat == 0] = 1

            # Compute weighted result matrices
            hard_loss_matrix = hard_weights * ((student_diff_flat - teacher_diff_flat) / (0.1 + teacher_diff_flat.abs()))
            simple_loss_matrix = simple_weights * ((student_diff_flat - teacher_diff_flat) / (0.1 + teacher_diff_flat.abs()))

            # Relation KD loss (beta and gama terms)
            hard_rd_loss = beta * torch.mean(torch.norm(hard_loss_matrix, p=2, dim=1)) / (topk - 1)

            simple_rd_loss = gamma * torch.mean(torch.norm(simple_loss_matrix, p=2, dim=1)) / (topk - 1)


            loss += hard_rd_loss + simple_rd_loss


            if writer is not None:
                writer.add_scalar('loss/hard_rd_loss', hard_rd_loss.item(), global_step)
                writer.add_scalar('loss/simple_rd_loss', simple_rd_loss.item(), global_step)

        if hard_features is not None and soft_features is not None:
            soft_hard_mse = F.mse_loss(hard_features, soft_features)
            soft_hard_mse_all = 1 * soft_hard_mse
            loss += soft_hard_mse_all
            if writer is not None:
                writer.add_scalar('loss/soft_hard_mse_all', soft_hard_mse_all.item(), global_step)


        if self.hp_lambda != 0:
            # compute entropy regularization loss
            entropy_reg_loss = None
            if view1_soft_codes is not None:
                entropy_reg_loss = self._entropy_regularization(view1_soft_codes)
            if view2_soft_codes is not None:
                if entropy_reg_loss is None:
                    entropy_reg_loss = self._entropy_regularization(view2_soft_codes)
                else:
                    entropy_reg_loss = (entropy_reg_loss + \
                                        self._entropy_regularization(view2_soft_codes)) / 2
            if entropy_reg_loss is not None:
                loss += (self.hp_lambda * entropy_reg_loss)
                if writer is not None:
                    writer.add_scalar('loss/entropy_reg_loss',
                                      entropy_reg_loss.item(), global_step)

        if self.hp_gamma != 0:
            # compute codeword regularization loss
            codeword_reg_loss = self._codeword_regularization(codebooks)
            loss += (self.hp_gamma * codeword_reg_loss)
            if writer is not None:
                writer.add_scalar('loss/codeword_reg_loss',
                                  codeword_reg_loss.item(), global_step)

        if writer is not None:
            writer.add_scalar('loss/total_loss', loss.item(), global_step)
        return loss

    def _simclr_loss(self, view1_feats, view2_feats, queue_feats=None):
        cur_batch_size = view1_feats.shape[0]
        features = torch.cat([view1_feats, view2_feats], dim=0)
        features = F.normalize(features, dim=-1)
        similarity_matrix = torch.matmul(features, features.T)
        labels = torch.eye(cur_batch_size).repeat(2, 2).to(self.device)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        labels = labels[~mask].view(labels.shape[0], -1)

        pos_logits = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        neg_logits = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        if queue_feats is not None:
            queue_logits = torch.matmul(features, queue_feats.T)
            neg_logits = torch.cat([neg_logits, queue_logits], dim=-1)
        pos_logits /= self.temperature
        neg_logits /= self.temperature
        pos_probs = pos_logits.exp()
        neg_probs = neg_logits.exp()

        if self.mode == 'debias':
            N = cur_batch_size * 2 - 2
            Ng = torch.clamp((-self.pos_prior * N * pos_probs + neg_probs.sum(dim=-1)) / (1 - self.pos_prior),
                             min=math.exp(N * (-1 / self.temperature)))
        else:  # 'simple'
            Ng = neg_probs.sum(dim=-1)

        loss = (- torch.log(pos_probs / (pos_probs + Ng))).mean()
        return loss


    def _entropy_regularization(self, soft_codes):
        return (- soft_codes * soft_codes.log()).sum(dim=-1).mean()

    def _codeword_regularization(self, codebooks):
        return torch.einsum('mkd,mjd->mkj', codebooks, codebooks).mean()


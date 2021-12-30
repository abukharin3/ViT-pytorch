import os
import sys 
import argparse
import logging
import random
import torch
import numpy as np


class Pruner(object):
    def __init__(self, model, args, total_step, tb_writer=None, \
                mask_param_name=['attention.self', 'attention.output.dense',\
                'output.dense', 'intermediate.dense'], non_mask_name = ["embedding", "norm"], \
                use_no_mask=True):
        self.model = model
        self.config = vars(args)
        self.args = args
        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.mask_param_name = mask_param_name 
        self.non_mask_name = non_mask_name 
        self.use_no_mask = use_no_mask
        self.total_step = total_step
        self.tb_writer = tb_writer


    def whether_mask_para(self, n):
        if not self.use_no_mask:
            return any(nd in n for nd in self.mask_param_name)
        else:
            # print("Using no mask name")
            return not any([nd in n for nd in self.non_mask_name])


    def schedule_threshold_comb(self, step: int):
        args = self.args
        total_step = self.total_step
        initial_threshold, final_threshold = self.config['initial_threshold'], self.config['final_threshold']
        initial_warmup, final_warmup = self.config['initial_warmup'], self.config['final_warmup']
        warmup_steps = self.config['warmup_steps']
        mask_ind = False
        if step <= initial_warmup * warmup_steps:
            threshold = initial_threshold
            mask_ind = False
        elif step > (total_step - final_warmup * warmup_steps):
            threshold = final_threshold
            mask_ind = True
        elif self.config['prune_schedule'] == 'cubic':
            spars_warmup_steps = initial_warmup * warmup_steps
            spars_schedu_steps = (final_warmup + initial_warmup) * warmup_steps
            mul_coeff = 1 - (step - spars_warmup_steps) / (total_step - spars_schedu_steps)
            threshold = final_threshold + (initial_threshold - final_threshold) * (mul_coeff ** 3)
            mask_ind = True if step % self.config['deltaT'] == 0 else False
        else:
            raise ValueError("Incorrect prune schedule option.")
        return threshold, mask_ind


    def update_ipt_with_local_window(self, model, global_step):
        for n,p in model.named_parameters():
            # if any(nd in n for nd in self.mask_param_name):
            if self.whether_mask_para(n):
                if n not in self.exp_avg_ipt:
                    self.exp_avg_ipt[n] = torch.zeros_like(p)
                    self.ipt[n] = torch.zeros_like(p)
                    if self.config['beta_meta']>0 and self.config['beta_meta']!=1:
                        self.exp_avg_unc[n] = torch.zeros_like(p)
                local_step = global_step % self.config['deltaT']
                update_step = global_step // self.config['deltaT']
                if local_step == 0: 
                    self.exp_avg_ipt[n] = self.config["beta3"] * self.exp_avg_ipt[n] + (1 - self.config["beta3"]) * self.ipt[n]
                    if self.config['beta_meta'] > 0 and self.config['beta_meta'] < 1:
                        self.exp_avg_unc[n] = self.config['beta_meta'] * self.exp_avg_unc[n] + (1 - self.config['beta_meta']) * (self.ipt[n]-self.exp_avg_ipt[n]).abs()
                    elif self.config['beta_meta'] == 2.:
                        self.exp_avg_unc[n] = (update_step * self.exp_avg_unc[n] + (self.ipt[n]-self.exp_avg_ipt[n])**2 )/(update_step+1)
                    self.ipt[n] = (p * p.grad).abs().detach()
                else:
                    self.ipt[n] = (self.ipt[n] * local_step + (p * p.grad).abs().detach())/(local_step+1)


    def mask_with_threshold(self, model, threshold):
        is_dict = {}
        for n,p in model.named_parameters():
            # if any(nd in n for nd in self.mask_param_name):
            if self.whether_mask_para(n):
                if self.config['beta_meta'] > 0 and self.config['beta_meta']<1:
                    is_dict[n] = self.exp_avg_ipt[n] * self.exp_avg_unc[n] 
                elif self.config['beta_meta'] == 1.:
                    is_dict[n] = self.exp_avg_ipt[n]
                elif self.config['beta_meta'] == 2.:
                    is_dict[n] = self.exp_avg_ipt[n] * self.exp_avg_unc.sqrt()
                else:
                    is_dict[n] = self.exp_avg_ipt[n] * (self.ipt[n] - self.exp_avg_ipt[n]).abs()
        all_is = torch.cat([is_dict[n].view(-1) for n in is_dict])
        mask_threshold = torch.kthvalue(all_is, int(all_is.shape[0]*(1 - threshold)))[0].item()
        # return is_dict, mask_threshold
        for n,p in model.named_parameters():
            # if any(nd in n for nd in self.mask_param_name):
            if self.whether_mask_para(n):
                p.data.masked_fill_(is_dict[n] < mask_threshold, 0.0)
        return mask_threshold


    def mask(self, model, is_dict, mask_threshold):
        for n,p in model.named_parameters():
            # if any(nd in n for nd in self.mask_param_name):
            if self.whether_mask_para(n):
                p.data.masked_fill_(is_dict[n] < mask_threshold, 0.0)


    def update_and_pruning(self, model, global_step):
        # update importance score after optimizer stepping
        self.update_ipt_with_local_window(model, global_step)
        threshold, mask_ind = self.schedule_threshold_comb(global_step)
        if mask_ind:
            mask_threshold = self.mask_with_threshold(model, threshold)
        else:
            mask_threshold = None
        return threshold, mask_threshold



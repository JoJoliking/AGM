# Loss functions

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from utils.general import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class mot_loss(torch.nn.Module): ##尝试添加成类的形式
    def __init__(self, device, model, embedding=128, nid=14455, train_id=True):
        super(mot_loss, self).__init__()
        self.h = model.hyp
        self.device = device
        self.nid = nid
        self.embedding = embedding
        self.linear = nn.Linear(self.embedding, self.nid).to(self.device)
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([self.h['cls_pw']])).to(self.device)
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([self.h['obj_pw']])).to(self.device)
        self.IDLOSS = nn.CrossEntropyLoss(ignore_index=-1).to(self.device)
        self.emb_scale = math.sqrt(2) * math.log(self.nid - 1) if self.nid > 1 else 1
        # self.s_c = nn.Parameter(-4.15*torch.ones(1)).to(self.device)  # -4.15
        # self.s_r = nn.Parameter(-4.85*torch.ones(1)).to(self.device)  # -4.85
        self.s_id = nn.Parameter(0.05*torch.ones(1)).to(self.device)  # 0.05
        self.train_id = train_id
        g = self.h['fl_gamma']
        if g > 0:
            self.BCEcls, self.BCEobj = FocalLoss(self.BCEcls, g), FocalLoss(self.BCEobj, g)
            print('Using Focal Loss')
        print('Success loading mot loss items', self.linear)

    def forward(self, pre, targets, model):
        # print('Begin to computer loss')
        cp, cn = smooth_BCE(eps=0.0)
        device = targets.device
        # h = model.hyp
        lcls, lbox, lobj, lids = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors, id_dex, tid = build_targets(pre, targets, model)
        # g = h['fl_gamma']
        # if g > 0:
        #     self.BCEcls, self.BCEobj = FocalLoss(self.BCEcls, g), FocalLoss(self.BCEobj, g)
            # print('Using Focal Loss')
        nt = 0
        no = len(pre)
        balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
        for i, pi in enumerate(pre):
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            # bb, aa, ggj, ggi = id_dex[i]
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                nt += n  # cumulative targets
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
                # embedding = pi[bb, aa, ggj, ggi] # embedding subsets
                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                if model.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:6], cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = cp
                    lcls += self.BCEcls(ps[:, 5:6], t)  # BCE
                ##################  embedding ##################
                if self.train_id :
                    if len(tcls[i]) > 0:
#####################################   test  ####################################################################
                        # tcls_new = tcls.copy()
                        # _,new =tcls[i].sort()
                        # tcls_new[i],ps_new = tcls[i][new].clone(),ps[new].clone()
                        # test,test1,test2=tcls_new[i][new].unique(sorted=True,return_counts=True,return_inverse=True)
                        # new_1 = [test[x] if x==0 else test2[:x].sum() for x in range(len(test))]
                        # new_1=torch.LongTensor(new_1)
                        # new_1 = new_1.cuda()
                        # # print('dang qian di {%d} '%i)
                        # embedding = ps_new[:, 6:][new_1]
                        # tcls_new_all = tcls[i][new_1]
                        # embedding =embedding.cuda().contiguous()
                        # tcls_new_all = tcls_new_all.cuda()
                        # id_output = self.emb_scale * F.normalize(embedding).contiguous()
                        # id_output = self.linear(id_output)
                        # print('id_output',id_output.shape,tcls_new_all.shape)
                        # lids += self.IDLOSS(id_output, tcls_new_all)
                        # del new_1,tcls_new,embedding,id_output,ps_new,test,test1,test2
###################################  old  ##############################################
                        embedding = ps[:, 6:].contiguous()
                        id_output = self.emb_scale * F.normalize(embedding)
                        id_output = self.linear(id_output).contiguous()
                        lids += self.IDLOSS(id_output, tcls[i])
#################################################################################
                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
                ##################  embedding ##################

            lobj += self.BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss


        s = 3 / no  # output count scaling
        lbox *= self.h['box'] * s
        lobj *= self.h['obj'] * s * (1.4 if no == 4 else 1.)
        lcls *= self.h['cls'] * s
        lids *= self.s_id * s
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lids
        # 'Auto Balance'
        # loss = torch.exp(-self.s_r) * lbox + torch.exp(-self.s_c) * lobj + torch.exp(-self.s_id) * lids + \
        #        (self.s_r + self.s_c + self.s_id)
        # loss *= 0.5

        return loss * bs, torch.cat((lbox, lobj, lids, loss)).detach()


# should be modify
def compute_loss(p, targets, model):  # predictions, targets, model
    device = targets.device
    lcls, lbox, lobj,lids = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    tcls, tbox, indices, anchors = build_targets(p, targets, model)  # targets
    h = model.hyp  # hyperparameters

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['cls_pw']])).to(device)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['obj_pw']])).to(device)
    IDLOSS = nn.CrossEntropyLoss()
    # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0)

    # Focal loss
    g = h['fl_gamma']  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    # Losses
    nt = 0  # number of targets
    no = len(p)  # number of outputs
    balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

        n = b.shape[0]  # number of targets
        if n:
            nt += n  # cumulative targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # Regression
            pxy = ps[:, :2].sigmoid() * 2. - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
            pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
            iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
            lbox += (1.0 - iou).mean()  # iou loss

            # Objectness
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

            # Classification
            if model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:85], cn, device=device)  # targets
                t[range(n), tcls[i]] = cp
                lcls += BCEcls(ps[:, 5:85], t)  # BCE
            ###embeind####
            embeding = ps[:, 85:].contiguous()
            id_output = math.sqrt(2) * math.log(519-1) * F.normalize(embeding)
            lids += IDLOSS(id_output, tcls[i])
            ####end######
            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

    s = 3 / no  # output count scaling
    lbox *= h['box'] * s
    lobj *= h['obj'] * s * (1.4 if no == 4 else 1.)
    lcls *= h['cls'] * s
    # !!!
    lids *= h['box'] * s
    bs = tobj.shape[0]  # batch size

    # loss = lbox + lobj + lcls + lids
    loss = lbox + lobj + lcls
    # lobx: boxes; lobj: boxes中有物体的概率; lcls: 分类概率
    return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()


def build_targets(p, targets, model):
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
    na, nt = det.na, targets.shape[0]  # number of anchors, targets
    tcls, tbox, indices, anch = [], [], [], []
    id_index, tid = [], []
    gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
    # anchor=3个数，将target变成3xtarget格式，，方便后面算Loss
    # anchor索引，后面有用，用于表示当前bbox和当前层的哪个anchor匹配
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    # 先repeat和当前层anchor个数一样,相当于每个bbox变成了三个，然后和3个anchor单独匹配
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

    g = 0.5  # bias
    # 附近的4个网格
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], device=targets.device).float() * g  # offsets
    # 遍历三个输出分支
    for i in range(det.nl):
        anchors = det.anchors[i]
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
        # targets的xywh本身是归一化尺度，故需要变成特征图尺度
        # Match targets to anchors
        # t : (images, class, x, y, w, anchors)
        t = targets * gain
        if nt:
            ######################### id index & target #########################
            # batch, id = t[i][:, :2].long().T  # image, class
            # ggxy = t[i][:, 2:4]  # grid xy
            # # gwh = t[0][:, 4:6]  # grid wh
            # ggij = ggxy.long()
            # ggi, ggj = ggij.T  # grid xy indices
            # an_index = t[i][:, 6].long()  # anchor indices
            # tid.append(id)
            # id_index.append((batch, an_index, ggj, ggi))
            ######################### id index & target #########################

            # Matches
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
            j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            t = t[j]  # filter

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            # 5是因为预设的off是5个，现在选择出最近的3个(包括0,0也就是自己)
            t = t.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
        else:
            t = targets[0]
            offsets = 0

        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        # Append
        a = t[:, 6].long()  # anchor indices
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class


    return tcls, tbox, indices, anch, id_index, tid



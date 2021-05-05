from numba import jit
from collections import deque
import torch
from utils.kalman_filter import KalmanFilter
from utils.log import logger
from models import *
from tracker import matching
from .basetrack import BaseTrack, TrackState
import numpy as np
from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.utils import scale_coords
import time
from utils.general import non_max_suppression

class STrack(BaseTrack):

    def __init__(self, tlwh, score, temp_feat, buffer_size=30, lost_frame_time=0, convince_time=0):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.lost_frame_time = lost_frame_time
        self.convince_time = convince_time

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

    def set_updata_var(self, mean, cov, bbox):
        self.mean = mean
        self.covariance = cov
        self._tlwh = bbox


    def set_box(self, bbox):
        self._tlwh = bbox


    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat 
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha *self.smooth_feat + (1-self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
        
    @staticmethod
    def multi_predict(stracks, kalman_filter):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
#            multi_mean, multi_covariance = STrack.kalman_filter.multi_predict(multi_mean, multi_covariance)
            multi_mean, multi_covariance = kalman_filter.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    @jit
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    @jit
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    @jit
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    @jit
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    @jit
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker(object):
    def __init__(self, opt, frame_rate=30):
        device = 'cuda:0'
        self.opt = opt
        self.model = attempt_load(opt.weights, map_location=device)  # load FP32 model
        # load_darknet_weights(self.model, opt.weights)
        # self.model.load_state_dict(torch.load(opt.weights, map_location='cpu')['model'], strict=False)
        self.model.cuda().eval()

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size

        self.kalman_filter = KalmanFilter()

    def update(self, im_blob, img0):
        """
        Processes the image frame and finds bounding box(detections).

        Associates the detection with corresponding tracklets and also handles lost, removed, refound and active tracklets

        Parameters
        ----------
        im_blob : torch.float32
                  Tensor of shape depending upon the size of image. By default, shape of this tensor is [1, 3, 608, 1088]

        img0 : ndarray
               ndarray of shape depending on the input image sequence. By default, shape is [608, 1080, 3]

        Returns
        -------
        output_stracks : list of Strack(instances)
                         The list contains information regarding the online_tracklets for the recieved image tensor.

        """

        self.frame_id += 1

        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // 4,
                'out_width': inp_width // 4}

        activated_starcks = []      # for storing active tracks, for the current frame
        refind_stracks = []         # Lost Tracks whose detections are obtained in the current frame
        lost_stracks = []           # The tracks which are not obtained in the current frame but are not removed.(Lost for some time lesser than the threshold for removing)
        removed_stracks = []

        t1 = time.time()
        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            pred, _ = self.model(im_blob, augment=False)
            # pred = self.model(im_blob, augment=False)[0]
        # pred is tensor of all the proposals (default number of proposals: 54264). Proposals have information associated with the bounding box and embeddings
        pred = pred[pred[:, :, 4] > self.opt.conf_thres]
        # pred = pred[0][pred[0][:,:,4]>self.opt.conf_thres]
        # pred now has lesser number of proposals. Proposals rejected on basis of object confidence score
        if len(pred) > 0:
            dets = non_max_suppression(pred.unsqueeze(0), self.opt.conf_thres, self.opt.nms_thres)[0].cpu()
            # Final proposals are obtained in dets. Information of bounding box and embeddings also included
            # Next step changes the detection scales (864, 480)
            scale_coords((1088, 608), dets[:, :4], img0.shape).round()
            '''Detections is list of (x1, y1, x2, y2, object_conf, class_score, class_pred)'''
            # class_pred is the embeddings.

            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f.numpy(), 30) for
                          (tlbrs, f) in zip(dets[:, :5], dets[:, 6:])]
        else:
            detections = []

        t2 = time.time()
        # print('Forward: {} s'.format(t2-t1))

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                # previous tracks which are not active in the current frame are added in unconfirmed list
                unconfirmed.append(track)
                # print("Should not be here, in unconfirmed")
            else:
                # Active tracks are added to the local list 'tracked_stracks'
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        # Combining currently tracked_stracks and lost_stracks
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool, self.kalman_filter)


        dists = matching.embedding_distance(strack_pool, detections)
        # dists = matching.gate_cost_matrix(self.kalman_filter, dists, strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        # The dists is the list of distances of the detection with the tracks in strack_pool
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)
        # The matches is the array for corresponding matches of the detection with the corresponding strack_pool

        for itracked, idet in matches:
            # itracked is the id of the track and idet is the detection
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                # If the track is active, add the detection to the track
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)

                track.convince_time+=1  # calcuate times
                track.lost_frame_time=0
            else:
                # We have obtained a detection from a track which is not active, hence put the track in refind_stracks list
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

                track.convince_time+=1  # calcuate times
                track.lost_frame_time=0

        # None of the steps below happen if there are no undetected tracks.
        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        # detections is now a list of the unmatched detections
        r_tracked_stracks = [] # This is container for stracks which were tracked till the
        # previous frame but no detection was found for it in the current frame
        for i in u_track:
            if strack_pool[i].state == TrackState.Tracked:
                r_tracked_stracks.append(strack_pool[i])
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)
        # matches is the list of detections which matched with corresponding tracks by IOU distance method
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)

                track.convince_time+=1   # calcuate times
                track.lost_frame_time=0
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

                track.convince_time+=1   # calcuate times
                track.lost_frame_time=0
        # Same process done for some unmatched detections, but now considering IOU_distance as measure

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        # If no detections are obtained for tracks (u_track), the tracks are added to lost_tracks list and are marked lost

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])

            detections[idet].convince_time+=1   # calcuate times
            detections[idet].lost_frame_time=0

        # The tracks which are yet not matched
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # after all these confirmation steps, if a new detection is found, it is initialized for a new track
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

            track.convince_time += 1  # calcuate times
            track.lost_frame_time = 0

        """ Step 5: Update state"""
        # If the tracks are lost for more frames than the threshold number, the tracks are removed.
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        # print('Remained match {} s'.format(t4-t3))

        # Update the self.tracked_stracks and self.lost_stracks using the updates in this step.
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        # self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]  # type: list[STrack]
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)


        ''' Get Lost_stracks information and  deal with lost object'''
        new_lost_stracks = []

        # print(self.lost_stracks)
        for ob in self.lost_stracks:

            lost_stracks_bbox = STrack.tlwh_to_xyah(ob.tlwh)   # bbox xyah

            flag = None

            pre_lost_mean, pre_lost_convariance = self.kalman_filter.predict(ob.mean, ob.covariance)
            pre_lost_mean, pre_lost_convariance = self.kalman_filter.update(pre_lost_mean, pre_lost_convariance, lost_stracks_bbox)

            ob.set_updata_var(pre_lost_mean, pre_lost_convariance, pre_lost_mean[:4])
            ob.lost_frame_time += 1

            new_bboxxyah, revise_flag = bbox_manxsize_limit(lost_stracks_bbox, pre_lost_mean[:4])
            if revise_flag:
                print('REVISE')
                ob.set_box(new_bboxxyah)


            flag = lost_stracks_limitation(ob)

            # if flag:
            #     new_lost_stracks.append(ob)

            if flag:
                for detect in self.tracked_stracks:
                    test_flag = compare_iou(detect.tlwh, ob.tlwh)

                    if (test_flag ==False):
                        flag=test_flag

                        ob.lost_frame_time=100

                        break

                if (ob.mean[0]<ob.mean[2] * ob.mean[3] *0.5) | (ob.mean[0] >width- ob.mean[2] * ob.mean[3] *0.5):
                    flag =False
                    print('边界滤除')
                    ob.lost_frame_time=100


            if flag:
                new_lost_stracks.append(ob)


        output_new = joint_stracks(self.tracked_stracks, new_lost_stracks)   # 整合两个数据
        # get scores of lost tracks
        output_new_stracks = [track for track in output_new if track.is_activated]    #选出已经跟踪过的目标
        # original output
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]


        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        return output_new_stracks


def lost_stracks_limitation(track):   # limit lsot min frame
    # i =0
    # while(i<30):
    #     if track.lost_frame_time < i+1 & track.convince_time > i:
    #         return True
    #     else:
    #         i+=1
    i=0
    while(i<3):
        if track.frame_id <10:
            if track.lost_frame_time<i+3:
                return True
        if track.lost_frame_time <i+1 & track.convince_time >i :
            return True
        else:
            i+=1

def calculate_S(first_bbox,pred_bbox,setting = 1.015):
    fist_xyah = STrack.tlwh_to_xyah(first_bbox)
    pred_xyah = STrack.tlwh_to_xyah(pred_bbox)
    ratio = (pred_bbox[2] *pred_bbox[3])/(first_bbox[2]*first_bbox[3])
    flag =False
    if ratio>setting:
        pred_xyah[:2] /=setting
        pred_xyah[2] =first_bbox[2]
        flag =True
    if pred_xyah[3] / fist_xyah[3] >setting:
        pred_xyah[3]=fist_xyah[3] * setting
        flag=True
    return first_bbox,flag



def bbox_manxsize_limit(bbox1, bbox2, scale=1.015):   # bbox1 is the origin box and bbox is the updated box (bbox use xyah)


    if len(bbox1) == len(bbox2):
        flag =False
        box1, box2 = bbox1, bbox2
        area1 = box1[2] * box1[3] * box1[3]
        area2 = box2[2] * box2[3] * box2[3]
        ratio = area2 / area1
        if ratio > scale:
            flag = True
            var = box2[3] / ratio
            box2[3] = var


        return bbox2, flag


    else:
        print('Two box size do not match,please check them size')



def compare_iou(detection_tlwh, pred_tlwh):
    detection_xy, pred_xy = tlwh_to_xmin_ymin_xmax_ymax(detection_tlwh), tlwh_to_xmin_ymin_xmax_ymax(pred_tlwh)    # 转换坐标到tlbr
    if (detection_xy[2] < pred_xy[0]) or (detection_xy[0] > pred_xy[2]) or  (detection_xy[1] > pred_xy[3] ) or (detection_xy[3] < pred_xy[1]):  # 判断是否相交
        return None

    w = min(detection_xy[2], pred_xy[2]) - max(detection_xy[0], pred_xy[0])  # 相交获取宽度
    h = min(detection_xy[3], pred_xy[3]) - max(detection_xy[1], pred_xy[1])  # 相交换取高度
    pred_S = (pred_xy[2]-pred_xy[0]) * (pred_xy[3] - pred_xy[1])             # 预测框面积计算
    detect_S = (detection_xy[2]-detection_xy[0]) * (detection_xy[3]-detection_xy[1])   # 当前帧所跟踪到的框面积计算
    I = w * h   # 相交面积
    U = (detection_xy[2] - detection_xy[0]) * (detection_xy[3] - detection_xy[1]) + (pred_xy[2] - pred_xy[0]) * (
                pred_xy[3] - pred_xy[1]) - I   # 两个框并集面积
    if( pred_S == I):    # 判断是否完全重合
        print('this is cover')
        return False
    if (I /pred_S  > 0.9):    # default = 0.7 判断重合是否超过0.9
        print('little ')
        return False
    if I/U > 0.95:           # 判读那IOU是否超过0.95
        print('this is iOU >95 &')
        return False
    if (I/U < 0.05) & (pred_S / detect_S < 0.5):  # 判断遮挡框下的错误框
        return False


def tlwh_to_xmin_ymin_xmax_ymax(detection_tlwh):
    xmin = detection_tlwh[1]
    ymin = detection_tlwh[0] - detection_tlwh[3]
    xmax = detection_tlwh[1] + detection_tlwh[2]
    ymax = detection_tlwh[0]
    return [xmin, ymin, xmax, ymax]



def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist<0.15)
    dupa, dupb = list(), list()
    for p,q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i,t in enumerate(stracksa) if not i in dupa]
    resb = [t for i,t in enumerate(stracksb) if not i in dupb]
    return resa, resb
            


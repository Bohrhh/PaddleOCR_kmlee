# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import paddle
from shapely.geometry import Polygon
import pyclipper


class DBPostProcess(object):
    """
    The post process for Differentiable Binarization (DB).
    """

    def __init__(self,
                 thresh=0.3,
                 box_thresh=0.7,
                 max_candidates=1000,
                 unclip_ratio=2.0,
                 use_dilation=False,
                 score_mode="fast",
                 polygon=False,
                 **kwargs):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.score_mode = score_mode
        self.polygon = polygon
        assert score_mode in [
            "slow", "fast"
        ], "Score mode must be in [slow, fast] but got: {}".format(score_mode)

        self.dilation_kernel = None if not use_dilation else np.array(
            [[1, 1], [1, 1]])

    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (H, W),
            whose values are binarized as {0, 1}
        '''

        assert len(_bitmap.shape) == 2
        bitmap = _bitmap  # The first channel
        height, width = bitmap.shape
        boxes = []
        scores = []

        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[:self.max_candidates]:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            # _, sside = self.get_mini_boxes(contour)
            # if sside < self.min_size:
            #     continue
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box)
            scores.append(score)
        return boxes, scores

    def bboxCutLine(self, bbox):
        """
        Args:
            bbox: (center (x,y), (width, height), angle of rotation)
        Return:
            line: (a,b,c), line represent ax+by+c=0
        """
        p1,p2,p3,p4 = cv2.boxPoints(bbox)
        if ((p1-p2)**2).sum()>((p2-p3)**2).sum():
            p1 = (p1 + p2)/2
            p2 = (p3 + p4)/2
        else:
            p1 = (p1 + p4)/2
            p2 = (p2 + p3)/2
        x1,y1 = p1; x2,y2 = p2
        a = y2-y1
        b = x1-x2
        c = x2*y1-x1*y2
        return a,b,c

    def polygon2boxes(self, contour, recursions=0, maxRecursions=3):
        """
        Args:
            contour: (n, 2)
        
        Return:
            regular polygon: (n, 2)
        """
        contour = np.array(contour).astype(np.int64)
        bbox = cv2.minAreaRect(contour) # (center (x,y), (width, height), angle of rotation)
        area_contour = cv2.contourArea(contour)
        area_bbox = bbox[1][0] * bbox[1][1]
        print(f"current recursions:{recursions}")
        if area_contour/area_bbox > 0.9 or recursions>=maxRecursions:
            return [cv2.boxPoints(bbox)]
        elif area_contour/area_bbox > 0.7 and bbox[1][0]/bbox[1][1] < 1.1 and bbox[1][0]/bbox[1][1] > 0.9:
            return [cv2.boxPoints(bbox)]
        else:
            a,b,c = self.bboxCutLine(bbox)
            contour1, contour2 = self.split_polygon_by_line(contour, a, b, c)
            return self.polygon2boxes(contour1, recursions+1, maxRecursions)+self.polygon2boxes(contour2, recursions+1, maxRecursions)

    def regular_polygon(self, contour, maxRecursions):
        boxes = self.polygon2boxes(contour, maxRecursions=maxRecursions)
        length = len(boxes)
        if length==1:
            return boxes[0]
        pointSide1 = [boxes[0][1]]
        pointSide2 = [boxes[0][2]]
        for i in range(length-1):
            pointSide1.append((boxes[i][0]+boxes[i+1][1])/2)
            pointSide2.append((boxes[i][3]+boxes[i+1][2])/2)
        pointSide1 += [boxes[-1][0]]
        pointSide2 += [boxes[-1][3]]
        polygon = np.array(pointSide1 + pointSide2[::-1])
        return polygon
        
    def split_polygon_by_line(self, contour, a, b, c):
        """
        line: ax+by+c=0
        """
        contour1 = []
        contour2 = []
        def get_vertical_point(x0,y0,a,b,c):
            x1 = (b**2*x0-a*b*y0-a*c)/(a**2+b**2)
            y1 = (a**2*y0-a*b*x0-b*c)/(a**2+b**2)
            return x1, y1
        def get_near_point(p1,p2,a,b,c):
            d1 = abs(a*p1[0]+b*p1[1]+c)/np.sqrt(a**2+b**2)
            d2 = abs(a*p2[0]+b*p2[1]+c)/np.sqrt(a**2+b**2)
            if d1>d2:
                return p2
            else:
                return p1
        last_point_side = 1 if a * contour[0][0] + b * contour[0][1] + c > 0 else -1
        cross_num = 0
        for i, point in enumerate(contour):
            if a * point[0] + b * point[1] + c > 0:
                if last_point_side == 1:
                    contour1.append(point)
                else:
                    near_point = get_near_point(contour[i-1], point, a, b, c)
                    vertical_point = get_vertical_point(near_point[0], near_point[1], a, b, c)
                    contour1.append(vertical_point)
                    contour2.append(vertical_point)
                    contour1.append(point)
                    cross_num += 1
                last_point_side = 1
            else:
                if last_point_side == -1:
                    contour2.append(point)
                else:
                    near_point = get_near_point(contour[i-1], point, a, b, c)
                    vertical_point = get_vertical_point(near_point[0], near_point[1], a, b, c)
                    contour1.append(vertical_point)
                    contour2.append(vertical_point)
                    contour2.append(point)
                    cross_num += 1
                last_point_side = -1
        assert cross_num==2, f"current cross num is {cross_num}, should be 2"
        return contour1, contour2

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            if self.score_mode == "fast":
                score = self.box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self.box_score_slow(pred, contour)
            if self.box_thresh > score:
                continue

            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int16))
            scores.append(score)
        return np.array(boxes, dtype=np.int16), scores

    def unclip(self, box):
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        '''
        box_score_fast: use bbox mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def box_score_slow(self, bitmap, contour):
        '''
        box_score_slow: use polyon mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, outs_dict, shape_list):
        pred = outs_dict['maps']
        if isinstance(pred, paddle.Tensor):
            pred = pred.numpy()
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(
                    np.array(segmentation[batch_index]).astype(np.uint8),
                    self.dilation_kernel)
            else:
                mask = segmentation[batch_index]
            if self.polygon:
                boxes, scores = self.polygons_from_bitmap(pred[batch_index], mask,
                                                    src_w, src_h)
            else:
                boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask,
                                                    src_w, src_h)

            boxes_batch.append({'points': boxes})
        return boxes_batch


class DistillationDBPostProcess(object):
    def __init__(self, model_name=["student"],
                 key=None,
                 thresh=0.3,
                 box_thresh=0.6,
                 max_candidates=1000,
                 unclip_ratio=1.5,
                 use_dilation=False,
                 score_mode="fast",
                 **kwargs):
        self.model_name = model_name
        self.key = key
        self.post_process = DBPostProcess(thresh=thresh,
                 box_thresh=box_thresh,
                 max_candidates=max_candidates,
                 unclip_ratio=unclip_ratio,
                 use_dilation=use_dilation,
                 score_mode=score_mode)

    def __call__(self, predicts, shape_list):
        results = {}
        for k in self.model_name:
            results[k] = self.post_process(predicts[k], shape_list=shape_list)
        return results

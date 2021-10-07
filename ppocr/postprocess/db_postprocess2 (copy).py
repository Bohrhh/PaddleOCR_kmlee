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
from skimage.morphology._skeletonize import thin


class DBPostProcess2(object):
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
                 pts_num=6,
                 offset_expand=1.5,
                 **kwargs):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.score_mode = score_mode
        self.pts_num = pts_num
        self.offset_expand = offset_expand
        assert score_mode in [
            "slow", "fast"
        ], "Score mode must be in [slow, fast] but got: {}".format(score_mode)

        self.dilation_kernel = None if not use_dilation else np.array(
            [[1, 1], [1, 1]])


    def unclip(self, box):
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_SQUARE, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded


    def sort_with_direction(self, pos_list):
        # compute direction
        pos_list = np.array(pos_list)
        rank = 4
        coeff = np.polyfit(pos_list[:,1], pos_list[:,0], rank)
        coeff_derivative = np.array([(rank-i)*coeff[i] for i in range(rank)])
        direction_y = np.poly1d(coeff_derivative)(pos_list[:,1])
        direction_x = np.ones_like(direction_y)
        direction = np.stack([direction_y, direction_x], axis=1)
        direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
        
        # sort with direction
        def sort_part_with_direction(pos_list, point_direction):
            pos_list = np.array(pos_list).reshape(-1, 2)
            point_direction = np.array(point_direction).reshape(-1, 2)
            average_direction = np.mean(point_direction, axis=0, keepdims=True)
            pos_proj_leng = np.sum(pos_list * average_direction, axis=1)
            sorted_list = pos_list[np.argsort(pos_proj_leng)].tolist()
            sorted_direction = point_direction[np.argsort(pos_proj_leng)].tolist()
            return sorted_list, sorted_direction

        sorted_point, sorted_direction = sort_part_with_direction(pos_list, direction)

        point_num = len(sorted_point)
        if point_num >= 64:
            middle_num = point_num // 2
            first_part_point = sorted_point[:middle_num]
            first_point_direction = sorted_direction[:middle_num]
            sorted_fist_part_point, sorted_fist_part_direction = sort_part_with_direction(
                first_part_point, first_point_direction)

            last_part_point = sorted_point[middle_num:]
            last_point_direction = sorted_direction[middle_num:]
            sorted_last_part_point, sorted_last_part_direction = sort_part_with_direction(
                last_part_point, last_point_direction)
            sorted_point = sorted_fist_part_point + sorted_last_part_point
            sorted_direction = sorted_fist_part_direction + sorted_last_part_direction

        return sorted_point, sorted_direction

    def sort_and_expand(self, pos_list, binary_tcl_map):

        h, w = binary_tcl_map.shape

        sorted_list, point_direction = self.sort_with_direction(pos_list)

        point_num = len(sorted_list)
        sub_direction_len = max(point_num // 3, 2)
        left_direction = point_direction[:sub_direction_len]
        right_dirction = point_direction[point_num - sub_direction_len:]

        left_average_direction = -np.mean(left_direction, axis=0, keepdims=True)
        left_average_len = np.linalg.norm(left_average_direction)
        left_start = np.array(sorted_list[0])
        left_step = left_average_direction / (left_average_len + 1e-6)

        right_average_direction = np.mean(right_dirction, axis=0, keepdims=True)
        right_average_len = np.linalg.norm(right_average_direction)
        right_step = right_average_direction / (right_average_len + 1e-6)
        right_start = np.array(sorted_list[-1])

        append_num = max(
            int((left_average_len + right_average_len) / 2.0 * 0.15), 1)
        max_append_num = 2 * append_num

        left_list = []
        right_list = []
        left_direction_list = []
        right_direction_list = []
        for i in range(max_append_num):
            ly, lx = np.round(left_start + left_step * (i + 1)).flatten().astype(
                'int32').tolist()
            if ly < h and lx < w and (ly, lx) not in left_list:
                if binary_tcl_map[ly, lx] > 0.5:
                    left_list.append((ly, lx))
                    left_direction_list.append(point_direction[0])
                else:
                    break

        for i in range(max_append_num):
            ry, rx = np.round(right_start + right_step * (i + 1)).flatten().astype(
                'int32').tolist()
            if ry < h and rx < w and (ry, rx) not in right_list:
                if binary_tcl_map[ry, rx] > 0.5:
                    right_list.append((ry, rx))
                    right_direction_list.append(point_direction[-1])
                else:
                    break

        all_list = left_list[::-1] + sorted_list + right_list
        all_direction = left_direction_list[::-1] + point_direction + right_direction_list
        return all_list, all_direction

    def reduce_points(self, points, directions, pts_num=4):
        detal = len(points) // (pts_num - 1)
        keep_idx_list = [0] + [detal * (i + 1) for i in range(pts_num - 2)] + [-1]
        keep_points = [points[idx] for idx in keep_idx_list]
        keep_directions = [directions[idx] for idx in keep_idx_list]
        return keep_points, keep_directions

    def point_pair2poly(self, point_pair_list):
        """
        Transfer vertical point_pairs into poly point in clockwise.
        """
        point_num = len(point_pair_list) * 2
        point_list = [0] * point_num
        for idx, point_pair in enumerate(point_pair_list):
            point_list[idx] = point_pair[0]
            point_list[point_num - 1 - idx] = point_pair[1]
        return np.array(point_list).reshape(-1, 2)

    def shrink_quad_along_width(self, quad, begin_width_ratio=0., end_width_ratio=1.):
        ratio_pair = np.array(
            [[begin_width_ratio], [end_width_ratio]], dtype=np.float32)
        p0_1 = quad[0] + (quad[1] - quad[0]) * ratio_pair
        p3_2 = quad[3] + (quad[2] - quad[3]) * ratio_pair
        return np.array([p0_1[0], p0_1[1], p3_2[1], p3_2[0]])

    def expand_poly_along_width(self, poly, shrink_ratio_of_width=0.3):
        """
        expand poly along width.
        """
        point_num = poly.shape[0]
        left_quad = np.array(
            [poly[0], poly[1], poly[-2], poly[-1]], dtype=np.float32)
        left_ratio = -shrink_ratio_of_width * np.linalg.norm(left_quad[0] - left_quad[3]) / \
                    (np.linalg.norm(left_quad[0] - left_quad[1]) + 1e-6)
        left_quad_expand = self.shrink_quad_along_width(left_quad, left_ratio, 1.0)
        right_quad = np.array(
            [
                poly[point_num // 2 - 2], poly[point_num // 2 - 1],
                poly[point_num // 2], poly[point_num // 2 + 1]
            ],
            dtype=np.float32)
        right_ratio = 1.0 + shrink_ratio_of_width * np.linalg.norm(right_quad[0] - right_quad[3]) / \
                    (np.linalg.norm(right_quad[0] - right_quad[1]) + 1e-6)
        right_quad_expand = self.shrink_quad_along_width(right_quad, 0.0, right_ratio)
        poly[0] = left_quad_expand[0]
        poly[-1] = left_quad_expand[-1]
        poly[point_num // 2 - 1] = right_quad_expand[1]
        poly[point_num // 2] = right_quad_expand[2]
        return poly

    def find_offset(self, y, x, dy, dx, p_tcl_map):
        offset = []
        h, w = p_tcl_map.shape[:2]
        max_len = int((h*w)**0.5)
        ny, nx = dx, -dy 
        for i in range(1, max_len):
            off_y, off_x = round(ny*i), round(nx*i)
            new_y, new_x = off_y+y, off_x+x
            if new_y==h-1 or new_y==0 or new_x==w-1 or new_x==0:
                break
            if p_tcl_map[new_y, new_x]<0.5:
                break
        offset.extend([off_y, off_x])

        ny, nx = -dx, dy 
        for i in range(1, max_len):
            off_y, off_x = round(ny*i), round(nx*i)
            new_y, new_x = off_y+y, off_x+x
            if new_y==h-1 or new_y==0 or new_x==w-1 or new_x==0:
                break
            if p_tcl_map[new_y, new_x]<0.5:
                break
        offset.extend([off_y, off_x])
        return offset

    def restore_poly(self, poly_yxs_list, direction_yxs_list, p_tcl_map, 
                     src_h, src_w, ratio_h, ratio_w):
        """
        Args:
            poly_yxs_list:
            direction_yxs_list:
            p_tcl_map: (h, w)
        """
        poly_list = []
        for yx_center_line, yx_direction in zip(poly_yxs_list, direction_yxs_list):
            point_pair_list = []
            num_points = len(yx_center_line)

            for i in range(num_points):
                y,x = yx_center_line[i]
                dy,dx = yx_direction[i]
                offset = self.find_offset(y,x,dy,dx,p_tcl_map)
                offset = np.array(offset).reshape(2,2)*self.offset_expand
                ori_yx = np.array([y, x], dtype=np.float32)
                point_pair = (ori_yx + offset)[:, ::-1] / np.array([ratio_w, ratio_h]).reshape(-1, 2)
                point_pair_list.append(point_pair)

            detected_poly = self.point_pair2poly(point_pair_list)
            # detected_poly = self.expand_poly_along_width(
            #     detected_poly, shrink_ratio_of_width=0.2)
            detected_poly = self.unclip(detected_poly)
            detected_poly[:, 0] = np.clip(detected_poly[:, 0], a_min=0, a_max=src_w)
            detected_poly[:, 1] = np.clip(detected_poly[:, 1], a_min=0, a_max=src_h)

            poly_list.append(detected_poly)
        return poly_list

    def __call__(self, outs_dict, shape_list):
        pred = outs_dict['maps']
        if isinstance(pred, paddle.Tensor):
            pred = pred.numpy()
        pred = pred[:, 0, :, :]
        p_tcl_maps = (pred > self.thresh) * 1.0

        boxes_batch = []
        for batch_index, p_tcl_map in enumerate(p_tcl_maps):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            if self.dilation_kernel is not None:
                p_tcl_map = cv2.dilate(
                    np.array(p_tcl_map).astype(np.uint8),
                    self.dilation_kernel)

            skeleton_map = thin(p_tcl_map.astype(np.uint8))
            instance_count, instance_label_map = cv2.connectedComponents(
                skeleton_map.astype(np.uint8), connectivity=8)
            # get TCL Instance
            all_pos_yxs = []
            all_direction_yxs = []
            if instance_count > 0:
                for instance_id in range(1, instance_count):
                    pos_list = []
                    ys, xs = np.where(instance_label_map == instance_id)
                    pos_list = list(zip(ys, xs))

                    if len(pos_list) < 3:
                        continue
                        
                    pos_list_sorted, direction_list_sorted = self.sort_and_expand(pos_list, p_tcl_map)
                    pos_list_sorted, direction_list_sorted = self.reduce_points(pos_list_sorted, direction_list_sorted, self.pts_num)
                    all_pos_yxs.append(pos_list_sorted)
                    all_direction_yxs.append(direction_list_sorted)
            # resore poly
            poly_list = self.restore_poly(all_pos_yxs, all_direction_yxs, p_tcl_map, src_h, src_w, ratio_h, ratio_w)
            return poly_list
            boxes_batch.append({'points': poly_list})
        return boxes_batch


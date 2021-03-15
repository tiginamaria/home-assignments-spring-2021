#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import cv2
import numpy as np

import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    triangulate_correspondences,
    build_correspondences, TriangulationParameters, rodrigues_and_translation_to_view_mat3x4,
    eye3x4
)
from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose

MAX_REPROJECTION_ERROR = 1.0
MAX_TRIANGULATION_ANGLE_DEG = 1.0
MIN_DEPTH = 0.1

THRESHOLD = 1.0
CONFIDENCE = 0.99
REPROJECTION_ERROR = 2.0
MIN_ANGLE = 1
MIN_INLINERS = 10
ESSENTIAL_VALIDATION_THRESHOLD = 0.9


class CameraTrackBuilder:
    def __init__(self,
                 corner_storage: CornerStorage,
                 intrinsic_mat: np.ndarray,
                 point_cloud_builder: PointCloudBuilder,
                 known_view_1: Optional[Tuple[int, Pose]] = None,
                 known_view_2: Optional[Tuple[int, Pose]] = None):
        self.parameters = TriangulationParameters(max_reprojection_error=MAX_REPROJECTION_ERROR,
                                                  min_triangulation_angle_deg=MAX_TRIANGULATION_ANGLE_DEG,
                                                  min_depth=MIN_DEPTH)
        self.corner_storage = corner_storage
        self.intrinsic_mat = intrinsic_mat
        self.point_cloud_builder = point_cloud_builder

        self.frames_cnt = len(corner_storage)
        self.view_mats: List[Optional[np.ndarray]] = [None] * self.frames_cnt

        self._init_camera(known_view_1, known_view_2)

    def _init_camera(self, known_view_1: Optional[Tuple[int, Pose]], known_view_2: Optional[Tuple[int, Pose]]):
        if known_view_1 is None or known_view_2 is None:
            frame_id1, frame_id2, mat1, mat2 = self._calc_camera_mats()
        else:
            frame_id1, mat1 = known_view_1[0], pose_to_view_mat3x4(known_view_1[1])
            frame_id2, mat2 = known_view_2[0], pose_to_view_mat3x4(known_view_2[1])

        self.view_mats[frame_id1] = mat1
        self.view_mats[frame_id2] = mat2
        self._update_points(frame_id1, frame_id2, None)

    def _calc_camera_mats(self):
        essential_validation_threshold = 0.9
        min_inliers = 10

        best_points_count = 0
        best_frame_id1, best_frame_id2 = None, None
        best_mat1, best_mat2 = eye3x4(), None

        triangulation_parameters = TriangulationParameters(2.0, 1.0, 0)

        for frame_id1 in range(0, self.frames_cnt, 3):
            for frame_id2 in range(frame_id1 + 2, min(frame_id1 + 100, self.frames_cnt), 3):
                print(f'try frames pair ({frame_id1}, {frame_id2}) to build camera mats')

                corrs = build_correspondences(self.corner_storage[frame_id1],
                                              self.corner_storage[frame_id2])
                points_ids = corrs.ids
                if len(points_ids) < 10:
                    continue

                points1, points2 = corrs.points_1, corrs.points_2

                e_retval, e_mask = cv2.findEssentialMat(points1, points2, self.intrinsic_mat, method=cv2.RANSAC, threshold=1)
                h_retval, h_mask = cv2.findHomography(points1, points2, method=cv2.RANSAC, ransacReprojThreshold=2)

                e_mask = e_mask.flatten() == 1
                h_mask = h_mask.flatten() == 1
                e_inliers = points1[e_mask]
                h_inliers = points2[h_mask]

                if len(e_inliers) / len(h_inliers) < essential_validation_threshold or len(e_inliers) < min_inliers:
                    print("too close centers, can not build essential matrix")
                    continue

                e_outliers = np.delete(points_ids, e_mask)
                corrs = build_correspondences(self.corner_storage[frame_id1], self.corner_storage[frame_id2], e_outliers)

                R1, R2, t = cv2.decomposeEssentialMat(e_retval)

                for R in [R1, R2]:
                    for t in [t, -t]:
                        mat1, mat2 = best_mat1, np.hstack((R, t))
                        points, ids, _ = triangulate_correspondences(corrs, mat1, mat2, self.intrinsic_mat,
                                                                     triangulation_parameters)

                        if len(points) > best_points_count:
                            best_points_count = len(points)
                            best_mat2 = mat2
                            best_frame_id1, best_frame_id2 = frame_id1, frame_id2
        return best_frame_id1, best_frame_id2, best_mat1, best_mat2

    def _prepare_view_mats(self) -> List[np.ndarray]:
        for i in range(self.frames_cnt):
            if self.view_mats[i] is None:
                print(f'{i}-th camera matrix is None')
                self.view_mats[i] = self.view_mats[(i - 1) % self.frames_cnt]
        return self.view_mats

    def build_camera_mats(self) -> List[np.ndarray]:
        for frame_id, corners in enumerate(self.corner_storage):
            if self.view_mats[frame_id] is not None:
                continue

            intersections, corners_ids, cloud_ids = np.intersect1d(corners.ids,
                                                                   self.point_cloud_builder.ids,
                                                                   return_indices=True)
            points_2d = corners.points[corners_ids]
            points_3d = self.point_cloud_builder.points[cloud_ids]

            if len(intersections) < 3:
                continue

            if len(intersections) < 6:
                print(f'Cannot process frame with corners count less then 6')
                continue

            print(f'Processing frame: {frame_id}')
            retval0, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d, points_2d,
                                                              self.intrinsic_mat, None)
            retval1, rvec, tvec = cv2.solvePnP(points_3d[inliers], points_2d[inliers],
                                               self.intrinsic_mat, None,
                                               rvec=rvec, tvec=tvec, useExtrinsicGuess=True)
            if not retval0 or not retval1:
                print(f'Cannot process frame')
                continue

            self.view_mats[frame_id] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
            print(f'Inlieners count: {len(inliers)}')

            outliers = np.delete(intersections, inliers)
            for next_frame_id, _ in enumerate(self.corner_storage):
                if self.view_mats[next_frame_id] is not None:
                    self._update_points(frame_id, next_frame_id, outliers)

        return self._prepare_view_mats()

    def _update_points(self, frame_id1, frame_id2, outliers):
        frame1, frame2 = self.corner_storage[frame_id1], self.corner_storage[frame_id2]
        mat1, mat2 = self.view_mats[frame_id1], self.view_mats[frame_id2]
        correspondences = build_correspondences(frame1, frame2, outliers)
        if len(correspondences.ids) == 0:
            return 0
        points, ids, _ = triangulate_correspondences(correspondences, mat1, mat2,
                                                     self.intrinsic_mat,
                                                     self.parameters)
        print(f'Updated points count: {len(points)}')
        self.point_cloud_builder.add_points(ids, points)
        print(f'Total points count: {len(self.point_cloud_builder.ids)}')
        return ids


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    point_cloud_builder = PointCloudBuilder()
    camera_track_builder = CameraTrackBuilder(corner_storage, intrinsic_mat, point_cloud_builder,
                                              known_view_1, known_view_2)
    view_mats = camera_track_builder.build_camera_mats()

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()

#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import cv2
import numpy as np

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    triangulate_correspondences,
    build_correspondences, TriangulationParameters, rodrigues_and_translation_to_view_mat3x4
)


class CameraTrackBuilder:

    def __init__(self,
                 corner_storage: CornerStorage,
                 intrinsic_mat: np.ndarray,
                 point_cloud_builder: PointCloudBuilder,
                 known_view_1: Optional[Tuple[int, Pose]] = None,
                 known_view_2: Optional[Tuple[int, Pose]] = None):
        self.parameters = TriangulationParameters(max_reprojection_error=1.0,
                                                  min_triangulation_angle_deg=1.0,
                                                  min_depth=0.1)
        self.corner_storage = corner_storage
        self.intrinsic_mat = intrinsic_mat
        self.point_cloud_builder = point_cloud_builder

        self.frames_cnt = len(corner_storage)
        self.view_mats: List[Optional[np.ndarray]] = [None] * self.frames_cnt

        self.view_mats[known_view_1[0]] = pose_to_view_mat3x4(known_view_1[1])
        self.view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])
        self._update_points(known_view_1[0], known_view_2[0])

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

            corner_ids = []
            points_3d = []
            points_2d = []
            for corner_id, point in zip(corners.ids, corners.points):
                cloud_corner_id, _ = np.where(self.point_cloud_builder.ids == corner_id)
                if len(cloud_corner_id) == 0:
                    continue
                corner_ids.append(corner_id)
                points_3d.append(self.point_cloud_builder.points[cloud_corner_id[0]])
                points_2d.append(point)

            if len(corner_ids) < 6:
                print(f'Cannot process frame with corners count less then 3')
                continue

            print(f'Processing frame: {frame_id}')
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(np.array(points_3d), np.array(points_2d),
                                                             self.intrinsic_mat, None, flags=cv2.SOLVEPNP_ITERATIVE)
            if not retval:
                print(f'Cannot process frame')
                continue

            self.view_mats[frame_id] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
            print(f'Inlieners count: {len(inliers)}')
            for next_frame_id, corners in enumerate(self.corner_storage):
                if self.view_mats[next_frame_id] is not None:
                    self._update_points(frame_id, next_frame_id)

        return self._prepare_view_mats()

    def _update_points(self, frame_id1, frame_id2):
        frame1, frame2 = self.corner_storage[frame_id1], self.corner_storage[frame_id2]
        mat1, mat2 = self.view_mats[frame_id1], self.view_mats[frame_id2]
        correspondences = build_correspondences(frame1, frame2)
        if len(correspondences.ids) == 0:
            return 0
        points, ids, _ = triangulate_correspondences(correspondences, mat1, mat2,
                                                     self.intrinsic_mat,
                                                     self.parameters)
        print(f'Updated points count: {len(points)}')
        self.point_cloud_builder.add_points(ids, points)
        print(f'Total points count: {len(self.point_cloud_builder.ids)}')
        return len(ids)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    point_cloud_builder = PointCloudBuilder()
    camera_track_builder = CameraTrackBuilder(corner_storage, intrinsic_mat, point_cloud_builder, known_view_1, known_view_2)
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

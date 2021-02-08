#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims
import matplotlib.pyplot as plt

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _show_corners(img, old_points, new_points):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for c in np.int0(new_points):
        x, y = c.ravel()
        cv2.circle(img, (x, y), 3, (1, 0, 0), -1)
    for c in np.int0(old_points):
        x, y = c.ravel()
        cv2.circle(img, (x, y), 3, (0, 0, 1), -1)
    cv2.imshow(f"corners", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def make_uint8(img):
    return (img * 256).astype(np.uint8)


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    block_size = 9
    winSize = 45
    qualityLevel = 0.0005
    maxCorners = 200
    minDistance = 30
    minRadius = 30
    maxLevel = 5

    lk_params = dict(winSize=(winSize, winSize),
                     maxLevel=maxLevel)

    prev_img = None
    prev_corners = None
    last_id = 0

    for frame_id, img in enumerate(frame_sequence[0:]):
        draw_img = img
        img = make_uint8(img)
        if frame_id == 0:
            points = cv2.goodFeaturesToTrack(img, maxCorners, qualityLevel, minDistance, blockSize=block_size)
            corners_ids = [i for i in range(len(points))]
            new_points_count = last_id = len(points)
        else:
            cur_points = cv2.goodFeaturesToTrack(img, maxCorners, qualityLevel, minDistance, blockSize=block_size)
            track_points, track_st, _ = cv2.calcOpticalFlowPyrLK(prev_img, img, prev_corners.points, None, **lk_params)

            corners_ids = []
            points = []
            for i in range(len(track_points)):
                dist = np.linalg.norm(track_points[:i] - track_points[i], axis=1)
                if len(dist) == 0:
                    if i == 0 and track_st[i][0] == 1:
                        corners_ids.append(prev_corners.ids[i][0])
                        points.append(track_points[i])
                    continue
                j = np.argmin(dist)
                if dist[j] > minRadius and track_st[j][0] == 1:
                    corners_ids.append(prev_corners.ids[i][0])
                    points.append(track_points[i])

            new_points_count = 0
            for i in range(len(cur_points)):
                dist = np.linalg.norm(track_points - cur_points[i][0], axis=1)
                if len(dist) == 0 or dist[np.argmin(dist)] > minRadius:
                    corners_ids.append(last_id)
                    points.append(cur_points[i][0])
                    last_id += 1
                    new_points_count += 1

        # if new_points_count == 0:
        #     _show_corners(draw_img, points[:-new_points_count], [])
        # else:
        #     _show_corners(draw_img, points[:-new_points_count], points[-new_points_count:])

        corners = FrameCorners(
            np.array(corners_ids),
            np.array(points),
            np.array([block_size for _ in range(len(points))])
        )
        builder.set_corners_at_frame(frame_id, corners)
        prev_corners = corners
        prev_img = img.copy()


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter

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
    return


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    block_size = 9

    win_size = 45
    quality_level = 0.0005
    max_corners = frame_sequence[0].shape[0] * frame_sequence[0].shape[1] // 2000
    min_distance = 30
    min_radius = 30
    max_level = 5

    lk_params = dict(winSize=(win_size, win_size),
                     maxLevel=max_level)

    prev_img = frame_sequence[0]
    points = cv2.goodFeaturesToTrack(prev_img, max_corners, quality_level, min_distance, blockSize=block_size)
    corners_ids = list(range(len(points)))
    last_id = len(points)

    prev_corners = FrameCorners(
        np.array(corners_ids),
        np.array(points),
        np.array([block_size for _ in range(len(points))])
    )

    for frame_id, img in enumerate(frame_sequence[1:], 1):
        prev_img_8bit = cv2.convertScaleAbs(prev_img, alpha=255)
        img_8bit = cv2.convertScaleAbs(img, alpha=255)
        track_points, track_st, _ = cv2.calcOpticalFlowPyrLK(prev_img_8bit, img_8bit, prev_corners.points, None,
                                                             **lk_params)

        track_st = track_st.reshape(-1)
        points = track_points[track_st == 1]
        corners_ids = prev_corners.ids[track_st == 1]

        new_points = cv2.goodFeaturesToTrack(img, max_corners, quality_level, min_distance, blockSize=block_size)

        for i in range(len(new_points)):
            if len(points) == 0 or min(np.linalg.norm(points - new_points[i][0], axis=1)) > min_radius:
                corners_ids = np.append(corners_ids, [[last_id]], axis=0)
                points = np.append(points, [new_points[i][0]], axis=0)
                last_id += 1

        corners = FrameCorners(
            corners_ids,
            points,
            np.array([block_size for _ in range(len(points))])
        )
        builder.set_corners_at_frame(frame_id, corners)
        prev_corners = corners
        prev_img = img


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

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


def _show_corners(img, corners_points):
    for i in np.int0(corners_points):
        x, y = i.ravel()
        cv2.circle(img, (x, y), 3, (255, 0, 0), -1)

    plt.imshow(img)
    plt.show()


def _show_tracks(img_0, track_corners_0, img_1, track_corners_1):
    mask = np.zeros_like(img_0)
    color = np.random.randint(0, 255, (100, 3))

    for i, (new, old) in enumerate(zip(track_corners_0, track_corners_1)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        img_1 = cv2.circle(img_1, (a, b), 5, color[i].tolist(), -1)

    img = cv2.add(img_1, mask)
    cv2.imshow('tracks', img)


def make_grey(img):
    return (cv2.merge((img, img, img)) * 255.999).astype(np.uint8)


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    block_size = 7
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    prev_img = None
    prev_corners = None
    last_id = 0

    for frame_id, img in enumerate(frame_sequence[0:]):
        if frame_id == 0:
            points = cv2.goodFeaturesToTrack(img, 200, 0.01, 5, blockSize=block_size)
            corners_ids = [i for i in range(len(points))]
            last_id = len(points)
            img = make_grey(img)
        else:
            img = make_grey(img)
            points, prev_st, _ = cv2.calcOpticalFlowPyrLK(prev_img, img, prev_corners.points, None, **lk_params)
            prev_points, st, _ = cv2.calcOpticalFlowPyrLK(img, prev_img, points, None, **lk_params)

            corners_ids = []
            for i in range(len(prev_points)):
                if np.linalg.norm(prev_corners.points[i] - prev_points[i]) < 5:
                    corners_ids.append(prev_corners.ids[i][0])
                else:
                    corners_ids.append(last_id)
                    last_id += 1

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

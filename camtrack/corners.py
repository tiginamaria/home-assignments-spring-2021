#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims
import matplotlib.pyplot as plt

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


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


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    # TODO
    image_0 = frame_sequence[0]

    corners = FrameCorners(
        np.array([0]),
        np.array([[0, 0]]),
        np.array([7])
    )
    builder.set_corners_at_frame(0, corners)
    c = cv2.goodFeaturesToTrack(image_0, 100, 0.01, 10, blockSize=7)

    # convert corners values to integer
    # So that we will be able to draw circles on them
    corners = np.int0(c)

    plt.imshow(image_0)
    plt.show()
    # draw red color circles on all corners
    for i in c:
        x, y = i.ravel()
        cv2.circle(image_0, (x, y), 3, (255, 0, 0), -1)

        # resulting image
    plt.imshow(image_0)
    plt.show()

    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        builder.set_corners_at_frame(frame, corners)
        image_0 = image_1


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


def calc_track_len_array_mapping(corner_storage):
    return np.zeros(len(corner_storage))


def calc_track_interval_mappings(corner_storage):
    return np.array([f for f, _ in enumerate(corner_storage)]), \
           np.array([f for f, _ in enumerate(corner_storage)])


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter

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
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=500,
                          qualityLevel=0.01,
                          minDistance=7,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take first frame and find corners in it
    image_0 = frame_sequence[0]
    #image_0_gray = cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(image_0, mask=None, **feature_params).squeeze()
    ids = np.arange(len(p0))
    corners = FrameCorners(
        ids=ids,
        points=p0,
        sizes=np.array([feature_params['blockSize'] for i in range(len(ids))])
    )
    builder.set_corners_at_frame(0, corners)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(image_0)

    image_0 = np.uint8(image_0 * 255)
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        image_1 = np.uint8(image_1 * 255)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(image_0, image_1, p0, None, **lk_params)
        # Select good points
        detected_mask = (st == 1).squeeze()

        good_new = p1[detected_mask]
        ids = ids[detected_mask]

        corners = FrameCorners(
            ids=ids,
            points=good_new,
            sizes=np.array([feature_params['blockSize'] for i in range(len(ids))])
        )
        builder.set_corners_at_frame(frame, corners)

        image_0 = image_1
        p0 = p1


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

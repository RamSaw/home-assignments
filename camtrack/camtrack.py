#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import cv2

from corners import CornerStorage, FrameCorners
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import *


def add_points(frame1: Tuple[FrameCorners, np.ndarray], frame2: Tuple[FrameCorners, np.ndarray],
               points: List[Optional[np.ndarray]],
               intrinsic_mat: np.ndarray,
               triangulation_parameters: TriangulationParameters
               ) -> int:
    correspondences = build_correspondences(frame1[0], frame2[0])
    if len(correspondences.points_1) == 0 or len(correspondences.points_2) == 0:
        return 0
    possible_new_points, ids, _ = triangulate_correspondences(correspondences, frame1[1], frame2[1],
                                                              intrinsic_mat, triangulation_parameters)
    new_points_with_ids = list(filter(lambda el: points[el[1]] is None, zip(possible_new_points, ids)))
    for point, i in new_points_with_ids:
        points[i] = point
    return len(new_points_with_ids)


def get_points_cloud_size(points):
    return len(list(filter(lambda point: point is not None, points)))


def track_frame(track_id: int,
                track: List[Optional[np.ndarray]],
                points: List[Optional[np.ndarray]],
                corner_storage: CornerStorage,
                intrinsic_mat: np.ndarray,
                triangulation_parameters: TriangulationParameters
                ) -> bool:
    frame_corners_ids = corner_storage[track_id].ids.reshape(-1)
    existing = [(i, pt, points[i]) for i, pt in zip(frame_corners_ids, corner_storage[track_id].points) if points[i] is not None]
    if len(existing) < 5:
        return False
    existing_frame_corner_ids, frame_corners_points, existing_points = zip(*existing)
    existing_frame_corner_ids, frame_corners_points, existing_points = \
        np.array(existing_frame_corner_ids), np.array(frame_corners_points), np.array(existing_points)

    retval, rvec, tvec, inliers = cv2.solvePnPRansac(existing_points, frame_corners_points, intrinsic_mat, None)
    if not retval:
        return False

    untracked_point_ids = [i for i in frame_corners_ids if i not in inliers]
    for i in untracked_point_ids:
        points[i] = None

    track[track_id] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)

    print(f"Tracked frame {track_id} successfully")
    for j in range(len(track)):
        if track_id == j or track[j] is None:
            continue
        number_of_added_points = add_points((corner_storage[track_id], track[track_id]),
                                            (corner_storage[j], track[j]),
                                            points, intrinsic_mat, triangulation_parameters)
        if number_of_added_points != 0:
            print(f'New triangled points: {number_of_added_points}, frames: {track_id} -> {j}')
    print(f"inliers size: {len(inliers)}, point cloud size: {get_points_cloud_size(points)}")
    return True


def init_from_known_views(known_view_1: Tuple[int, Pose],
                          known_view_2: Tuple[int, Pose],
                          track: List[Optional[np.ndarray]],
                          points: List[Optional[np.ndarray]],
                          corner_storage: CornerStorage,
                          intrinsic_mat: np.ndarray,
                          triangulation_parameters: TriangulationParameters):
    print(f'Initial frame ids: {known_view_1[0]}, {known_view_2[0]}')

    track[known_view_1[0]] = pose_to_view_mat3x4(known_view_1[1])
    track[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])
    number_of_added_points = add_points((corner_storage[known_view_1[0]], track[known_view_1[0]]),
                                        (corner_storage[known_view_2[0]], track[known_view_2[0]]),
                                        points, intrinsic_mat, triangulation_parameters)
    if number_of_added_points != 0:
        print(f'New triangled points: {number_of_added_points}, frames: {known_view_1[0]} -> {known_view_2[0]}')


def track_camera(corner_storage: CornerStorage,
                 intrinsic_mat: np.ndarray,
                 triangulation_parameters: TriangulationParameters,
                 known_view_1: Tuple[int, Pose],
                 known_view_2: Tuple[int, Pose]) \
        -> Tuple[np.ndarray, PointCloudBuilder]:
    track = [None for _ in range(len(corner_storage))]
    points = [None for _ in range(corner_storage.max_corner_id() + 1)]

    init_from_known_views(known_view_1, known_view_2,
                          track, points,
                          corner_storage, intrinsic_mat, triangulation_parameters)

    has_new_track = True
    while has_new_track:
        has_new_track = False
        for i in range(len(track)):
            if track[i] is None:
                is_successful = track_frame(i, track, points, corner_storage, intrinsic_mat, triangulation_parameters)
                if is_successful:
                    has_new_track = True

    for i in range(1, len(track)):
        if track[i] is None:
            track[i] = track[i - 1]
    view_mats = np.array(track)

    point_cloud_builder = PointCloudBuilder(
        ids=np.array([i for i in range(len(points)) if points[i] is not None]),
        points=np.array([points[i] for i in range(len(points)) if points[i] is not None])
    )
    return view_mats, point_cloud_builder


def get_known_views(corner_storage: CornerStorage,
                    intrinsic_mat: np.ndarray,
                    triangulation_parameters: TriangulationParameters) -> Tuple[Tuple[int, Pose], Tuple[int, Pose]]:

    def get_pose_and_cloud_size(frame2: FrameCorners):
        correspondences = build_correspondences(corner_storage[initial_frame_ind], frame2)
        if len(correspondences.points_1) < 5:
            return None, -1

        _, homography_mask = cv2.findHomography(correspondences.points_1, correspondences.points_2,
                                                method=cv2.RANSAC,
                                                ransacReprojThreshold=triangulation_parameters.max_reprojection_error,
                                                confidence=0.999)
        essential_matrix, essential_mask = cv2.findEssentialMat(correspondences.points_1, correspondences.points_2,
                                                                intrinsic_mat)
        if essential_matrix is None or essential_matrix.shape != (3, 3) or \
                np.count_nonzero(essential_mask) < np.count_nonzero(homography_mask):
            return None, -1

        R1, R2, t = cv2.decomposeEssentialMat(essential_matrix)
        correspondences = remove_correspondences_with_ids(correspondences, np.argwhere(essential_mask == 0))
        poses = [Pose(R1.T, np.dot(R1.T, t)), Pose(R2.T, np.dot(R2.T, t)),
                 Pose(R1.T, np.dot(R1.T, (-t))), Pose(R2.T, np.dot(R2.T, (-t)))]
        triangulated = [triangulate_correspondences(correspondences, eye3x4(), pose_to_view_mat3x4(pose),
                                                    intrinsic_mat, triangulation_parameters) for pose in poses]
        _cloud_size, _best_ind = max((len(triangulated[i][0]), i) for i in range(4))
        return poses[_best_ind], _cloud_size

    max_cloud_size = -1
    best_pose = None
    best_ind = None
    initial_frame_ind = 0
    for i in range(1, len(corner_storage)):
        pose, cloud_size = get_pose_and_cloud_size(corner_storage[i])
        if cloud_size > max_cloud_size:
            max_cloud_size = cloud_size
            best_pose = pose
            best_ind = i
        print(f'Frames: {initial_frame_ind} and {i}, cloud size: {cloud_size}')

    return (0, view_mat3x4_to_pose(eye3x4())), (best_ind, best_pose)


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
    parameters = TriangulationParameters(max_reprojection_error=2.2, min_triangulation_angle_deg=1., min_depth=0.1)

    if known_view_1 is None or known_view_2 is None:
        print('No intital known views provided. Calculating initial views...')
        known_view_1, known_view_2 = get_known_views(corner_storage, intrinsic_mat, parameters)

    view_mats, point_cloud_builder = track_camera(
        corner_storage,
        intrinsic_mat,
        parameters,
        known_view_1,
        known_view_2
    )

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

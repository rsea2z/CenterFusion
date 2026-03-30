# Copyright (c) Xingyi Zhou. All Rights Reserved
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
from pyquaternion import Quaternion
import numpy as np
import json
import os
import warnings
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from nuscenes.utils.data_classes import Box
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.common.loaders import load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.render import visualize_sample
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.detection.utils import category_to_detection_name
from lightning.pytorch.utilities import rank_zero_only

from ..generic_dataset import GenericDataset, GenericDemo
from utils.pointcloud import map_pointcloud_to_image
from utils.pointcloud import RadarPointCloudWithVelocity as RadarPointCloud
from utils.image import getGaussianRadius
from utils.utils import safe_run, createFolder
from config import config, updateDatasetAndModelConfig


class nuScenes(GenericDataset):
    default_resolution = [900, 1600]
    num_categories = 10
    focal_length = 1200
    max_objs = 128

    class_name = [
        "car",
        "truck",
        "bus",
        "trailer",
        "construction_vehicle",
        "pedestrian",
        "motorcycle",
        "bicycle",
        "traffic_cone",
        "barrier",
    ]
    class_ids = {i + 1: i + 1 for i in range(num_categories)}

    vehicles = ["car", "truck", "bus", "trailer", "construction_vehicle"]
    cycles = ["motorcycle", "bicycle"]
    pedestrians = ["pedestrian"]

    attribute_to_id = {
        "": 0,
        "cycle.with_rider": 1,
        "cycle.without_rider": 2,
        "pedestrian.moving": 3,
        "pedestrian.standing": 4,
        "pedestrian.sitting_lying_down": 5,
        "vehicle.moving": 6,
        "vehicle.parked": 7,
        "vehicle.stopped": 8,
    }
    id_to_attribute = {v: k for k, v in attribute_to_id.items()}

    SENSOR_NAME = {
        1: "CAM_FRONT",
        2: "CAM_FRONT_RIGHT",
        3: "CAM_BACK_RIGHT",
        4: "CAM_BACK",
        5: "CAM_BACK_LEFT",
        6: "CAM_FRONT_LEFT",
        7: "RADAR_FRONT",
        8: "LIDAR_TOP",
        9: "RADAR_FRONT_LEFT",
        10: "RADAR_FRONT_RIGHT",
        11: "RADAR_BACK_LEFT",
        12: "RADAR_BACK_RIGHT",
    }
    RADARS_FOR_CAMERA = {
        "CAM_FRONT_LEFT": ["RADAR_FRONT_LEFT", "RADAR_FRONT"],
        "CAM_FRONT": ["RADAR_FRONT_RIGHT", "RADAR_FRONT_LEFT", "RADAR_FRONT"],
        "CAM_FRONT_RIGHT": ["RADAR_FRONT_RIGHT", "RADAR_FRONT"],
        "CAM_BACK_LEFT": ["RADAR_BACK_LEFT", "RADAR_FRONT_LEFT"],
        "CAM_BACK": ["RADAR_BACK_RIGHT", "RADAR_BACK_LEFT"],
        "CAM_BACK_RIGHT": ["RADAR_BACK_RIGHT", "RADAR_FRONT_RIGHT"],
    }
    SPLITS = {
        "mini_val": "v1.0-mini",
        "mini_train": "v1.0-mini",
        "train": "v1.0-trainval",
        "val": "v1.0-trainval",
        "test": "v1.0-test",
    }

    nuscenes_att_range = {
        0: [0, 1],
        1: [0, 1],
        2: [2, 3, 4],
        3: [2, 3, 4],
        4: [2, 3, 4],
        5: [5, 6, 7],
        6: [5, 6, 7],
        7: [5, 6, 7],
    }

    mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
    std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)

    def __init__(self, config, split):
        data_dir = os.path.join(config.DATASET.ROOT, "nuscenes")
        ann_path = os.path.join(data_dir, "annotations", "{}.json").format(split)
        print(f"self.SPLITS[split]: {self.SPLITS[split]}")

        super(nuScenes, self).__init__(config, split, ann_path, data_dir)

        print("Loaded {} {} samples".format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def _to_float(self, x):
        """
        Convert to float and accurate to 2 decimal places
        """
        return float("{:.2f}".format(x))

    def loadRadarPointCloud(
        self, img, img_info, transMatInput, transMatOutput, isFlipped=False
    ):
        """
        Load the Radar point cloud data

        Args:
            img: the original image
            img_info: the image info
            transMatInput: the transformation matrix from the original image to the input image
            transMatOutput: the transformation matrix from the input image to the output image
            isFlipped: whether the image is flipped

        Returns:
            pc_z: the 2D point cloud data [x, y, d]
            pc_N: the original amount of points in 2D image
            pc_dep: the depth feature map [d, vel_x, vel_z]
            pc_3d: the 3D point cloud data in camera coordinate [x, y, z]
        """
        # Load radar point cloud from files
        assert {"sensor_id", "sample_token"} <= set(
            img_info
        ), "Missing required keys"

        radar_pc = None
        sensor_name = self.SENSOR_NAME[img_info["sensor_id"]]
        sample_token = img_info["sample_token"]

        radarFile = os.path.join(
            self.img_dir,
            "annotations",
            "radar_pc",
            sensor_name,
            f"{sample_token}.bin",
        )
        with open(radarFile, "rb") as f:
            radar_pc = np.array(pickle.load(f))

        if radar_pc is None:
            return None, None, None, None

        # get distance to points
        depth = radar_pc[2, :]
        maxDistance = self.config.DATASET.MAX_PC_DIST

        # filter points by distance
        if maxDistance > 0:
            mask = depth <= maxDistance
            radar_pc = radar_pc[:, mask]
            depth = depth[mask]

        # add z offset to radar points / raise all Radar points in z direction
        if self.config.DATASET.PC_Z_OFFSET != 0:
            radar_pc[1, :] -= self.config.DATASET.PC_Z_OFFSET

        # map points to the image and filter ones outside
        pc_2d, mask = map_pointcloud_to_image(
            radar_pc,
            np.array(img_info["camera_intrinsic"]),
            img_shape=(img_info["width"], img_info["height"]),
        )
        pc_3d = radar_pc[:, mask]

        # sort points by distance
        index = np.argsort(pc_2d[2, :])
        if not self.config.DATASET.PC_REVERSE:
            index = index[::-1]
        pc_2d = pc_2d[:, index]
        pc_3d = pc_3d[:, index]

        # flip points if image is flipped
        if isFlipped:
            pc_2d[0, :] = img.shape[1] - 1 - pc_2d[0, :]
            pc_3d[0, :] *= -1  # flipping the x dimension
            pc_3d[8, :] *= -1  # flipping x velocity (x is right, z is front)

        pc_2d, pc_3d, pc_dep = self.processPointCloud(
            pc_2d, pc_3d, img, transMatInput, transMatOutput, img_info
        )
        pc_N = np.array(pc_2d.shape[1])

        # pad point clouds with zero to avoid size mismatch error in dataloader
        n_points = min(self.config.DATASET.MAX_PC, pc_2d.shape[1])
        pc_z = np.zeros((pc_2d.shape[0], self.config.DATASET.MAX_PC))
        pc_z[:, :n_points] = pc_2d[:, :n_points]
        pc_3dz = np.zeros((pc_3d.shape[0], self.config.DATASET.MAX_PC))
        pc_3dz[:, :n_points] = pc_3d[:, :n_points]

        return pc_z, pc_N, pc_dep, pc_3dz

    def getDepthMap(self, maxDistance: int, isOneHot: bool) -> np.ndarray:
        """
        This function will return the empty depth map of the point cloud

        Args:
            maxDistance(int): the maximum distance of the point cloud
        """
        depChannelSize = maxDistance * 3 if isOneHot else 3
        depthMap = np.zeros(
            (depChannelSize, *self.config.MODEL.OUTPUT_SIZE), np.float32
        )
        return depthMap

    def drawPcHeat(self, depthMap, box, depth, maxDist, isOneHot, pc_3d, *_):
        """
        This function will draw the heat value of the point cloud on depth map
        Add depth, x velocity, and z velocity to depth map

        Args:
            depthMap(np.ndarray): the depth map of the point
            box(np.ndarray): the bounding box of the object (y1, y2, x1, x2)
            depth(float): the depth value of the point
            maxDist(int): the maximum distance of the point cloud
            isOneHot(bool): whether the point cloud is one hot
            pc_3d(np.ndarray): the 3D point cloud data in camera coordinate

        Returns:
            depthMap(np.ndarray): the depth map
        """
        if isOneHot:
            depthLayer = int(depth)
            xVelLayer = depthLayer + maxDist
            zVelLayer = depthLayer + maxDist * 2

            depthMap[depthLayer, box[0] : box[1], box[2] : box[3]] = depth
            depthMap[xVelLayer, box[0] : box[1], box[2] : box[3]] = pc_3d[8]
            depthMap[zVelLayer, box[0] : box[1], box[2] : box[3]] = pc_3d[9]
        else:
            depthMap[0, box[0] : box[1], box[2] : box[3]] = depth
            depthMap[-2, box[0] : box[1], box[2] : box[3]] = pc_3d[8]
            depthMap[-1, box[0] : box[1], box[2] : box[3]] = pc_3d[9]

        return depthMap

    def drawPcPoints(self, depthMap, points, depths, maxDist, isOneHot, pc_3d, *_):
        """
        This function will draw the point cloud on depth map

        Args:
            depthMap(np.ndarray): the depth map of the point
            points(np.ndarray): the list of points in depth map (x, y)
            depths(np.ndarray): the list of depth values of the points
            maxDist(int): the maximum distance of the point cloud
            isOneHot(bool): whether the point cloud is one hot
            pc_3d(np.ndarray): the 3D point cloud data in camera coordinate

        Returns:
            depthMap(np.ndarray): the depth map
        """
        points = points.astype(np.int32)
        if isOneHot:
            depthLayer = depths.astype(np.int32)
            xVelLayer = depthLayer + maxDist
            zVelLayer = depthLayer + maxDist * 2

            depthMap[depthLayer, points[1], points[0]] = depths
            depthMap[xVelLayer, points[1], points[0]] = pc_3d[8]
            depthMap[zVelLayer, points[1], points[0]] = pc_3d[9]
        else:
            depthMap[0, points[1], points[0]] = depths
            depthMap[-2, points[1], points[0]] = pc_3d[8]
            depthMap[-1, points[1], points[0]] = pc_3d[9]

        return depthMap

    def loadLidarPointCloud(self, img_info, isFlipped=False):
        """
        Load the Lidar point cloud data
        In out model, we only use lidar point cloud to be auxiliary data for training depth estimation

        Args:
            img_info: the image info
            isFlipped: whether the image is flipped

        Returns:
            lidar_pc(np.array): the lidar point cloud data on the image plane (3, N) [x, y, d(m)]
        """
        ## Load lidar point cloud from files
        assert {"sensor_id", "sample_token"} <= set(
            img_info
        ), "Missing required keys"

        lidar_pc = None
        sensor_name = self.SENSOR_NAME[img_info["sensor_id"]]
        sample_token = img_info["sample_token"]

        lidarFile = os.path.join(
            self.img_dir,
            "annotations",
            "lidar_pc",
            sensor_name,
            f"{sample_token}.bin",
        )
        with open(lidarFile, "rb") as f:
            lidar_pc = np.array(pickle.load(f))  # (3, N) [x, y, d]

        ## Process the lidar point cloud data
        # Convert pc coordinate to output image coordinate
        lidar_pc[:2] /= np.array([[img_info["width"]], [img_info["height"]]])
        outputHeight, outputWidth = self.config.MODEL.OUTPUT_SIZE
        lidar_pc[:2] *= np.array([[outputWidth], [outputHeight]])

        # Filter points by distance
        maxDistance = self.config.DATASET.MAX_PC_DIST
        mask = lidar_pc[2] <= maxDistance
        lidar_pc = lidar_pc[:, mask]

        # flip points if image is flipped
        if isFlipped:
            lidar_pc[0] = outputWidth - 1 - lidar_pc[0]

        # Limit the number of points
        fixedLidarPc = np.zeros((3, 4000), dtype=np.float32)
        fixedLidarPc[:, : min(4000, lidar_pc.shape[1])] = lidar_pc[:, :4000]

        return fixedLidarPc

    def initReturn(self, item, target):
        """
        Initialize the return dictionary

        Args:
            item (dict): The dictionary to be returned.
            target (dict): The dictionary containing the empty annotations.
                |- ...
                |- nuscenes_att (np.array): [max_objs, 8]
                |- velocity (np.array): [max_objs, 3]

        Returns:
            None
        """
        super(nuScenes, self).initReturn(item, target)

        if self.config.DATASET.RADAR_PC:
            item["pc_hm"] = np.zeros(
                (
                    3,
                    self.config.MODEL.OUTPUT_SIZE[0],
                    self.config.MODEL.OUTPUT_SIZE[1],
                ),
                np.float32,
            )

        regression_head_dims = {
            "nuscenes_att": 8,
            "velocity": 3,
        }

        for head in regression_head_dims:
            if head in self.config.heads:
                item[head] = np.zeros(
                    (self.max_objs, regression_head_dims[head]), dtype=np.float32
                )
                target[head] = np.zeros(
                    (self.max_objs, regression_head_dims[head]), dtype=np.float32
                )

        if "nuscenes_att" in self.config.heads:
            item["nuscenes_att_mask"] = np.zeros(
                (self.max_objs, regression_head_dims["nuscenes_att"]), dtype=np.float32
            )

    def convert_coco_format(self, all_bboxes):
        """
        Convert 2D bbox to coco format
        """
        detections = []
        for image_id in all_bboxes:
            if type(all_bboxes[image_id]) != dict:
                for j in range(len(all_bboxes[image_id])):
                    item = all_bboxes[image_id][j]
                    category_id = item["class"]
                    bbox = item["bbox"]
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    bbox_out = list(map(self._to_float, bbox[0:4]))
                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(item["score"])),
                    }
                    detections.append(detection)
        return detections

    @staticmethod
    def getEvalFormatItem(args):
        item = args["item"]
        image_info = args["image_info"]
        trans_matrix = args["trans_matrix"]
        velocity_mat = args["velocity_mat"]
        sensor_id = args["sensor_id"]
        sample_token = args["sample_token"]

        class_name = nuScenes.class_name[int(item["class"] - 1)]
        score = float(item["score"])
        size = item["dimension"][[1, 2, 0]].tolist()
        location = item["location"].tolist()
        location[1] -= size[2]
        translation = np.dot(trans_matrix, np.array([*location, 1], np.float32))

        if not ("rotation" in item):
            rot_cam = Quaternion(axis=[0, 1, 0], angle=item["yaw"])
            location = item["location"].numpy()
            box = Box(location, size, rot_cam, name="2", token="1")
            box.translate(np.array([0, -box.wlh[2] / 2, 0]))
            box.rotate(Quaternion(image_info["cs_record_rot"]))
            box.translate(np.array(image_info["cs_record_trans"]))
            box.rotate(Quaternion(image_info["pose_record_rot"]))
            box.translate(np.array(image_info["pose_record_trans"]))
            rotation = box.orientation.q.tolist()
        else:
            rotation = item["rotation"]

        att = ""
        if "nuscenes_att" in item:
            nuscenes_att = item["nuscenes_att"].numpy()
            if class_name in nuScenes.cycles:
                att = nuScenes.id_to_attribute[np.argmax(nuscenes_att[0:2]) + 1]
            elif class_name in nuScenes.pedestrians:
                att = nuScenes.id_to_attribute[np.argmax(nuscenes_att[2:5]) + 3]
            elif class_name in nuScenes.vehicles:
                att = nuScenes.id_to_attribute[np.argmax(nuscenes_att[5:8]) + 6]

        if "velocity" in item and len(item["velocity"]) == 2:
            velocity = item["velocity"]
        else:
            velocity = item["velocity"] if "velocity" in item else np.zeros(3)
            velocity = np.dot(
                velocity_mat,
                np.array([*velocity[:3].tolist(), 0], np.float32),
            )
            velocity = velocity[:2].tolist()

        result = {
            "sample_token": sample_token,
            "translation": translation[:3].tolist(),
            "size": size,
            "rotation": rotation,
            "velocity": velocity,
            "detection_name": class_name,
            "attribute_name": (
                att if not ("attribute_name" in item) else item["attribute_name"]
            ),
            "detection_score": score,
            "tracking_name": class_name,
            "tracking_score": score,
            "tracking_id": 1,
            "sensor_id": sensor_id,
            "det_id": -1,
        }
        return result

    def convert_eval_format(self, results):
        """
        Convert the results to the format of nuScenes evaluation

        Args:
            results (dict): the results of the model
                imgid (int): the image id
                 |-class
                 |-score
                 |-dimension
                 |-location
                 |-yaw
                 |-bboxes
                 |-nuscenes_att
                 |-velocity

        Returns:
            ret (dict): the results in the format of nuScenes evaluation
        """
        ret = {
            "meta": {
                "use_camera": True,
                "use_lidar": False,
                "use_radar": self.config.DATASET.RADAR_PC,
                "use_map": False,
                "use_external": False,
            },
            "results": {},
        }

        for image_id in self.images:
            if not (image_id in results):
                continue
            image_info = self.coco.loadImgs(ids=[image_id])[0]
            sample_token = image_info["sample_token"]
            trans_matrix = np.array(image_info["trans_matrix"], np.float32)
            velocity_mat = np.array(image_info["velocity_trans_matrix"], np.float32)
            sensor_id = image_info["sensor_id"]

            sample_results = []
            for item in results[image_id]:
                sample_results.append(
                    self.getEvalFormatItem(
                        {
                            "item": item,
                            "image_info": image_info,
                            "trans_matrix": trans_matrix,
                            "velocity_mat": velocity_mat,
                            "sensor_id": sensor_id,
                            "sample_token": sample_token,
                        }
                    )
                )

            if sample_token in ret["results"]:
                ret["results"][sample_token] = (
                    ret["results"][sample_token] + sample_results
                )
            else:
                ret["results"][sample_token] = sample_results

        for sample_token in ret["results"].keys():
            confs = sorted(
                [
                    (-d["detection_score"], ind)
                    for ind, d in enumerate(ret["results"][sample_token])
                ]
            )
            ret["results"][sample_token] = [
                ret["results"][sample_token][ind]
                for _, ind in confs[: min(500, len(confs))]
            ]

        return ret

    @rank_zero_only
    def run_eval(self, results, save_dir, n_plots=10):
        split = self.config.DATASET.VAL_SPLIT
        version = "v1.0-mini" if "mini" in split else "v1.0-trainval"
        json.dump(
            self.convert_eval_format(results),
            open(f"{save_dir}/results_nuscenes_det_{split}.json", "w"),
        )

        if split == "test":
            return save_dir

        # Call from nuscenes lib directly will cause error in multi-processing
        # So we call it by os.system
        output_dir = f"{save_dir}/nuscenes_eval_det_output_{split}/"
        os.system(
            "python "
            + "src/lib/nuScenes_lib/evaluate.py "
            + f"{save_dir}/results_nuscenes_det_{split}.json "
            + f"--output_dir {output_dir} "
            + f"--eval_set {split} "
            + "--dataroot data/nuscenes/ "
            + f"--version {version} "
            + f"--plot_examples {n_plots} "
            + "--render_curves 1 "
            + "--verbose 0 "
        )

        return output_dir

    @rank_zero_only
    @safe_run
    def logValidResult(self, logger, output_dir):
        """
        This function will log result from file to logger and console.

        Args:
            logger: logger object
            output_dir: output directory

        Returns:
            None
        """
        evalRanges = ["10", "30", "50", "all"]
        evalRanges_ = ["0-10", "10-30", "30-50", "0-50"]
        for extreme in [False, True]:
            for i, evalRange in enumerate(evalRanges):
                logger.info(f"\nEval range: {evalRanges_[i]}")
                logger.info(f"Extreme scenes: {extreme}")
                with open(
                    os.path.join(
                        output_dir,
                        f"nuscenes_eval_det_output_{self.split}",
                        f"range_{evalRange}{'_extreme' if extreme else ''}",
                        "metrics_summary.json",
                    ),
                    "r",
                ) as f:
                    metrics = json.load(f)
                logger.info(f'AP/overall: {metrics["mean_ap"]*100.0:.2f}%')

                for k, v in metrics["mean_dist_aps"].items():
                    logger.info(f"AP/{k}: {v * 100.0:.2f}%")

                for k, v in metrics["tp_errors"].items():
                    logger.info(f"Scores/{k}: {v}")

                logger.info(f'Scores/NDS: {metrics["nd_score"]}')

    @staticmethod
    def getDemoInstance(args):
        return Demo(args)


class Demo(GenericDemo):
    SENSOR_ID = {
        "RADAR_FRONT": 7,
        "RADAR_FRONT_LEFT": 9,
        "RADAR_FRONT_RIGHT": 10,
        "RADAR_BACK_LEFT": 11,
        "RADAR_BACK_RIGHT": 12,
        "LIDAR_TOP": 8,
        "CAM_FRONT": 1,
        "CAM_FRONT_RIGHT": 2,
        "CAM_BACK_RIGHT": 3,
        "CAM_BACK": 4,
        "CAM_BACK_LEFT": 5,
        "CAM_FRONT_LEFT": 6,
    }
    CATS = [
        "car",
        "truck",
        "bus",
        "trailer",
        "construction_vehicle",
        "pedestrian",
        "motorcycle",
        "bicycle",
        "traffic_cone",
        "barrier",
    ]

    VAL_SCENES = eval(
        """['scene-0003', 'scene-0012', 'scene-0013', 'scene-0014', 'scene-0015', 'scene-0016', 'scene-0017', 'scene-0018',
     'scene-0035', 'scene-0036', 'scene-0038', 'scene-0039', 'scene-0092', 'scene-0093', 'scene-0094', 'scene-0095',
     'scene-0096', 'scene-0097', 'scene-0098', 'scene-0099', 'scene-0100', 'scene-0101', 'scene-0102', 'scene-0103',
     'scene-0104', 'scene-0105', 'scene-0106', 'scene-0107', 'scene-0108', 'scene-0109', 'scene-0110', 'scene-0221',
     'scene-0268', 'scene-0269', 'scene-0270', 'scene-0271', 'scene-0272', 'scene-0273', 'scene-0274', 'scene-0275',
     'scene-0276', 'scene-0277', 'scene-0278', 'scene-0329', 'scene-0330', 'scene-0331', 'scene-0332', 'scene-0344',
     'scene-0345', 'scene-0346', 'scene-0519', 'scene-0520', 'scene-0521', 'scene-0522', 'scene-0523', 'scene-0524',
     'scene-0552', 'scene-0553', 'scene-0554', 'scene-0555', 'scene-0556', 'scene-0557', 'scene-0558', 'scene-0559',
     'scene-0560', 'scene-0561', 'scene-0562', 'scene-0563', 'scene-0564', 'scene-0565', 'scene-0625', 'scene-0626',
     'scene-0627', 'scene-0629', 'scene-0630', 'scene-0632', 'scene-0633', 'scene-0634', 'scene-0635', 'scene-0636',
     'scene-0637', 'scene-0638', 'scene-0770', 'scene-0771', 'scene-0775', 'scene-0777', 'scene-0778', 'scene-0780',
     'scene-0781', 'scene-0782', 'scene-0783', 'scene-0784', 'scene-0794', 'scene-0795', 'scene-0796', 'scene-0797',
     'scene-0798', 'scene-0799', 'scene-0800', 'scene-0802', 'scene-0904', 'scene-0905', 'scene-0906', 'scene-0907',
     'scene-0908', 'scene-0909', 'scene-0910', 'scene-0911', 'scene-0912', 'scene-0913', 'scene-0914', 'scene-0915',
     'scene-0916', 'scene-0917', 'scene-0919', 'scene-0920', 'scene-0921', 'scene-0922', 'scene-0923', 'scene-0924',
     'scene-0925', 'scene-0926', 'scene-0927', 'scene-0928', 'scene-0929', 'scene-0930', 'scene-0931', 'scene-0962',
     'scene-0963', 'scene-0966', 'scene-0967', 'scene-0968', 'scene-0969', 'scene-0971', 'scene-0972', 'scene-1059',
     'scene-1060', 'scene-1061', 'scene-1062', 'scene-1063', 'scene-1064', 'scene-1065', 'scene-1066', 'scene-1067',
     'scene-1068', 'scene-1069', 'scene-1070', 'scene-1071', 'scene-1072', 'scene-1073']"""
    )

    def __init__(self, args):
        super(Demo, self).__init__(args)

        self.nusc = NuScenes(
            version=nuScenes.SPLITS[args.split],
            dataroot="./data/nuscenes",
            verbose=True,
        )
        # Load nuScenes ground truth
        self.cfg = config_factory("detection_cvpr_2019")
        nuscGtBoxes = load_gt(self.nusc, args.split, DetectionBox)
        nuscGtBoxes = add_center_dist(self.nusc, nuscGtBoxes)
        self.nuscGtBoxes = filter_eval_boxes(
            self.nusc, nuscGtBoxes, self.cfg.class_range
        )
        self.fps = 2
        self.CAT_IDS = {v: i + 1 for i, v in enumerate(self.CATS)}
        self.single = args.single

    def run(self):
        # Optional side tracker + rule-based classifier
        use_side_cls = getattr(self.args, 'state_classify', False)
        dump_states = getattr(self.args, 'dump_states', False)
        output_dir = getattr(self.args, 'output_dir', '.')
        if use_side_cls:
            from lib.utils.simple_tracker import SimpleTracker
            self._side_trackers = None  # lazy init list of per-camera trackers
            side_model_path = getattr(self.args, 'side_model', '')
            if side_model_path:
                import numpy as _np
                data = _np.load(side_model_path, allow_pickle=True)
                def _obj(x):
                    # Accept 0-d object array, 1-d length-1 array, or direct python object
                    if isinstance(x, _np.ndarray) and x.dtype == object:
                        if x.shape == ():  # 0-d
                            return x.item()
                        if x.size == 1:
                            return x.reshape(()).item()
                        # fallback: first element
                        return x.flat[0]
                    return x
                self._side_clf = _obj(data['model'])
                self._side_state_to_id = _obj(data['STATE_TO_ID'])
                self._side_id_to_state = {v: k for k, v in self._side_state_to_id.items()}
                self._side_feature_dim = int(_obj(data.get('feature_dim', 0)) or 0)
                def _extract_features(track):
                    import numpy as np
                    # 与训练阶段同步：用位置差分推导速度
                    hist = track.history[-10:]
                    xs, zs, ts, yaws = [], [], [], []
                    for t, loc, yaw, vel, score in hist:
                        xs.append(float(loc[0])); zs.append(float(loc[2])); ts.append(float(t)); yaws.append(float(yaw))
                    vfs, vls, speeds = [], [], []
                    for i in range(1, len(xs)):
                        dt = 0.5
                        dz = zs[i]-zs[i-1]; dx = xs[i]-xs[i-1]
                        vf = dz/dt; vl = dx/dt; vfs.append(vf); vls.append(vl); speeds.append((vf*vf+vl*vl)**0.5)
                    if len(vfs)==0:
                        vfs=[0.0]; vls=[0.0]; speeds=[0.0]
                    yaw_rate=0.0
                    if len(yaws)>=2:
                        y_arr=np.unwrap(np.array(yaws)); dy=np.diff(y_arr)
                        yaw_rate=float(np.mean(dy)/0.5)
                    if self._side_feat_minimal:
                        # minimal 特征顺序: vf_mean, vf_std, vl_mean, vl_std, speed_mean, speed_std, yaw_rate, ratio_lat_forward, dir_changes, mean_abs_vl
                        vf_arr=np.array(vfs); vl_arr=np.array(vls); sp_arr=np.array(speeds)
                        def ms(a):
                            return float(a.mean()), float(a.std())
                        vf_m, vf_s = ms(vf_arr)
                        vl_m, vl_s = ms(vl_arr)
                        sp_m, sp_s = ms(sp_arr)
                        mean_abs_vl=float(np.mean(np.abs(vl_arr)))
                        mean_abs_vf=float(np.mean(np.abs(vf_arr)))
                        ratio_lat=float(mean_abs_vl/(mean_abs_vf+1e-6))
                        dir_changes=0
                        for i in range(1,len(vls)):
                            if vls[i-1]*vls[i] < 0 and abs(vls[i]-vls[i-1])>0.1:
                                dir_changes+=1
                        feat=[vf_m,vf_s,vl_m,vl_s,sp_m,sp_s,yaw_rate,ratio_lat,float(dir_changes),mean_abs_vl]
                    else:
                        def agg(x):
                            if len(x)==0: return [0,0,0,0]
                            a=np.array(x); return [float(np.mean(a)), float(np.std(a)), float(np.max(a)), float(np.min(a))]
                        feat=[]; feat+=agg(vfs); feat+=agg(vls); feat+=agg(speeds); feat+=[yaw_rate]
                        # 加速度特征
                        acc_f=[]; acc_l=[]
                        for i in range(1,len(vfs)):
                            acc_f.append( (vfs[i]-vfs[i-1])/0.5 )
                            acc_l.append( (vls[i]-vls[i-1])/0.5 )
                        def agg2(x):
                            if len(x)==0: return [0,0,0,0]
                            a=np.array(x); return [float(np.mean(a)), float(np.std(a)), float(np.max(a)), float(np.min(a))]
                        feat+=agg2(acc_f); feat+=agg2(acc_l)
                        dir_changes=0
                        for i in range(1,len(vls)):
                            if vls[i-1]*vls[i]<0 and abs(vls[i]-vls[i-1])>0.1:
                                dir_changes+=1
                        mean_abs_vf=float(np.mean(np.abs(vfs)))
                        mean_abs_vl=float(np.mean(np.abs(vls)))
                        ratio_lat_forward=float(mean_abs_vl/(mean_abs_vf+1e-6))
                        feat.extend([float(dir_changes), mean_abs_vf, mean_abs_vl, ratio_lat_forward])
                        max_vl=float(max(np.abs(vls))) if vls else 0.0
                        max_yaw_rate=abs(yaw_rate)
                        feat.extend([max_vl, max_yaw_rate])
                        # Radar聚合 (minimal 模式已忽略)
                        try:
                            rstats=getattr(track,'radar_stats',[]); rsel=rstats[-20:]
                            def gather(key):
                                return [rs.get(key) for rs in rsel if rs and rs.get('count',0)>0 and key in rs]
                            radar_keys=[
                                'count','vx_mean','vz_mean','vr_mean','vr_std','rcs_mean','rcs_std',
                                'dyn_moving','dyn_stationary','dyn_oncoming','dyn_cross','dyn_stopped',
                                'vx_rms_mean','vy_rms_mean','valid_ratio',
                            ]
                            def agg1(x):
                                if len(x)==0: return [0.0,0.0,0.0,0.0]
                                a=np.array(x,dtype=np.float32); return [float(np.mean(a)),float(np.std(a)),float(np.max(a)),float(np.min(a))]
                            for k in radar_keys:
                                feat.extend(agg1(gather(k)))
                        except Exception:
                            pass
                    arr = np.array(feat,dtype=np.float32)
                    if getattr(self,'_side_feature_dim',0) and arr.size!=self._side_feature_dim:
                        if arr.size<self._side_feature_dim:
                            arr=np.pad(arr,(0,self._side_feature_dim-arr.size),constant_values=0.0)
                        else:
                            arr=arr[:self._side_feature_dim]
                    return arr.reshape(1,-1)
                # 读取标准化统计
                self._side_feat_mean = data.get('feature_mean') if 'feature_mean' in data else None
                self._side_feat_std = data.get('feature_std') if 'feature_std' in data else None
                self._side_feat_standardized = bool(data.get('standardized')) if 'standardized' in data else False
                self._side_feat_minimal = bool(data.get('minimal')) if 'minimal' in data else False
                self._side_feat = _extract_features
                self._side_rule = None
            else:
                from lib.utils.state_classifier import RuleBasedStateClassifier
                self._side_rule = RuleBasedStateClassifier()
                self._side_feat = None
            self._side_states_export = { 'frames': [] }
            # Attach one-pass callback to detector for all cameras
            gate = getattr(self.args, 'side_gate', 2.5)
            def _side_cb(predictBoxesBatch, metas, img_infos):
                import numpy as _np, time as _time
                now = _time.time()
                vehicle_ids = {1,2,3,4,5}  # car, truck, bus, trailer, construction_vehicle
                dump_feat_path = getattr(self.args, 'side_dump_feat', '')
                if dump_feat_path and not hasattr(self, '_side_feat_file'):
                    # 延迟打开，写表头
                    import os as _os, csv as _csv
                    _os.makedirs(_os.path.dirname(dump_feat_path) or '.', exist_ok=True)
                    self._side_feat_file = open(dump_feat_path, 'w', newline='')
                    self._side_feat_csv = _csv.writer(self._side_feat_file)
                    # 构建表头 (feat_dim 可能还未知，先写通用前缀，后续第一帧补充)
                    self._side_feat_header_written = False
                # 之前版本：为避免相机坐标差异导致特征分布漂移，仅在前视相机运行分类。
                # 现在按需求改为：所有相机都运行侧向状态分类。假定 track.location 为全局或统一坐标，可跨相机比较。
                # 若将来发现多相机分布再次漂移，可再加开关恢复单相机模式。
                if self._side_trackers is None:
                    self._side_trackers = [SimpleTracker() for _ in range(len(predictBoxesBatch))]
                    self._side_dbg_counter = 0
                for cam_index, boxes in enumerate(predictBoxesBatch):
                    # 所有相机均执行分类
                    is_single = (len(predictBoxesBatch) == 1)
                    is_front = (cam_index == 1)
                    dets = []
                    # If radar features desired we attempt to aggregate per box using already passed radar pcs (if any in metas) - fallback skip
                    radar_points = None
                    try:
                        # metas may contain radar pc list aligned with cameras under key 'radarPcs'
                        radar_pcs_full = metas.get('radarPcs') if isinstance(metas, dict) else None
                        if radar_pcs_full is not None and cam_index < len(radar_pcs_full):
                            radar_points = radar_pcs_full[cam_index]
                    except Exception:
                        radar_points = None
                    for p in boxes:
                        cls_id = int(p['class'])
                        if cls_id not in vehicle_ids:
                            continue  # skip non-vehicle objects entirely
                        loc_np = p['location'].detach().cpu().numpy() if hasattr(p['location'], 'detach') else _np.array(p['location'])
                        radar_feat = None
                        if radar_points is not None and isinstance(radar_points, _np.ndarray) and radar_points.shape[1] > 0:
                            # radar_points expected shape (18, N)
                            pts = radar_points
                            dists = _np.linalg.norm(_np.stack([pts[0]-loc_np[0], pts[2]-loc_np[2]], axis=1), axis=1)
                            m = dists < 2.0
                            sel = pts[:, m]
                            if sel.shape[1] > 0:
                                vx = sel[8]; vz = sel[9]
                                vr = _np.sqrt(vx**2 + vz**2)
                                rcs = sel[5]
                                # 质量过滤：is_quality_valid (idx 10)
                                valid_mask = (sel[10] == 1)
                                valid_sel = sel[:, valid_mask] if valid_mask.sum() > 0 else sel
                                n = valid_sel.shape[1] if valid_sel.shape[1] > 0 else 1
                                # dyn_prop (idx 3): 0=moving, 1=stationary, 2=oncoming,
                                # 5=crossing_stationary, 6=crossing_moving, 7=stopped
                                dyn = valid_sel[3] if valid_sel.shape[1] > 0 else _np.array([], dtype=_np.int32)
                                dyn_moving = float(_np.sum(dyn == 0)) / n
                                dyn_stationary = float(_np.sum(dyn == 1)) / n
                                dyn_oncoming = float(_np.sum(dyn == 2)) / n
                                dyn_cross = float(_np.sum((dyn == 5) | (dyn == 6))) / n
                                dyn_stopped = float(_np.sum(dyn == 7)) / n
                                radar_feat = {
                                    'count': int(sel.shape[1]),
                                    'vx_mean': float(_np.mean(vx)),
                                    'vz_mean': float(_np.mean(vz)),
                                    'vr_mean': float(_np.mean(vr)),
                                    'vr_std': float(_np.std(vr)),
                                    'rcs_mean': float(_np.mean(rcs)),
                                    'rcs_std': float(_np.std(rcs)),
                                    # dyn_prop 聚合（核心新特征）
                                    'dyn_moving': dyn_moving,
                                    'dyn_stationary': dyn_stationary,
                                    'dyn_oncoming': dyn_oncoming,
                                    'dyn_cross': dyn_cross,
                                    'dyn_stopped': dyn_stopped,
                                    # 质量特征
                                    'vx_rms_mean': float(_np.mean(valid_sel[16])) if valid_sel.shape[1] > 0 else 0.0,
                                    'vy_rms_mean': float(_np.mean(valid_sel[17])) if valid_sel.shape[1] > 0 else 0.0,
                                    'valid_ratio': float(valid_mask.sum()) / max(sel.shape[1], 1),
                                }
                        dets.append({
                            'class': cls_id,
                            'score': float(p['score']),
                            'location': loc_np,
                            'yaw': float(p['yaw'] if not hasattr(p['yaw'], 'detach') else p['yaw'].detach().cpu().numpy()),
                            'velocity': p.get('velocity', _np.zeros(3)),
                            'radar_feat': radar_feat,
                        })
                    tracker = self._side_trackers[cam_index]
                    mapping = tracker.update(dets, now)
                    # classify
                    states = {}
                    debug = getattr(self.args, 'side_debug', False)
                    feat_dim = None
                    for tid, tr in tracker.get_active_tracks().items():
                        if self._side_feat is not None:
                            feat = self._side_feat(tr)
                            if feat_dim is None:
                                feat_dim = feat.shape[1]
                                if dump_feat_path and not self._side_feat_header_written:
                                    header = [f'f{j:02d}' for j in range(feat_dim)] + ['pred','state','prob_max','entropy']
                                    self._side_feat_csv.writerow(header)
                                    self._side_feat_header_written = True
                            # 标准化
                            if self._side_feat_standardized and self._side_feat_mean is not None and self._side_feat_std is not None:
                                try:
                                    feat = (feat - self._side_feat_mean.reshape(1,-1)) / (self._side_feat_std.reshape(1,-1)+1e-6)
                                except Exception:
                                    pass
                            sid = int(self._side_clf.predict(feat)[0])
                            prob = None; pmax=1.0; pentropy=0.0
                            if hasattr(self._side_clf, 'predict_proba'):
                                prob = self._side_clf.predict_proba(feat)[0]
                                try:
                                    pmax=float(_np.max(prob)); pentropy=float(-_np.sum(prob*_np.log(prob+1e-8)))
                                except Exception:
                                    pass
                                # 低置信度尝试规则修正
                                if (pmax < 0.5 or pentropy > 1.2):
                                    if self._side_feat_minimal and feat.shape[1] >= 10:
                                        ratio_lat_forward = float(feat[0,7])
                                        dir_changes = float(feat[0,8])
                                        yaw_rate = float(feat[0,6])
                                        vf_mean = float(feat[0,0])
                                        vl_mean = float(feat[0,2])
                                        speed_mean = float(feat[0,4])
                                    elif feat.shape[1] >= 25:  # 扩展特征布局
                                        dir_changes = float(feat[0,21])
                                        mean_abs_vf = float(feat[0,22])
                                        mean_abs_vl = float(feat[0,23])
                                        ratio_lat_forward = float(feat[0,24])
                                        yaw_rate = float(feat[0,12])
                                        vf_mean = float(feat[0,0])
                                        vl_mean = float(feat[0,5])   # vls agg: mean at idx 5
                                        speed_mean = float(feat[0,8])
                                    else:
                                        ratio_lat_forward=0; dir_changes=0; yaw_rate=0; vf_mean=0; vl_mean=0; speed_mean=0
                                    # 规则: 基于横向速度方向判断左/右行驶
                                    if vl_mean > 0.5:
                                        sid = self._side_state_to_id.get('going_right', sid)
                                    elif vl_mean < -0.5:
                                        sid = self._side_state_to_id.get('going_left', sid)
                                    elif speed_mean < 0.3 and abs(vf_mean) < 0.3:
                                        sid = self._side_state_to_id.get('parking', sid)
                            states[tid] = self._side_id_to_state.get(sid, '')
                            # 可选写CSV
                            if dump_feat_path and prob is not None:
                                row = list(map(lambda x: f"{x:.4f}", feat[0].tolist())) + [sid, states[tid], f"{pmax:.4f}", f"{pentropy:.4f}"]
                                try:
                                    self._side_feat_csv.writerow(row)
                                    if hasattr(self, '_side_feat_file'):
                                        self._side_feat_file.flush()
                                except Exception:
                                    pass
                    if debug:
                        # Print distribution
                        from collections import Counter
                        c = Counter(states.values())
                        cam_name = f'cam{cam_index}'
                        print(f"[side-debug] cam={cam_index}({cam_name}) tracks={len(states)} dist={dict(c)} feat_dim={feat_dim}")
                        # 额外每隔若干帧打印若干 track 的特征与概率，帮助诊断分类塌陷
                        self._side_dbg_counter += 1
                        if self._side_dbg_counter % 10 == 1 and self._side_feat is not None and hasattr(self._side_clf, 'predict_proba'):
                            # 选前 5 个 track
                            sample_items = list(tracker.get_active_tracks().items())[:5]
                            dbg_lines = []
                            for tid, tr in sample_items:
                                feat_arr = self._side_feat(tr)
                                if self._side_feat_standardized and self._side_feat_mean is not None and self._side_feat_std is not None:
                                    try:
                                        feat_arr = (feat_arr - self._side_feat_mean.reshape(1,-1)) / self._side_feat_std.reshape(1,-1)
                                    except Exception:
                                        pass
                                prob = self._side_clf.predict_proba(feat_arr)[0]
                                # 反算出我们构造的聚合（与训练一致：vfs agg(4) + vls(4) + speeds(4) + yaw_rate(1) ...）
                                # 简单取出前 13 个元素含义
                                if feat_arr.shape[1] >= 13:
                                    vf_mean, vf_std, vf_max, vf_min, vl_mean, vl_std, vl_max, vl_min, sp_mean, sp_std, sp_max, sp_min, yaw_rate = feat_arr[0][:13]
                                else:
                                    vf_mean=vl_mean=sp_mean=yaw_rate=0; vf_std=vl_std=sp_std=0; vf_max=vl_max=sp_max=0; vf_min=vl_min=sp_min=0
                                prob_map = {self._side_id_to_state[k]: float(v) for k, v in enumerate(prob)}
                                dbg_lines.append(f"tid={tid} vf_mean={vf_mean:.2f} vl_mean={vl_mean:.2f} yaw_rate={yaw_rate:.3f} probs={prob_map}")
                            for l in dbg_lines:
                                print('[side-debug]   ' + l)
                        # 周期性输出概率分布直方（前若干 track 聚合）：
                        if self._side_dbg_counter % 30 == 1 and hasattr(self._side_clf, 'predict_proba'):
                            try:
                                sample_items = list(tracker.get_active_tracks().items())[:20]
                                probs = []
                                for tid, tr in sample_items:
                                    ftmp = self._side_feat(tr)
                                    if self._side_feat_standardized and self._side_feat_mean is not None and self._side_feat_std is not None:
                                        ftmp = (ftmp - self._side_feat_mean.reshape(1,-1)) / (self._side_feat_std.reshape(1,-1)+1e-6)
                                    p = self._side_clf.predict_proba(ftmp)[0]
                                    probs.append(p)
                                if probs:
                                    probs = _np.stack(probs,0)
                                    avg = probs.mean(0)
                                    mp = {self._side_id_to_state[i]: float(v) for i,v in enumerate(avg)}
                                    print(f"[side-debug] avg_proba (sampled_tracks) {mp}")
                            except Exception:
                                pass
                        if debug and hasattr(self,'_side_radar_usage') and self._side_radar_usage['frames'] % 50 == 1:
                            ru=self._side_radar_usage
                            ratio=ru['radar_nonzero']/(ru['tracks']+1e-6)
                            if ratio < 0.05:
                                print(f"[side-debug] WARNING: Radar feature usage very low ({ratio*100:.1f}%), 可能推理无雷达点导致与训练分布偏移。建议重新训练时去掉 --use-radar 或在 demo 启用 RADAR_PC 配置。")
                    # assign to boxes by nearest center
                    for tid, det in mapping.items():
                        loc = det['location']
                        best_j=-1; best_d=1e9
                        for j, p in enumerate(boxes):  # search only vehicle boxes
                            if int(p['class']) not in vehicle_ids:
                                continue
                            ploc = p['location'] if not hasattr(p['location'], 'detach') else p['location'].detach().cpu().numpy()
                            d = float(_np.linalg.norm([ploc[0]-loc[0], ploc[2]-loc[2]]))
                            if d < best_d:
                                best_d = d; best_j = j
                        if best_j>=0 and best_d <= gate and int(boxes[best_j]['class']) in vehicle_ids:
                            # 先写入侧端分类结果
                            boxes[best_j]['side_state'] = states.get(tid, '')
                            # 若主模型 nuScenes attribute 判定为 parked，则直接覆盖为 parking（跳过侧端判别）
                            if getattr(self.args, 'attr_parking_override', False):
                                try:
                                    att_vec = boxes[best_j].get('nuscenes_att')
                                    if att_vec is not None:
                                        if hasattr(att_vec, 'detach'):
                                            att_vec = att_vec.detach().cpu().numpy()
                                        att_vec = _np.array(att_vec).reshape(-1)
                                        cls_id = int(boxes[best_j].get('class', 0)) - 1
                                        # 车辆类别集合（与 visualize 中一致）
                                        if cls_id in {0,1,2,3,4}:  # car, truck, bus, trailer, construction_vehicle
                                            if att_vec.size >= 8:
                                                # 对应 visualize：车辆使用 att[5:8] -> moving / parked / stopped
                                                sub = att_vec[5:8]
                                                if sub.size == 3 and int(_np.argmax(sub)) == 1:
                                                    boxes[best_j]['side_state'] = 'parking'
                                except Exception:
                                    pass
                    if dump_states:
                        self._side_states_export['frames'].append({
                            'sample_token': getattr(self, 'current_sample_token', ''),
                            'camera_index': cam_index,
                            'states': [{'track_id': int(tid), 'state': st} for tid, st in states.items()]
                        })
            self.detector.side_state_callback = _side_cb

        if self.args.sample is not None:
            sample = self.nusc.get("sample", self.args.sample)
            self.predictFrame(None, sample)
            cv2.waitKey(0)

        else:
            outputSizes = {
                "main": (2400, 896),
                "depth": (600, 224),
                "2d": (2400, 896),
                "bev": (500, 500),
                "pc_hm_out": (600, 224),
                "pc_hm_in": (600, 224),
                "pc_hm_mid": (600, 224),
            }
            if self.single:
                outputSizes = {
                    "main": (800, 448),
                    "depth": (200, 112),
                    "2d": (800, 448),
                    "bev": (500, 500),
                    "pc_hm_out": (200, 112),
                    "pc_hm_in": (200, 112),
                    "pc_hm_mid": (200, 112),
                }
            for i, scene in enumerate(self.nusc.scene[self.args.min : self.args.max]):
                # Define video writer
                if self.args.save:
                    self.initWriter(scene_tag=scene.get('name', f'scene_{i}'), **outputSizes)

                print(f"Processing scene {i} {scene['name']}\n\n", end="\r")

                # Predict
                first_sample_token = scene["first_sample_token"]
                try:
                    sample = self.nusc.get("sample", first_sample_token)
                except:
                    raise ValueError(
                        f"Error: Invalid `first_sample_token` in dataset split {self.args.split}"
                    )
                self.predictFrame(sample)


                j = 0
                while sample["next"] != "":
                    sample = self.nusc.get("sample", sample["next"])
                    self.predictFrame(sample)
                    j += 1

        cv2.destroyAllWindows()
        print()
        for key in self.writer:
            if self.writer[key] is not None:
                self.writer[key].release()
        # Dump side states JSON
        if use_side_cls and dump_states:
            out = os.path.join(output_dir, 'states_side.json')
            with open(out, 'w') as f:
                json.dump(self._side_states_export, f, indent=2)
            print(f"Saved side states to {out}")

    def getImageInfo(self, sample_token):
        """
        Get image information from nuScenes.

        Args:
            sample_token(str): sample token of the image.

        Returns:
            dict: image information.
        """
        # Get transform matrix
        sample_data = self.nusc.get("sample_data", sample_token)
        calib_sensor_data = self.nusc.get(
            "calibrated_sensor", sample_data["calibrated_sensor_token"]
        )
        pose_record = self.nusc.get("ego_pose", sample_data["ego_pose_token"])
        global_from_car = transform_matrix(
            pose_record["translation"],
            Quaternion(pose_record["rotation"]),
            inverse=False,
        )
        car_from_sensor = transform_matrix(
            calib_sensor_data["translation"],
            Quaternion(calib_sensor_data["rotation"]),
            inverse=False,
        )
        transMat = np.dot(global_from_car, car_from_sensor)

        # Get velocity transform matrix
        vel_global_from_car = transform_matrix(
            np.array([0, 0, 0]),
            Quaternion(pose_record["rotation"]),
            inverse=False,
        )
        vel_car_from_sensor = transform_matrix(
            np.array([0, 0, 0]),
            Quaternion(calib_sensor_data["rotation"]),
            inverse=False,
        )
        velTransMat = np.dot(vel_global_from_car, vel_car_from_sensor)

        return {
            "trans_matrix": transMat,
            "velocity_trans_matrix": velTransMat,
            "sensor_id": self.SENSOR_ID[sample_data["channel"]],
            "cs_record_rot": calib_sensor_data["rotation"],
            "cs_record_trans": calib_sensor_data["translation"],
            "pose_record_rot": pose_record["rotation"],
            "pose_record_trans": pose_record["translation"],
        }

    def convertNuscFormat(self, results, useRadar):
        """
        Convert detection results to nuScenes format.

        Args:
            results(dict): {sample_token: [predict_boxes]}.
            useRadar(bool): whether use radar data.
        """
        ret = {
            "meta": {
                "use_camera": True,
                "use_lidar": False,
                "use_radar": useRadar,
                "use_map": False,
                "use_external": False,
            },
            "results": {},
        }

        for sample_token in results:
            for sampleDataToken in results[sample_token]:
                image_info = self.getImageInfo(sampleDataToken)
                trans_matrix = np.array(image_info["trans_matrix"], np.float32)
                velocity_mat = np.array(image_info["velocity_trans_matrix"], np.float32)
                sensor_id = image_info["sensor_id"]

                sample_results = []
                for item in results[sample_token][sampleDataToken]:
                    sample_results.append(
                        nuScenes.getEvalFormatItem(
                            {
                                "item": item,
                                "image_info": image_info,
                                "trans_matrix": trans_matrix,
                                "velocity_mat": velocity_mat,
                                "sensor_id": sensor_id,
                                "sample_token": sample_token,
                            }
                        )
                    )

                if sample_token in ret["results"]:
                    ret["results"][sample_token] = (
                        ret["results"][sample_token] + sample_results
                    )
                else:
                    ret["results"][sample_token] = sample_results

                for sample_token in ret["results"].keys():
                    confs = sorted(
                        [
                            (-d["detection_score"], ind)
                            for ind, d in enumerate(ret["results"][sample_token])
                        ]
                    )
                    ret["results"][sample_token] = [
                        ret["results"][sample_token][ind]
                        for _, ind in confs[: min(500, len(confs))]
                    ]

        return ret

    def loadTargetLabel(self, targetBBoxes):
        """
        Load the target label

        Args:
            targetBBoxes: nuScenes target label

        Returns:
            target: the target label [{class, score, dimension, location, yaw}]
        """
        target = []
        for box in targetBBoxes:
            det_name = category_to_detection_name(box.name)
            if det_name is None:
                continue
            category_id = self.CAT_IDS[det_name]
            v = np.dot(box.rotation_matrix, np.array([1, 0, 0]))
            yaw = -np.arctan2(v[2], v[0])
            box.translate(np.array([0, box.wlh[2] / 2, 0]))

            target.append(
                {
                    "class": category_id,
                    "score": 1,
                    "dimension": [box.wlh[2], box.wlh[0], box.wlh[1]],
                    "location": [box.center[0], box.center[1], box.center[2]],
                    "yaw": yaw,
                }
            )
        return target

    def predictFrame(self, sample):
        if not self.single:
            allCamImages3DBox = np.zeros((896, 2400, 3), dtype=np.uint8)
            allCamImages2DBox = np.zeros((896, 2400, 3), dtype=np.uint8)
            allCamImages = np.zeros((896, 2400, 3), dtype=np.uint8)
            allCamDepths = np.zeros((224, 600), dtype=np.uint8)
            allPcHmOut = np.zeros((224, 600), dtype=np.uint8)
            allPcHmIn = np.zeros((224, 600), dtype=np.uint8)
            allPcHmMid = np.zeros((224, 600), dtype=np.uint8)
        allPredictBoxes = {}

        # ====================== Load data ======================
        modelInputs = {k: [] for k in ["frames", "img_infos", "sampleDataTokens"]}
        modelInputs["radarPcs"] = [] if config.DATASET.RADAR_PC else None
        sensors = ["CAM_FRONT"] if self.single else nuScenes.RADARS_FOR_CAMERA.keys()
        for camera in sensors:
            sampleDataToken = sample["data"][camera]
            sampleData = self.nusc.get("sample_data", sampleDataToken)
            _, targetBBoxes, camera_intrinsic = self.nusc.get_sample_data(
                sampleDataToken, box_vis_level=BoxVisibility.ANY
            )
            calib = np.eye(4, dtype=np.float32)
            calib[:3, :3] = camera_intrinsic
            calib = calib[:3]
            img_info = {
                "width": sampleData["width"],
                "height": sampleData["height"],
                "camera_intrinsic": camera_intrinsic,
                "calib": calib.tolist(),
                "target": self.loadTargetLabel(targetBBoxes),
            }
            modelInputs["img_infos"].append(img_info)
            modelInputs["frames"].append(
                cv2.imread(os.path.join(self.nusc.dataroot, sampleData["filename"]))
            )
            modelInputs["sampleDataTokens"].append(sampleDataToken)

            # Load radar data
            if config.DATASET.RADAR_PC:
                all_radar_pcs = RadarPointCloud(np.zeros((18, 0)))
                for radar_channel in nuScenes.RADARS_FOR_CAMERA[camera]:
                    radar_pcs, _ = RadarPointCloud.from_file_multisweep(
                        self.nusc, sample, radar_channel, camera, 6
                    )
                    all_radar_pcs.points = np.hstack(
                        (all_radar_pcs.points, radar_pcs.points)
                    )
                radar_pc = all_radar_pcs.points
                modelInputs["radarPcs"].append(radar_pc)

        # ====================== Run model ======================
        ret, inferenceTime = self.detector.run(
            modelInputs["frames"], modelInputs["img_infos"], modelInputs["radarPcs"]
        )
        ret["total"] = inferenceTime
        self.updateTimeBar(ret)

        # ====================== Merge results ======================
        if self.args.not_show and not self.args.save:
            return

        if self.single:
            allCamImages3DBox = ret["results3D"][0]
            allCamImages2DBox = ret["results2D"][0]
            allCamImages = ret["images"][0]
            allCamDepths = ret["depthmaps"]["depth"][0]
            allPcHmOut = ret["depthmaps"].get("pc_hm_out", [None])[0]
            allPcHmIn = ret["depthmaps"].get("pc_hm_in", [None])[0]
            allPcHmMid = ret["depthmaps"].get("pc_hm_mid", [None])[0]
            allPredictBoxes[sampleDataToken] = ret["predictBoxes"][0]
            bev = ret["resultsBev"][0]
        else:
            for i, camera in enumerate(nuScenes.RADARS_FOR_CAMERA.keys()):
                currentRow = int(i > 2)
                currentColumn = i % 3
                allCamImages3DBox[
                    currentRow * 448 : (currentRow + 1) * 448,
                    currentColumn * 800 : (currentColumn + 1) * 800,
                    :,
                ] = ret["results3D"][i]
                allCamImages2DBox[
                    currentRow * 448 : (currentRow + 1) * 448,
                    currentColumn * 800 : (currentColumn + 1) * 800,
                    :,
                ] = ret["results2D"][i]
                allCamImages[
                    currentRow * 448 : (currentRow + 1) * 448,
                    currentColumn * 800 : (currentColumn + 1) * 800,
                    :,
                ] = ret["images"][i]
                allDepthmaps = [allCamDepths, allPcHmOut, allPcHmIn, allPcHmMid]
                for j, mapKey in enumerate(
                    ["depth", "pc_hm_out", "pc_hm_in", "pc_hm_mid"]
                ):
                    if (
                        mapKey not in ret["depthmaps"]
                        or ret["depthmaps"][mapKey] is None
                    ):
                        continue
                    allDepthmaps[j][
                        currentRow * 112 : (currentRow + 1) * 112,
                        currentColumn * 200 : (currentColumn + 1) * 200,
                    ] = ret["depthmaps"][mapKey][i]

                sampleDataToken = sample["data"][camera]
                allPredictBoxes[sampleDataToken] = ret["predictBoxes"][i]

                # Visualize BEV by offical nuScenes lib
                nuscResult = self.convertNuscFormat(
                    {sample["token"]: allPredictBoxes}, config.DATASET.RADAR_PC
                )
                nuscPredBoxes = EvalBoxes.deserialize(
                    nuscResult["results"], DetectionBox
                )
                nuscPredBoxes = add_center_dist(self.nusc, nuscPredBoxes)
                if len(nuscPredBoxes.boxes[sample["token"]]):
                    nuscPredBoxes = filter_eval_boxes(
                        self.nusc, nuscPredBoxes, self.cfg.class_range
                    )
                visualize_sample(
                    self.nusc,
                    sample["token"],
                    self.nuscGtBoxes,
                    nuscPredBoxes,
                    eval_range=max(self.cfg.class_range.values()),
                    savepath="./tempNuscBev.png",
                    verbose=False,
                )
                bev = cv2.imread("./tempNuscBev.png")
                bev = cv2.resize(bev, (500, 500))
                if os.path.exists("./tempNuscBev.png"):
                    os.remove("./tempNuscBev.png")

        # Apply colormap
        applyPlasma = lambda x: cv2.applyColorMap(x, cv2.COLORMAP_PLASMA) if x is not None else None
        if allPcHmIn is not None:
            allPcHmIn[allPcHmIn == 255] = 0
            allPcHmIn = applyPlasma(allPcHmIn)
        allCamDepths = applyPlasma(allCamDepths)
        allPcHmOut = applyPlasma(allPcHmOut)
        allPcHmMid = applyPlasma(allPcHmMid)

        self.showAttention(ret["depthmaps"], allCamImages, allPcHmOut)

        outputs = {
            "bev": bev,
            "main": allCamImages3DBox,
            "2d": allCamImages2DBox,
            "depth": allCamDepths,
            "pc_hm_out": allPcHmOut,
            "pc_hm_in": allPcHmIn,
            "pc_hm_mid": allPcHmMid,
        }
        for key in outputs:
            if outputs[key] is not None and not self.args.not_show:
                cv2.imshow(key, outputs[key])
            if key in self.writer and self.writer[key] is not None:
                self.writer[key].write(outputs[key])
        key = cv2.waitKey(1)
        if key == 27:
            print()
            exit()
        elif key == ord("p"):
            cv2.waitKey(0)

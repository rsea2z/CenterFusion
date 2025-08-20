from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import time
import torch
from torch.nn.functional import interpolate

from utils.postProcess import postProcess
from utils.image import getAffineTransform, affineTransform
from model import getModel, loadModel, fusionDecode
from dataset import getDataset
from utils.logger import WandbLogger
from utils.pointcloud import map_pointcloud_to_image
from utils.utils import return_time
from utils import ddd


class Detector(object):
    def __init__(self, config, show=False, pause=False):
        if config.GPUS[0] != -1 and torch.cuda.device_count() >= len(config.GPUS):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print("Creating model...")
        self.model = getModel(config)
        if config.MODEL.LOAD_DIR != "":
            _, self.model, _ = loadModel(self.model, config)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.config = config
        dataset = getDataset(config.DATASET.DATASET)
        self.dataset = dataset(config, config.DATASET.TRAIN_SPLIT)
        self.mean = np.array(self.dataset.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(self.dataset.std, dtype=np.float32).reshape(1, 1, 3)
        self.pause = pause
        self.show = show
        self.visualization = show
        # External annotation hook (callable) to enrich predictBoxes before visualize
        self.side_state_callback = None
        self.hide_attr = False  # 由 demo 通过参数开关控制是否显示属性文本

    @return_time
    def run(self, imgInput, img_info=None, radar_pc=None):
        # Load data
        (imageOrigin, img_info, radar_pc, pre_processed), load_time = self.loadData(
            imgInput, img_info, radar_pc
        )

        # Pre-process
        scale_start_time = time.time()
        if not pre_processed:
            # not prefetch testing
            (images, pc_dep, metas, calibs), pre_process_time = self.pre_process(
                imageOrigin, img_info, radar_pc
            )
        else:
            # prefetch testing
            raise NotImplementedError

        # Network inference
        (outputs, detects, decode_time), forward_time = self.process(
            images, calibs=calibs, pc_dep=pc_dep
        )

        # Post-process
        (detects, depthmaps), post_process_time = self.post_process(
            detects, outputs, metas, calibs
        )

        # Merge outputs
        batchSize = detects["scores"].shape[0]
        predictBoxes, merge_time = self.merge_outputs(detects, batchSize)

        # Invoke external hook to enrich predictBoxes (add custom fields like side_state)
        if callable(self.side_state_callback):
            try:
                self.side_state_callback(predictBoxes, metas, img_info)
            except Exception as e:
                print(f"[Detector] side_state_callback error: {e}")

        # Show results
        (results3D, results2D, resultsBev), show_results_time = self.visualize(
            imageOrigin, img_info, predictBoxes, metas, batchSize
        )

        imgOrigin = [
            cv2.resize(img, self.config.MODEL.INPUT_SIZE[::-1]) for img in imageOrigin
        ]

        if self.config.DEBUG > 0:
            testImg = imgOrigin[0].copy()

            # Get Car heatmap
            testHeatmap = outputs[0]["heatmap"].detach().cpu().numpy()[0]
            testHeatmap = testHeatmap.max(axis=0)
            testHeatmap = (testHeatmap * 255).astype(np.uint8)
            testHeatmap = cv2.resize(testHeatmap, (testImg.shape[1], testImg.shape[0]))
            testHeatmap = cv2.applyColorMap(testHeatmap, cv2.COLORMAP_PLASMA)

            # Overlay heatmap on image
            testImg = cv2.addWeighted(testImg, 0.5, testHeatmap, 0.5, 0)
            testImg = cv2.copyMakeBorder(
                testImg, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=0
            )

            # Draw amodal offset
            for i in range(len(detects["centers"][0])):
                if detects["scores"][0][i] < self.config.CONF_THRESH:
                    continue

                # Draw amodal center
                amodalCenter = affineTransform(
                    detects["centers"][0][i].view(-1, 2).numpy(),
                    metas[0]["transMatInput"],
                )
                amodalCenter = amodalCenter.astype(np.int64) + 200
                testImg = cv2.circle(
                    testImg, tuple(amodalCenter[0]), 5, (255, 0, 0), -1
                )

                # Draw heat center
                center = detects["centers"][0][i].numpy() / (
                    [metas[0]["width"], metas[0]["height"]]
                )
                center *= [metas[0]["outputWidth"], metas[0]["outputHeight"]]
                center -= detects["amodal_offset"][0][i].numpy()
                center /= (metas[0]["outputWidth"], metas[0]["outputHeight"])
                center *= (metas[0]["inputWidth"], metas[0]["inputHeight"])
                center = center.astype(np.int64) + 200
                testImg = cv2.circle(testImg, center.tolist(), 3, (0, 255, 0), -1)

                # Draw amodal offset
                cv2.line(
                    testImg, center.tolist(), tuple(amodalCenter[0]), (0, 0, 255), 2
                )

            cv2.imshow("heatmap", testImg)
            key = cv2.waitKey(0)
            if key == 27:
                cv2.destroyAllWindows()
                exit()

        ret = {
            "results3D": results3D,
            "results2D": results2D,
            "resultsBev": resultsBev,
            "images": imgOrigin,
            "detects": detects,
            "predictBoxes": predictBoxes,
            "depthmaps": depthmaps,
            "load": load_time,
            "preprocess": pre_process_time,
            "net": forward_time,
            "decode": decode_time,
            "postprocess": post_process_time,
            "merge": merge_time,
            "display": show_results_time,
        }

        return ret

    @return_time
    def loadData(self, imgInput, img_info, radar_pc):
        """
        Prepare the input data for the network
        Convert args to multiple input format if it is single input

        Args:
            imgInput(ndarray or str or list of type above): image input
            img_info(dict): image information
            radar_pc(ndarray): radar point cloud
        """
        pre_processed = False

        if isinstance(imgInput, np.ndarray):
            imgInput = [imgInput]
        elif isinstance(imgInput, str):
            imgInput = [cv2.imread(imgInput)]
        elif isinstance(imgInput, dict):
            raise NotImplementedError
        elif isinstance(imgInput, list) and isinstance(imgInput[0], np.ndarray):
            pass
        elif isinstance(imgInput, list) and isinstance(imgInput[0], str):
            imgInput = [cv2.imread(img) for img in imgInput]
        else:
            assert False, f"unknown type: {type(imgInput)}"

        if not isinstance(img_info, list):
            img_info = [img_info]
            radar_pc = [radar_pc] if radar_pc is not None else None

        return imgInput, img_info, radar_pc, pre_processed

    @return_time
    def pre_process(self, imageOrigins, img_infos, radar_pcs):
        """
        Pre-process the image to match the training phase
        For multiple input, all inputs should have the same size

        Args:
            image(list of ndarray): image in original size
            img_infos(list of dict): image information
            radar_pcs(list of ndarray): radar point cloud

        Returns:
            images(Tensor): pre-processed image (B, C, H, W)
            pc_deps(Tensor): pre-processed radar point cloud (B, C, H, W)
            metas(list of dict): updated meta data
            calibs(Tensor): calibration matrix (B, 3, 4)
        """
        # Get transformation matrix
        height, width = imageOrigins[0].shape[0:2]
        inputHeight, inputWidth = self.config.MODEL.INPUT_SIZE
        center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
        scale = max(height, width) * 1.0

        transMatInput = getAffineTransform(
            center,
            scale,
            0,
            [inputWidth, inputHeight],
        )
        transMatOutput = getAffineTransform(
            center,
            scale,
            0,
            [self.config.MODEL.OUTPUT_SIZE[1], self.config.MODEL.OUTPUT_SIZE[0]],
        )

        # Affine transformation
        images = np.concatenate(imageOrigins, axis=-1)
        images = cv2.warpAffine(
            images, transMatInput, (inputWidth, inputHeight), flags=cv2.INTER_LINEAR
        )
        mean = np.concatenate([self.mean] * len(imageOrigins), axis=-1)
        std = np.concatenate([self.std] * len(imageOrigins), axis=-1)
        images = ((images / 255.0 - mean) / std).astype(np.float32)
        images = images.transpose(2, 0, 1).reshape(-1, 3, inputHeight, inputWidth)
        images = torch.from_numpy(images)

        # Process radar point cloud and meta data
        pc_deps = None
        metas = []
        calibs = []
        for i, img_info in enumerate(img_infos):
            pc_3d = None  # default when no radar provided for this view
            if img_info is not None and "calib" in img_info:
                calib = np.array(img_info["calib"], dtype=np.float32)

            else:
                calib = np.array(
                    [
                        [self.dataset.focalLength, 0, center[0], 0],
                        [0, self.dataset.focalLength, center[1], 0],
                        [0, 0, 1, 0],
                    ]
                )

            calibs.append(calib)

            pc_dep = None
            if self.config.DATASET.RADAR_PC and radar_pcs is not None:
                radar_pc = radar_pcs[i]
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
                pc_2d = pc_2d[:, index]
                pc_3d = pc_3d[:, index]

                pc_2d, pc_3d, pc_dep = self.dataset.processPointCloud(
                    pc_2d,
                    pc_3d,
                    imageOrigins[i],
                    transMatInput,
                    transMatOutput,
                    img_info,
                )
                pc_dep = torch.from_numpy(pc_dep).unsqueeze(0)
                if pc_deps is None:
                    pc_deps = pc_dep
                else:
                    pc_deps = torch.cat((pc_deps, pc_dep), dim=0)
            elif self.config.DATASET.RADAR_PC and radar_pcs is None:
                # create zero placeholder pc_dep for fusion pipeline
                out_h, out_w = self.config.MODEL.OUTPUT_SIZE
                n_chan = 1 if self.config.DATASET.ONE_HOT_PC else 1  # at least depth slice
                # Use 3 channels to be compatible with dataset-generated maps (depth, vx, vz)
                n_chan = 3 if not self.config.DATASET.ONE_HOT_PC else int(self.config.DATASET.MAX_PC_DIST) * 3
                z = torch.zeros((1, n_chan, out_h, out_w), dtype=torch.float32, device=self.device)
                if pc_deps is None:
                    pc_deps = z
                else:
                    pc_deps = torch.cat((pc_deps, z), dim=0)

            meta = {
                "calib": calib,
                "center": center,
                "scale": scale,
                "height": height,
                "width": width,
                "outputHeight": self.config.MODEL.OUTPUT_SIZE[0],
                "outputWidth": self.config.MODEL.OUTPUT_SIZE[1],
                "inputHeight": inputHeight,
                "inputWidth": inputWidth,
                "transMatInput": transMatInput,
                "transMatOutput": transMatOutput,
            }
            if self.config.DATASET.RADAR_PC and pc_3d is not None:
                meta["pc_3d"] = pc_3d  # For visualization
            metas.append(meta)

        images = images.to(self.device)
        if pc_deps is not None:
            pc_deps = pc_deps.to(self.device)
        calibs = np.stack(calibs, axis=0)
        calibs = torch.from_numpy(calibs).to(self.device)

        return images, pc_deps, metas, calibs

    @return_time
    def post_process(self, detects, outputs, metas, calibs):
        """
        Post-process the network output

        Args:
            detects(dict): network output (decoded)
            outputs(list of dict): network output
            metas(list of dict): meta data
            calibs(Tensor): calibration matrix (B, 3, 4)

        Returns:
            detects(dict): decoded output
            depthmap(ndarray): depthmap
            pc_hm_out(ndarray): radar heatmap
            pc_hm_in(ndarray): radar heatmap
        """
        detects = postProcess(
            detects,
            metas[0]["center"],
            metas[0]["scale"],
            *self.config.MODEL.OUTPUT_SIZE,
            calibs,
        )
        for k in detects:
            detects[k] = detects[k].detach().cpu()

        # Get depthmaps
        depthmaps = {}
        depthmaps["depth"] = [
            (
                output["depthMap"]
                if "depthMap" in output
                else output["depth2"] if "depth2" in output else output["depth"]
            )
            for output in outputs
        ]
        depthmaps["depth"] = [
            interpolate(d, (metas[0]["outputHeight"], metas[0]["outputWidth"]))
            for d in depthmaps["depth"]
        ]

        for key in ["pc_hm_out", "pc_hm_in", "pc_hm_mid"]:
            if key in outputs[0]:
                depthmaps[key] = [output[key] for output in outputs]

        for attHead in ["depth", "rotation", "velocity", "nuscenes_att"]:
            for attKey in ["AttImg", "AttRadar"]:
                attentionOutput = f"{attHead}{attKey}"
                if attentionOutput in outputs[0]:
                    depthmaps[attentionOutput] = [
                        output[attentionOutput] for output in outputs
                    ]

        if self.config.DATASET.ONE_HOT_PC:
            pc_hm_in = [torch.argmax(pc_hm, dim=1, keepdim=True) for pc_hm in pc_hm_in]

        # Process depthmaps
        for k, depthmap in depthmaps.items():
            if depthmap is None:
                continue
            depthmap = torch.cat(depthmap, dim=1)
            depthmap = depthmap.max(dim=1).values
            depthmap = depthmap.detach().cpu().numpy()
            depthmap[0, 0] = 0
            depthmap[0, :, 0] = 0
            minDepth = depthmap.min(axis=(1, 2), keepdims=True)
            maxDepth = depthmap.max(axis=(1, 2), keepdims=True)
            depthmap = (depthmap - minDepth) / (maxDepth - minDepth)
            depthmaps[k] = (depthmap * 255).astype(np.uint8)

        return (detects, depthmaps)

    @return_time
    @torch.no_grad()
    def process(self, images, calibs, pc_dep=None):
        """
        Inference and decode the network output

        Args:
            images(Tensor): pre-processed image (B, C, H, W)
            calibs(Tensor): calibration matrix (B, 3, 4)
            pc_dep(Tensor): pre-processed radar point cloud (B, C, H, W)

        Returns:
            outputs(list of dict): network output
            detects(dict): decoded output
            decode_time(float): decode time
        """

        @return_time
        def decode(outputs, config):
            return fusionDecode(
                outputs,
                outputSize=config.MODEL.OUTPUT_SIZE,
                K=config.MODEL.K,
                norm2d=config.MODEL.NORM_2D,
            )

        outputs = self.model(images, pc_dep=pc_dep, calib=calibs)
        detects, decode_time = decode(outputs, self.config)

        return outputs, detects, decode_time

    @return_time
    def merge_outputs(self, detects, batchSize):
        """
        Filter out invalid detections and convert the output to a list of dict

        Args:
            detects(dict): network output (processed)
            batchSize(int): batch size

        Returns:
            predictBoxes(list of dict): list of dict containing the detection information
        """
        keep = (detects["scores"] > -1) & torch.all(detects["dimension"] > 0, dim=2)
        predictBoxes = [[] for _ in range(batchSize)]
        for batch in range(batchSize):
            for j in range(len(detects["scores"][batch])):
                if not keep[batch, j]:
                    continue

                predictBoxes[batch].append(
                    {
                        "class": detects["classIds"][batch, j],
                        "score": detects["scores"][batch, j],
                        "dimension": detects["dimension"][batch, j],
                        "location": detects["locations"][batch, j],
                        "yaw": detects["yaws"][batch, j],
                        "bboxes": detects["bboxes"][batch, j],
                        "bboxes3d": detects["bboxes3d"][batch, j],
                    }
                )

                if "nuscenes_att" in detects:
                    predictBoxes[batch][-1].update(
                        {"nuscenes_att": detects["nuscenes_att"][0, j]}
                    )

                if "velocity" in detects:
                    predictBoxes[batch][-1].update(
                        {"velocity": detects["velocity"][0, j]}
                    )
        return predictBoxes

    @return_time
    def visualize(self, imageOrigin, img_info, predictBoxes, metas, batchSize):
        """
        Visualize the network output (3D bounding box overlay on image, bird's eye view and depthmap)

        Args:
            imageOrigin(list of ndarray): original images
            img_info(list of dict): image information
            predictBoxes(list of dict): list of dict containing the detection information
            metas(list of dict): meta data
            batchSize(int): batch size

        Returns:
            results(list of ndarray): list of visualized images
        """
        WandbLogger.conf_thresh = self.config.CONF_THRESH
        results2D, results3D, resultsBev = [], [], []
        if not self.visualization:
            return results3D, results2D, resultsBev

        for batch in range(batchSize):
            inputHeight, inputWidth = self.config.MODEL.INPUT_SIZE
            transMatInput = metas[batch]["transMatInput"]
            WandbLogger.image = cv2.warpAffine(
                imageOrigin[batch],
                transMatInput,
                (inputWidth, inputHeight),
                flags=cv2.INTER_LINEAR,
            )

            # Draw 2D predictions
            result2D = WandbLogger.image.copy()
            transMatOutput = getAffineTransform(
                metas[batch]["center"],
                metas[batch]["scale"],
                0,
                result2D.shape[0:2][::-1],
            )
            for predictBox in predictBoxes[batch]:
                if predictBox["score"] < self.config.CONF_THRESH:
                    continue
                bbox = (
                    affineTransform(
                        predictBox["bboxes"].view(-1, 2).numpy(), transMatOutput
                    )
                    .astype(np.int64)
                    .tolist()
                )
                result2D = cv2.rectangle(
                    result2D,
                    *bbox,
                    (255, np.random.randint(255), 0),
                    1,
                )
                # Overlay speed (m/s) if velocity is available
                try:
                    if "velocity" in predictBox and predictBox["velocity"] is not None:
                        v = predictBox["velocity"]
                        # Support tensor or list/ndarray
                        if hasattr(v, "detach"):
                            v = v.detach().cpu().numpy()
                        v = np.array(v).reshape(-1)
                        # Use horizontal-plane (x-z) speed magnitude if available
                        if v.size >= 3:
                            speed = float(np.linalg.norm([v[0], v[2]]))
                        elif v.size >= 2:
                            speed = float(np.linalg.norm(v[:2]))
                        elif v.size >= 1:
                            speed = float(abs(v[0]))
                        else:
                            speed = 0.0
                        # Place text slightly above the top-left corner
                        tl = tuple(bbox[0])
                        org = (int(tl[0]), max(0, int(tl[1]) - 6))
                        cv2.putText(
                            result2D,
                            f"{speed:.1f} m/s",
                            org,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                except Exception:
                    # Robust to any unexpected shape or missing fields
                    pass

                # Overlay nuScenes attribute unless隐藏开关开启
                try:
                    if (not getattr(self, 'hide_attr', False)) and "nuscenes_att" in predictBox and predictBox["nuscenes_att"] is not None:
                        att = predictBox["nuscenes_att"]
                        if hasattr(att, "detach"):
                            att = att.detach().cpu().numpy()
                        att = np.array(att).reshape(-1)

                        # Determine class name
                        cls_id = int(predictBox.get("class", 0)) - 1
                        cls_name = None
                        if hasattr(self.dataset, "class_name") and 0 <= cls_id < len(self.dataset.class_name):
                            cls_name = self.dataset.class_name[cls_id]

                        # Map attribute scores to id according to class groups
                        att_id = 0
                        if cls_name in ["motorcycle", "bicycle"] and att.size >= 2:
                            att_id = 1 + int(np.argmax(att[0:2]))  # 1~2
                        elif cls_name == "pedestrian" and att.size >= 5:
                            att_id = 3 + int(np.argmax(att[2:5]))  # 3~5
                        elif cls_name in [
                            "car",
                            "truck",
                            "bus",
                            "trailer",
                            "construction_vehicle",
                        ] and att.size >= 8:
                            att_id = 6 + int(np.argmax(att[5:8]))  # 6~8
                        else:
                            att_id = 0

                        # id -> attribute name
                        id_to_attribute = {
                            0: "",
                            1: "cycle.with_rider",
                            2: "cycle.without_rider",
                            3: "pedestrian.moving",
                            4: "pedestrian.standing",
                            5: "pedestrian.sitting_lying_down",
                            6: "vehicle.moving",
                            7: "vehicle.parked",
                            8: "vehicle.stopped",
                        }
                        att_name = id_to_attribute.get(att_id, "")

                        # Prefer short label for vehicles
                        short_label = att_name.split(".")[-1] if att_name else ""
                        if short_label:
                            # Place below speed text
                            tl = tuple(bbox[0])
                            org2 = (int(tl[0]), min(result2D.shape[0] - 1, int(tl[1]) + 14))
                            cv2.putText(
                                result2D,
                                short_label,
                                org2,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (50, 220, 50),
                                1,
                                cv2.LINE_AA,
                            )
                except Exception:
                    pass
                # Overlay side behavior state if provided by callback
                try:
                    if "side_state" in predictBox and predictBox["side_state"]:
                        tl = tuple(bbox[0])
                        org3 = (int(tl[0]), min(result2D.shape[0] - 1, int(tl[1]) + 28))
                        cv2.putText(
                            result2D,
                            str(predictBox["side_state"]),
                            org3,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 165, 255),
                            1,
                            cv2.LINE_AA,
                        )
                except Exception:
                    pass
            results2D.append(result2D)

            # Draw ground truth
            maxX, maxY = (500, 500)
            resultBev = np.ones((maxX, maxY, 3), dtype=np.uint8) * 255
            scale = 5
            if (
                img_info[batch] is not None
                and "target" in img_info[batch]
                and img_info[batch]["target"] is not None
            ):
                for predictBox in img_info[batch]["target"]:
                    predictBox["bboxes3d"] = ddd.get3dBox(
                        np.array(predictBox["dimension"]).reshape(1, 1, 3),
                        np.array(predictBox["location"]).reshape(1, 1, 3),
                        np.array(predictBox["yaw"]).reshape(1, 1),
                    ).squeeze()  # (8, 3)

                    bevBox = predictBox["bboxes3d"][:4]
                    bevBoxPixel = bevBox[:, ::2].copy()
                    bevBoxPixel[:, 0] += maxX / (scale * 2)
                    bevBoxPixel = bevBoxPixel * scale
                    bevBoxPixel[:, 1] = maxY - bevBoxPixel[:, 1]
                    bevBoxPixel = bevBoxPixel.astype(np.int64)

                    resultBev = cv2.fillPoly(
                        resultBev,
                        [bevBoxPixel.reshape(-1, 1, 2)],
                        color=(0, 255, 0),
                    )

            # Draw bird's eye view
            radarPoints = metas[batch].pop("pc_3d", np.array([]))
            for point in radarPoints.T:
                pointPixel = point[:3].copy()
                pointPixel[0] += maxX / (scale * 2)  # Shift x to positive
                pointPixel = pointPixel[::2] * scale  # Scale to pixel coordinate
                pointPixel[1] = maxY - pointPixel[1]  # Flip y
                if not (all(pointPixel > 0) and all(pointPixel < (maxX, maxY))):
                    continue
                cv2.circle(
                    resultBev,
                    tuple(pointPixel.astype(np.int64).tolist()),
                    1,
                    (0, 0, 255),
                    -1,
                )
            for predictBox in predictBoxes[batch]:
                if predictBox["score"] < self.config.CONF_THRESH:
                    continue

                bevBox = predictBox["bboxes3d"][:4].numpy()
                bevBoxPixel = bevBox[:, ::2].copy()
                bevBoxPixel[:, 0] += maxX / (scale * 2)
                bevBoxPixel = bevBoxPixel * scale
                bevBoxPixel[:, 1] = maxY - bevBoxPixel[:, 1]
                bevBoxPixel = bevBoxPixel.astype(np.int64)

                resultBev = cv2.polylines(
                    resultBev,
                    [bevBoxPixel.reshape(-1, 1, 2)],
                    isClosed=True,
                    color=(255, 0, 0),
                    thickness=2,
                )

            # Draw ruler
            rulerRange = [(maxY // scale) * i for i in range(1, scale)]
            for r in rulerRange:
                cv2.line(resultBev, (0, r), (10, r), (0, 0, 0), 2)
                cv2.putText(
                    resultBev,
                    f"{r // scale}m",
                    (12, maxY - r),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                )

                # FOV lines
                center_x = maxX // 2
                angle1 = np.radians(35)
                angle2 = np.radians(-35)
                end_x1 = int(center_x + maxY * np.sin(angle1))
                end_y1 = int(maxY - maxY * np.cos(angle1))
                end_x2 = int(center_x + maxY * np.sin(angle2))
                end_y2 = int(maxY - maxY * np.cos(angle2))
                cv2.line(resultBev, (center_x, maxY), (end_x1, end_y1), (0, 0, 0), 2)
                cv2.line(resultBev, (center_x, maxY), (end_x2, end_y2), (0, 0, 0), 2)

                # Legend
                cv2.putText(
                    resultBev,
                    "Ground Truth",
                    (maxX - 200, maxY - 20),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    (0, 255, 0),
                )
                cv2.putText(
                    resultBev,
                    "Predictions",
                    (maxX - 200, maxY - 40),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    (255, 0, 0),
                )

            resultsBev.append(resultBev)

            # Draw 3D predictions
            WandbLogger.drawBox3D(
                {"predictBoxes": predictBoxes[batch], "meta": metas[batch]},
                isTarget=False,
                drawOnTarget=True,
            )
            results3D.append(WandbLogger.predBox3DOverlay)
            if self.show:
                WandbLogger.show()

        return results3D, results2D, resultsBev

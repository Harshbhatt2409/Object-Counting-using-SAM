import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore
import matplotlib.pyplot as plt
import math
from typing import Any, Dict, List, Optional, Tuple
import torch.nn.functional as F
from collections import defaultdict
from skimage.transform import resize
from segment_anything.modeling import Sam
from segment_anything.predictor import SamPredictor
from segment_anything.utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
    points2box
)


def uncrop_boxes_xyxy(boxes: torch.Tensor, crop_box: List[int]) -> torch.Tensor:
    x0, y0, _, _ = crop_box
    offset = torch.tensor([[x0, y0, x0, y0]], device=boxes.device)
    # Check if boxes has a channel dimension
    if len(boxes.shape) == 3:
        offset = offset.unsqueeze(1)
    return boxes + offset

def pre_process_ref_box(ref_box, crop_box, layer_idx):
    if layer_idx == 0:
        return ref_box
    else:
        new_bbox = []
        x0, y0, x1, y1 = crop_box
        for ref in ref_box:
            x0_r, y0_r, x1_r, y1_r = ref
            area = (y1_r - y0_r) * (x1_r - x0_r)
            x_0_new = max(x0, x0_r)
            y_0_new = max(y0, y0_r)
            x_1_new = min(x1, x1_r)
            y_1_new = min(y1, y1_r)
            crop_area = (y_1_new - y_0_new) * (x_1_new - x_0_new)
            if crop_area / area > 0.7:
                new_bbox.append([x_0_new, y_0_new, x_1_new, y_1_new])

        return new_bbox

def show_anns(anns):
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for pi in anns['points']:
        x0, y0 = pi
        # ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
        ax.scatter(x0,y0, color='green', marker='*', s=10, edgecolor='white', linewidth=1.25)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_mask2(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

class SamAutomaticMaskGenerator:
    def __init__(
        self,
        model: Sam,
        points_per_side: Optional[int] = 300,
        points_per_batch: int = 32,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.5,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.3,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.8,
        crop_overlap_ratio: float = 700 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
    ) -> None:
        """
        Using a SAM model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM with a ViT-H backbone.

        Arguments:
          model (Sam): The SAM model to use for mask prediction.
          points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          crops_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.
          crops_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks between different crops.
          crop_overlap_ratio (float): Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.
          crop_n_points_downscale_factor (int): The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
          point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
          output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.
        """

        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils  # type: ignore # noqa: F401

        if min_mask_region_area > 0:
            import cv2  # type: ignore # noqa: F401

        self.predictor = SamPredictor(model)
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode

        self.prototype = defaultdict(list)

    @torch.no_grad()
    def generate(self, image: np.ndarray, ref_bbox) -> List[Dict[str, Any]]:
        """
        Generates masks for the given image.

        Arguments:
          image (np.ndarray): The image to generate masks for, in HWC uint8 format.


        Returns:
           list(dict(str, any)): A list over records for masks. Each record is
             a dict containing the following keys:
               segmentation (dict(str, any) or np.ndarray): The mask. If
                 output_mode='binary_mask', is an array of shape HW. Otherwise,
                 is a dictionary containing the RLE.
               bbox (list(float)): The box around the mask, in XYWH format.
               area (int): The area in pixels of the mask.
               predicted_iou (float): The model's own prediction of the mask's
                 quality. This is filtered by the pred_iou_thresh parameter.
               point_coords (list(list(float))): The point coordinates input
                 to the model to generate this mask.
               stability_score (float): A measure of the mask's quality. This
                 is filtered on using the stability_score_thresh parameter.
               crop_box (list(float)): The crop of the image used to generate
                 the mask, given in XYWH format.
        """

        # Generate masks
        mask_data = self._generate_masks(image, ref_bbox)

        

        # Filter small disconnected regions and holes in masks
        if self.min_mask_region_area > 0:
            mask_data = self.postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )

        # Encode masks
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [coco_encode_rle(rle) for rle in mask_data["rles"]]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # Write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)

        return curr_anns
    
    def _generate_similarity(self, image, ref_prompt):
        img_size = image.shape[:2]
        ref_prompt = np.array(ref_prompt)

        if ref_prompt.shape[1]<4: #point prompt
            ref_points = torch.tensor(ref_prompt, device=self.predictor.device)
            transformed_points = self.predictor.transform.apply_coords_torch(ref_points, img_size)
            in_labels = torch.ones(transformed_points.shape[0], dtype=torch.int, device=self.predictor.device)
            masks, iou_preds, low_res_masks = self.predictor.predict_torch(
                point_coords=transformed_points[:,None,:],
                point_labels=in_labels[:,None],
                boxes=None,
                multimask_output=False
                )
        else:#box prompt
            ref_bbox = torch.tensor(ref_prompt, device=self.predictor.device)
            transformed_boxes = self.predictor.transform.apply_boxes_torch(ref_bbox, img_size)
            masks, iou_preds, low_res_masks = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False
                )
        
        masks_cpu = masks.cpu()
        mask_size = [math.sqrt(np.sum(mask.float().numpy())) for mask in masks_cpu]
        mask_size = np.array(mask_size).min(0)
        
        feat = self.predictor.get_image_embedding().squeeze()
        # print(feat.shape)([256, 64, 64])
        # print(ref_prompt.shape)(3, 4)
        

        # print(low_res_masks.shape)torch.Size([3, 1, 256, 256])

        ref_feat = feat.permute(1, 2, 0)
        # print(ref_feat.shape)torch.Size([64, 64, 256])
        C, h, w = feat.shape
        test_feat = feat / feat.norm(dim=0, keepdim=True)
        test_feat = test_feat.reshape(C, h * w)
        low_res_masks = F.interpolate(low_res_masks, size=ref_feat.shape[0: 2], mode='bilinear', align_corners=False)
        low_res_masks = low_res_masks.flatten(2, 3)
        # print(low_res_masks.shape)torch.Size([3, 1, 4096])
        masks_low_res = (low_res_masks > self.predictor.model.mask_threshold).float()
        topk_idx = torch.topk(low_res_masks, 1)[1]
        masks_low_res.scatter_(2, topk_idx, 1.0)
        # print(low_res_masks.shape)torch.Size([3, 1, 4096])
        target_embedding = []
        sim = []
        for i, ref_mask in enumerate(masks_low_res.cpu()):
            ref_mask = ref_mask.squeeze().reshape(ref_feat.shape[0: 2])
            # print(ref_mask.shape)torch.Size([64, 64])
            # Target feature extraction
            target_feat = ref_feat[ref_mask > 0]
            # print(target_feat.shape)torch.Size([45, 256])

            if target_feat.shape[0]>0:
                target_embedding_ = target_feat.mean(0).unsqueeze(0)
                # print(target_embedding_.shape)torch.Size([1, 256])

                target_feat = target_embedding_ / target_embedding_.norm(dim=-1, keepdim=True)
                # print(target_feat.shape)torch.Size([1, 256])

                target_embedding_ = target_embedding_.unsqueeze(0)
                # print(target_feat.shape)torch.Size([1, 256])

                target_embedding.append(target_embedding_)

                # print(target_feat.shape)torch.Size([1, 256])
                # print(test_feat.shape)torch.Size([256, 4096])

                sim_ = target_feat @ test_feat
                # torch.Size([1, 4096])
                sim_ = sim_.reshape(1, 1, h, w)
                # torch.Size([1,1, 64,64])
                sim_ = F.interpolate(sim_, scale_factor=4, mode="bilinear")
                # print(sim_.shape)
                # torch.Size([1, 1, 256, 256])
                sim_ = self.predictor.model.postprocess_masks(
                                sim_,
                                input_size=self.predictor.input_size,
                                original_size=self.predictor.original_size).squeeze()#"""
                # print(sim_.shape)torch.Size([384, 633])
                sim_ = sim_.cpu().numpy()
                sim.append(sim_)

        sim = np.array(sim).mean(0)
        # print(sim.shape)(384, 633)

        target_embedding = torch.mean(torch.concat(target_embedding, dim=0), dim=0, keepdim=True)
        # print(target_embedding.shape)
        # torch.Size([1, 1, 256])
        # exit()
        return sim, target_embedding, mask_size


    def _generate_masks(self, image: np.ndarray, ref_box) -> MaskData:
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )


        b0 = ref_box[0]
        b1 = ref_box[1]
        b2 = ref_box[2]
        db = min(b0[2]-b0[0],b0[3]-b0[1],b1[2]-b1[0],b1[3]-b1[1],b2[2]-b2[0],b2[3]-b2[1])/2
        # print(db)
        # exit(0)
        

        number_of_points = max(int(orig_size[0]/db),int(orig_size[1]/db))
        self.point_grids = build_all_layer_point_grids(
                number_of_points,
                self.crop_n_layers,
                self.crop_n_points_downscale_factor,
            )

        # Iterate over image crops
        # data = MaskData()
        data_dic = defaultdict(MaskData)
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            # print(layer_idx)
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size, ref_box)
            # print(crop_data["points"])
            
            # plt.figure(figsize=(10,10))
            # plt.imshow(image)
            # show_mask(crop_data["segmentations"][0], plt.gca())
            # show_mask(crop_data["segmentations"][1], plt.gca())
            # show_mask(crop_data["segmentations"][2], plt.gca())
            # # show_anns(crop_data)
            # plt.axis('off')
            # plt.savefig('try.jpg')
            # plt.close()
            # exit()
            data_dic[layer_idx].cat(crop_data)

        data = MaskData()
        for layer_idx in data_dic.keys():
            # print(self.prototype[layer_idx][0].shape) [3,256]
            proto_fea = torch.concat(self.prototype[layer_idx], dim=0)
            if len(proto_fea) > 1:
                cos_dis = proto_fea @ proto_fea.t()
                # sim_thresh = torch.min(cos_dis)
                sim_thresh = 0.66

            else:
                sim_thresh = 0.7
            sub_data = data_dic[layer_idx]
            fea = sub_data['fea']
            # print(fea.shape) 91*256
            # print(proto_fea.shape) 3*256
            cos_dis = torch.max(fea @ proto_fea.t(), dim=1)[0]
            sub_data.filter(cos_dis>=sim_thresh)
            data.cat(sub_data)


        self.prototype = defaultdict(list)


        # Remove duplicate masks between crops
        if len(crop_boxes) > 1:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros(len(data["boxes"])),  # categories
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)

        data.to_numpy()
        return data

    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
        ref_box,
    ) -> MaskData:
        
        # print(ref_box)[[337, 176, 465, 240], [348, 292, 435, 310], [245, 75, 371, 135]]
        # exit()
        # Crop the image and calculate embeddings

        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]


        self.predictor.set_image(cropped_im)
        # sim, target_embedding, mask_size = self._generate_similarity(image, ref_box)
        # # print(sim.shape)
        # # exit(0)
        # target_size = self.predictor.transform.get_preprocess_shape(sim.shape[0], sim.shape[1], self.predictor.transform.target_length)
        # sim_map = resize(sim,target_size,preserve_range=True)

        # #"""
        # T = np.max(sim_map)/1.3
        # sim_map[sim_map<T]=0
        # sim_map[sim_map>=T]=1 #"""

        ref_box = pre_process_ref_box(ref_box, crop_box, crop_layer_idx)
        if len(ref_box) > 0:
            ref_box = torch.tensor(ref_box, device=self.predictor.device)
            transformed_boxes = self.predictor.transform.apply_boxes_torch(ref_box, cropped_im_size)

            masks, iou_preds, low_res_masks = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False
                )

            
            masks_for_ref_ori_size = F.interpolate(masks, size=cropped_im_size, mode='bilinear', align_corners=False)
            masks_for_ref_ori_size = torch.sum(masks_for_ref_ori_size, 0).flatten(0,1).to('cpu').unsqueeze(2)
            # # print(masks_for_ref_ori_size.shape)
            pixels_of_objects = image*masks_for_ref_ori_size.numpy()
            # # area_of_object = masks_for_ref_ori_size.numpy().sum()/3
            mean_pixels_of_objects = np.sum(pixels_of_objects, axis=(0,1))/masks_for_ref_ori_size.numpy().sum()

            self.points_per_batch = min(int(masks_for_ref_ori_size.numpy().sum()/6),32)

            # max_pixel_value = np.max(pixels_of_objects,axis=(0,1))
            # whitewall = 255*np.ones_like(image)
            # bg = whitewall*(1-masks_for_ref_ori_size.numpy())
            # bgimg = bg+image
            # # print(pixels_of_objects)
            # # exit()
            # min_pixel_value = np.min(bgimg,axis=(0,1))
            # print(max_pixel_value)
            # print(min_pixel_value)

            # exit()
            # print(0.9*mean_pixels_of_objects)
            points_maskgt1 = image[:,:,0]>1*mean_pixels_of_objects[0]
            points_masklt1 = image[:,:,0]< 1.0*mean_pixels_of_objects[0]

            points_maskgt2 = image[:,:,1]> 1*mean_pixels_of_objects[1]
            points_masklt2 = image[:,:,1]< 1.0*mean_pixels_of_objects[1]

            points_maskgt3 = image[:,:,2]> 1*mean_pixels_of_objects[2]
            points_masklt3 = image[:,:,2]< 1.0*mean_pixels_of_objects[2]


            # points_maskgt1 = image[:,:,0]>1.1*min_pixel_value[0]
            # points_masklt1 = image[:,:,0]< 0.9*max_pixel_value[0]

            # points_maskgt2 = image[:,:,1]> 1.1*min_pixel_value[1]
            # points_masklt2 = image[:,:,1]< 0.9*max_pixel_value[1]

            # points_maskgt3 = image[:,:,2]> 1.1*min_pixel_value[2]
            # points_masklt3 = image[:,:,2]< 0.9*max_pixel_value[2]

            # print(points_maskgt1.shape)
            # print(points_maskgt.sum())
            # print(points_masklt.sum())

            points_mask = (points_maskgt1 & points_masklt1)&(points_maskgt2 & points_masklt2)&(points_maskgt3 & points_masklt3)
            # print(points_mask.shape)


            # points_new = []
            # for pi in range(cropped_im_size[0]):
            #     for pj in range(cropped_im_size[1]):
            #         if points_mask[pi][pj]:
            #             points_new.append([pj,pi])


            # points_new=np.array(points_new)

            # print(points_new)
            # print(points_mask.shape)
            # plt.figure(figsize=(10,10))
            # plt.imshow(image)
            # show_mask2(masks_for_ref_ori_size.squeeze(), plt.gca()) 
            # show_mask2(points_mask.squeeze(), plt.gca()) 
            # # show_anns(crop_data)
            # plt.axis('off')
            # plt.savefig('try2.jpg')
            # plt.close()
            # # print(mean_pixels_of_objects)
            # # print(pixels_of_objects.shape)

            # exit(0)
            feature = self.predictor.get_image_embedding()
            # print(low_res_masks.shape)
            # print(low_res_masks.shape)[3, 1, 256, 256] 3 is the number of patches

            low_res_masks = F.interpolate(low_res_masks, size=feature.shape[-2:], mode='bilinear', align_corners=False)

            # print(low_res_masks.shape)[3, 1, 64, 64]
            
            feature = feature.flatten(2, 3)
            low_res_masks = low_res_masks.flatten(2, 3)

            # print(low_res_masks.shape)[3, 1, 4096]
            masks_low_res = (low_res_masks > self.predictor.model.mask_threshold).float()
            # print(masks_low_res.shape)[3, 1, 4096]
            topk_idx = torch.topk(low_res_masks, 1)[1]
            masks_low_res.scatter_(2, topk_idx, 1.0)

            # print(feature.shape)[1, 256, 4096]
            # print(masks_low_res.shape)[3, 1, 4096]
            prototype_fea = (feature * masks_low_res).sum(dim=2) / masks_low_res.sum(dim=2)
            prototype_fea = F.normalize(prototype_fea, dim=1)
            self.prototype[crop_layer_idx].append(prototype_fea)


        if crop_layer_idx == 0:                                     # add reference gounding
            x = ref_box[:, 0] + (ref_box[:, 2] - ref_box[:, 0]) / 2
            y = ref_box[:, 1] + (ref_box[:, 3] - ref_box[:, 1]) / 2
            points = torch.stack([x, y], dim=1)
            data = MaskData(
                masks=masks.flatten(0, 1),
                iou_preds= torch.ones_like(iou_preds.flatten(0, 1)),
                fea = prototype_fea,
                points=points.cpu(),
                stability_score = torch.ones_like(iou_preds.flatten(0, 1)),
            )
            
            data["boxes"] = batched_mask_to_box(data["masks"])
            data["rles"] = mask_to_rle_pytorch(data["masks"])
            del data["masks"]
        else:
            data = MaskData()

        

        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale


        # ind = 0
        points_new = []
        for p in points_for_image:
            # print(p[0])
            # print(p[1])
            if points_mask[int(p[1]),int(p[0])]:
                points_new.append(p)
        points_new=np.array(points_new)

        # Generate masks for this crop in batches
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            # print(ind)
            # ind = ind + 1
            batch_data = self._process_batch(points, cropped_im_size, 
                                             crop_box, orig_size)
            data.cat(batch_data)
            del batch_data
        self.predictor.reset_image()
        # exit(0)

        # Remove duplicates within this crop.
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros(len(data["boxes"]), device="cuda"),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)



        # keep_mask = area_from_rle(data["rles"]) > 0.4*area_of_object
        # data.filter(keep_mask)


        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

        return data

    def _process_batch(
        self,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...]
    ) -> MaskData:
        orig_h, orig_w = orig_size
        

        # print(crop_box)
        # exit()
        # Run model on this batch
        transformed_points = self.predictor.transform.apply_coords(points, im_size).astype(int)
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        # print(transformed_points)
        # exit()
        # transformed_labels = sim_map[(transformed_points[:,1],transformed_points[:,0])]
        # in_labels = torch.as_tensor(transformed_labels, device=self.predictor.device)
        # ref_prompt = np.array(ref_box.cpu())
        # transformed_boxes = self.predictor.transform.apply_boxes(ref_prompt, im_size)
        # boxes = points2box(in_points,transformed_boxes).to(device=self.predictor.device)

        masks, iou_preds, low_res_masks = self.predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            # boxes=boxes,
            multimask_output=False,
            return_logits=True,
        )
        
        masks_tmp = F.interpolate(masks, size=(im_size), mode='bilinear', align_corners=False)
        masks_tmp = torch.sum(masks_tmp, 0).flatten(0,1).to('cpu').unsqueeze(2)
        # show_mask2(masks_for_ref_ori_size.squeeze(), plt.gca()) 
        # show_mask2(masks_tmp.squeeze(), plt.gca()) 
        #     # show_anns(crop_data)
        # plt.axis('off')
        # plt.savefig('try2.jpg')
        # plt.close()
        
        feature = self.predictor.get_image_embedding()
        low_res_masks=low_res_masks.flatten(0, 1)
        low_res_masks = F.interpolate(low_res_masks[:, None, :, :], size=feature.shape[-2:], 
                                      mode='bilinear', align_corners=False)
        # low_res_masks = low_res_masks > self.predictor.model.mask_threshold

        # fea = feature.flatten(2, 3)
        # low_res_masks = low_res_masks.flatten(2, 3)
        # topk_idx = torch.topk(low_res_masks, 4)[1]
        # fea = fea.expand(topk_idx.shape[0], -1, -1)
        # topk_idx = topk_idx.expand(-1, fea.shape[1], -1)
        # fea = fea.gather(2, topk_idx)


        feature = feature.flatten(2, 3)
        low_res_masks = low_res_masks.flatten(2, 3)
        masks_low_res = (low_res_masks > self.predictor.model.mask_threshold).float()
        topk_idx = torch.topk(low_res_masks, 1)[1]
        masks_low_res.scatter_(2, topk_idx, 1.0)
        pool_fea = (feature * masks_low_res).sum(dim=2) / masks_low_res.sum(dim=2)
        pool_fea = F.normalize(pool_fea, dim=1)

        # print(pool_fea.shape)

        # k_val = torch.topk(torch.flatten(low_res_masks, start_dim=2, end_dim=3), k=4, dim=-1)[0][:, :, -1, None]
        # low_res_masks = (low_res_masks >= k_val).float()
        # low_res_masks = low_res_masks.float()
        # pool_fea = (feature * low_res_masks).sum(dim=(2, 3)) / low_res_masks.sum(dim=(2, 3))



        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
            fea = pool_fea,
        )
        del masks


        # Filter by predicted IoU
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)

        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.predictor.model.mask_threshold, self.stability_score_offset
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            # data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.predictor.model.mask_threshold
        # data["masks"] = data["masks"]
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries
        # keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        # if not torch.all(keep_mask):
        #     data.filter(keep_mask)

        # print(len(data['fea']))

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        return data

    @staticmethod
    def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        if len(mask_data["rles"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros(len(boxes)),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data

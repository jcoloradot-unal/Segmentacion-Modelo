import os
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from typing import List, TypedDict
from fastapi import UploadFile
from pathlib import Path
import shutil
import uuid
import torch._dynamo
torch._dynamo.config.suppress_errors = True

torch.set_default_dtype(torch.float32)
    
# Disable autocast if it's causing issues
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# helper functions
DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"

# change to customized path
OUTPUT_DIR = os.getcwd() + "\\output"
FRAMES_OUTPUT = "frames"
VIDEO_OUTPUT = "video"
MODEL_CONFIG = r"/workspace/Segmentacion-Modelo/MedSam2/MedSamRepo/sam2/configs/sam2.1_hiera_t512.yaml"
MODEL_CHECKPOINT = r"/workspace/Segmentacion-Modelo/MedSam2/MedSamRepo/MedSAM2_latest.pt"

class PromptData(TypedDict):
    box: tuple[int, int, int, int]
    frame: int

def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette

def get_mask(per_obj_mask, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    object_ids = sorted(per_obj_mask)[::-1]
    count = 0
    for object_id in object_ids:
        count += 1
        object_mask = per_obj_mask[object_id]
        object_mask = object_mask.reshape(height, width)
        mask[object_mask] = object_id
    return mask

def get_from_dic_no_errors(dic, key):
    try:
        return dic[key]
    except:
        return None

def clear_folder(folder_path):
    """
    Deletes all files and subdirectories inside a folder using only os.
    Keeps the folder itself.
    """
    for entry in os.listdir(folder_path):
        path = os.path.join(folder_path, entry)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            elif os.path.isdir(path):
                # Recursively delete subdirectory
                for root, dirs, files in os.walk(path, topdown=False):
                    for file in files:
                        os.remove(os.path.join(root, file))
                    for dir_ in dirs:
                        os.rmdir(os.path.join(root, dir_))
                os.rmdir(path)
        except Exception as e:
            print(f"Failed to delete {path}: {e}")

class PromptData(TypedDict):
    box: tuple[int, int, int, int]
    frame: int

INPUT_FOLDER_STORAGE = Path(r"C:\Users\MSI\Desktop\Apps\Apps\Segmentacion\Segmentacion-Modelo\MedSam2\MedSamRepo\inputs")

def save_images(uuid: str, images: List[UploadFile], lowest_frame, greatest_frame) -> Path:
    if not images:
        raise ValueError("No Images provided")
    
    # Create the main storage folder if it doesn't exist
    INPUT_FOLDER_STORAGE.mkdir(exist_ok=True, parents=True)
    
    # Create UUID-specific folder
    uuid_folder = INPUT_FOLDER_STORAGE / (uuid + "_fwd")
    uuid_folder_bwd = INPUT_FOLDER_STORAGE / (uuid + "_bwd")
    uuid_folder.mkdir(exist_ok=True, parents=True)
    uuid_folder_bwd.mkdir(exist_ok=True, parents=True)
    
    # Save each DICOM file
    saved_files = []
    try:
        for idx, file in enumerate(images):
            print("INDEX AS FILE?:", idx)
            if idx >= lowest_frame:
                # Just storing images from lowest moving forward
                # Use original filename or create sequential name
                # if file.filename:
                #     filename = file.filename
                # else:
                filename = f"{idx:04d}.jpg"
                
                # Full path for the file
                file_path = uuid_folder / filename
                
                # Save the file
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                saved_files.append(file_path)
                print(f"✓ Saved: {filename}")
            if idx < lowest_frame:
                filename = f"{(len(images) - idx - 1):04d}.jpg"
                file_path = uuid_folder_bwd / filename
                # Save the file
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
        
        print(f"✓ Saved {len(saved_files)} DICOM files to {uuid_folder}")
        return uuid_folder, uuid_folder_bwd
    
    except Exception as e:
        # Clean up on error (optional - remove if you want to keep partial uploads)
        if uuid_folder.exists():
            shutil.rmtree(uuid_folder)
        raise IOError(f"Failed to save DICOM images: {str(e)}")
    finally:
        # Close all file handles
        for file in images:
            file.file.close()

def predict_masks(files: List[UploadFile], boxes: List[PromptData]):
    id = str(uuid.uuid4())
    input_folder = save_images(id, files)
    
    predictor = build_sam2_video_predictor(
        config_file=MODEL_CONFIG,
        ckpt_path=MODEL_CHECKPOINT,
        apply_postprocessing=True,
        vos_optimized=False,
    )
    
    # Load the video frames
    frame_names = [
        os.path.splitext(p)[0] + ".jpg"
        for p in os.listdir(input_folder)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names = list(sorted(frame_names))
    
    inference_state = predictor.init_state(
        video_path=str(input_folder),  # Convert Path to string
        async_loading_frames=False
    )

    predictor.reset_state(inference_state)

    height = inference_state["video_height"]
    width = inference_state["video_width"]

    # Add prompts for each box
    for i in range(len(boxes)):
        frame = boxes[i]["frame"]  # Use dict key access
        prompt = boxes[i]["box"]   # Use dict key access
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame,
            obj_id=i + 1,
            box=prompt
        )

    # Run propagation throughout the video
    video_segments = {}  # Store the per-frame segmentation results
    
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        per_obj_output_mask = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        video_segments[out_frame_idx] = per_obj_output_mask

    # Generate masks for each frame
    masks = []
    for out_frame_idx, per_obj_output_mask in video_segments.items():
        mask = get_mask(per_obj_output_mask, height, width)
        masks.append({
            "frame": out_frame_idx,
            "mask": mask.tolist()  # Convert numpy array to list for JSON serialization
        })
    
    # Clean up the temporary folder
    try:
        shutil.rmtree(input_folder)
        print(f"✓ Cleaned up temporary folder: {input_folder}")
    except Exception as e:
        print(f"Warning: Could not clean up {input_folder}: {e}")
    
    print(f"RETURNING {len(masks)} masks")
    return masks

# predict_masks()

class PointPromptData(TypedDict):
    points: List[tuple[int, int]]  # List of (x, y) coordinates
    labels: List[int]  # List of labels (1 for positive, 0 for negative)
    frame: int

def predict_masks_with_points(files: List[UploadFile], point_prompts: List[PointPromptData]):
    id = str(uuid.uuid4())
    
    lowest_frame = min(point["frame"] for point in point_prompts)
    greatest_frame = max(point["frame"] for point in point_prompts)

    input_folder, input_folder_bwd = save_images(id, files, lowest_frame, greatest_frame)
    
    predictor = build_sam2_video_predictor(
        config_file=MODEL_CONFIG,
        ckpt_path=MODEL_CHECKPOINT,
        apply_postprocessing=True,
        vos_optimized=False,
    )
    
    # Load the video frames
    frame_names = [
        os.path.splitext(p)[0] + ".jpg"
        for p in os.listdir(input_folder)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names = list(sorted(frame_names))
    
    inference_state = predictor.init_state(
        video_path=str(input_folder),
        async_loading_frames=False
    )
    
    # inference_state2 = predictor.init_state(
    #     video_path=str(input_folder_bwd),
    #     async_loading_frames=False
    # )

    predictor.reset_state(inference_state)
    # predictor.reset_state(inference_state2)

    height = inference_state["video_height"]
    width = inference_state["video_width"]

    # Add prompts for each point set
    for i in range(len(point_prompts)):
        frame = point_prompts[i]["frame"] - lowest_frame
        points = np.array(point_prompts[i]["points"], dtype=np.float32)
        labels = np.array(point_prompts[i]["labels"], dtype=np.int32)
        
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame,
            obj_id=i + 1,
            points=points,
            labels=labels,
        )

    # Run propagation throughout the video
    video_segments = {}
    
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        print("THIS ID:", out_obj_ids)
        
        per_obj_output_mask = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        video_segments[out_frame_idx] = per_obj_output_mask
    
    # Generate masks for each frame fwd
    masks = []   
    for out_frame_idx, per_obj_output_mask in video_segments.items():
        mask = get_mask(per_obj_output_mask, height, width)
        masks.append({
            "frame": out_frame_idx + lowest_frame,
            "mask": mask.tolist()
        })
        
    
    predictor2 = build_sam2_video_predictor(
        config_file=MODEL_CONFIG,
        ckpt_path=MODEL_CHECKPOINT,
        apply_postprocessing=True,
        vos_optimized=False,
    ) 
    
    # checks if there is actually images to mask backwards
    if len(os.listdir(input_folder_bwd)) > 0:
        inference_state2 = predictor2.init_state(
            video_path=str(input_folder_bwd),
            async_loading_frames=False
        )

        predictor2.reset_state(inference_state2)
        
        point_prompts = [f for f in point_prompts if f["frame"] == lowest_frame]
        # Add prompts for each point set backwards
        for i in range(len(point_prompts)):
            frame = 0
            points = np.array(point_prompts[i]["points"], dtype=np.float32)
            labels = np.array(point_prompts[i]["labels"], dtype=np.int32)
            predictor2.add_new_points_or_box(
                inference_state=inference_state2,
                frame_idx=frame,
                obj_id=i + 1,
                points=points,
                labels=labels,
            )
            
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor2.propagate_in_video(
            inference_state2
        ):
            per_obj_output_mask = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            video_segments[out_frame_idx] = per_obj_output_mask
    
    # Generate masks for each frame bwd
    for out_frame_idx, per_obj_output_mask in video_segments.items():
        mask = get_mask(per_obj_output_mask, height, width)
        masks.append({
            "frame": greatest_frame - out_frame_idx,
            "mask": mask.tolist()
        })
        
    # Clean up the temporary folder
    try:
        shutil.rmtree(input_folder)
        shutil.rmtree(input_folder_bwd)
        print(f"✓ Cleaned up temporary folders: {input_folder}")
    except Exception as e:
        print(f"Warning: Could not clean up {input_folder}: {e}")
    
    print(f"RETURNING {len(masks)} masks")
    return masks
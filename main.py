import sieve
import os
import shutil

# Create output directory
output_dir = "sam2_output"
os.makedirs(output_dir, exist_ok=True)

# Use local file
file = sieve.File("vid.mov")

model_type = "tiny"
prompts = [
  {
    "frame_index": 0,
    "object_id": 1,
    "points": [
      [
        244.22,
        667.2
      ]
    ],
    "labels": [
      1
    ]
  }
]
debug_masks = True
multimask_output = False
bbox_tracking = True
pixel_confidences = True
start_frame = -1
end_frame = -1
preview = False
frame_interval = 3

sam2 = sieve.function.get("sieve/sam2")
output = sam2.run(file, model_type, prompts, debug_masks, multimask_output, bbox_tracking, pixel_confidences, start_frame, end_frame, frame_interval, preview)

# Save output files and print their paths
debug_video, files = output

def save_and_print(file_obj, filename):
    dest_path = os.path.join(output_dir, filename)
    shutil.copy(file_obj.path, dest_path)
    print(f"{filename} saved to: {dest_path}")

save_and_print(debug_video, "debug_video.mp4")
save_and_print(files['masks'], "masks.zip")
save_and_print(files['confidences'], "confidences.zip")
save_and_print(files['bbox_tracking'], "bbox_tracking.json")
save_and_print(files['debug_video'], "debug_video_from_files.mp4")

print(f"\nAll files have been saved to the '{output_dir}' directory.")
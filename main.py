import os
import shutil
import tempfile
import uuid
import json
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import sieve
import numpy as np

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

# Ensure the output directory exists
output_dir = "sam2_output"
os.makedirs(output_dir, exist_ok=True)

@app.route('/process_video', methods=['POST'])
def process_video():
    # Check if the post request has the file part
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    
    # Check if a filename is provided
    if video_file.filename == '':
        return jsonify({'error': 'No selected video file'}), 400
    
    # Get coordinates and pixel-to-meter conversion from the request
    x = request.form.get('x')
    y = request.form.get('y')
    pixel_to_meter = request.form.get('pixel_to_meter')
    
    if not x or not y or not pixel_to_meter:
        return jsonify({'error': 'Coordinates or pixel_to_meter not provided'}), 400
    
    try:
        x = float(x)
        y = float(y)
        pixel_to_meter = float(pixel_to_meter)
    except ValueError:
        return jsonify({'error': 'Invalid coordinates or pixel_to_meter value'}), 400

    # Convert pixel_to_meter to meters per pixel
    meters_per_pixel = 1 / pixel_to_meter

    # Generate a UUID for this video processing job
    job_id = str(uuid.uuid4())

    # Create a directory for this job
    job_dir = os.path.join(output_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mov') as temp_video:
        video_file.save(temp_video.name)
        
        # Process the video with SAM2
        file = sieve.File(temp_video.name)
        
        prompts = [
            {
                "frame_index": 0,
                "object_id": 1,
                "points": [[x, y]],
                "labels": [1]
            }
        ]
        
        model_type = "tiny"
        debug_masks = True
        multimask_output = False
        bbox_tracking = True
        pixel_confidences = False
        start_frame = -1
        end_frame = -1
        preview = False
        frame_interval = 2

        sam2 = sieve.function.get("sieve/sam2")
        output = sam2.run(file, model_type, prompts, debug_masks, multimask_output, bbox_tracking, pixel_confidences, start_frame, end_frame, frame_interval, preview)

        debug_video, files = output

        # Save bbox_tracking data
        bbox_tracking_path = os.path.join(job_dir, "bbox_tracking.json")
        shutil.copy(files['bbox_tracking'].path, bbox_tracking_path)

        # Save debug video
        debug_video_path = os.path.join(job_dir, "debug_video.mp4")
        shutil.copy(debug_video.path, debug_video_path)

        # Load the bbox_tracking JSON data
        with open(bbox_tracking_path, 'r') as f:
            bbox_data = json.load(f)

        # Extract X and Y coordinates and timesteps, then convert to meters
        x_values = []
        y_values = []
        time_steps = []

        max_y_value = None

        for frame_data in bbox_data.values():
            for entry in frame_data:
                bbox = entry['bbox']
                # Calculate center points
                x_center = (bbox[0] + bbox[2]) / 2
                y_center = (bbox[1] + bbox[3]) / 2
                x_values.append(x_center * meters_per_pixel)
                y_values.append(y_center)
                time_steps.append(entry['timestep'])

        # Flip Y values (video coordinate system flip)
        max_y_value = max(y_values)
        y_values_flipped = [(max_y_value - y) * meters_per_pixel for y in y_values]

        # Calculate velocities
        x_velocities = np.diff(x_values) / np.diff(time_steps)
        y_velocities = np.diff(y_values_flipped) / np.diff(time_steps)
        velocity_time_steps = time_steps[1:]  # Velocities correspond to intervals

        # Prepare data for JSON response
        response_data = {
            "positions": {
                "time_steps": time_steps,
                "x_positions_meters": x_values,
                "y_positions_meters_flipped": y_values_flipped
            },
            "velocities": {
                "time_steps": velocity_time_steps,
                "x_velocities_m_per_s": x_velocities.tolist(),
                "y_velocities_m_per_s": y_velocities.tolist()
            },
            "debug_video_url": f'/get_debug_video/{job_id}'  # Include URL to fetch debug video
        }

    # Clean up the temporary file
    os.unlink(temp_video.name)

    # Return the position, velocity data, and debug video path as JSON
    return jsonify(response_data)

@app.route('/get_debug_video/<job_id>', methods=['GET'])
def get_debug_video(job_id):
    debug_video_path = os.path.join(output_dir, job_id, "debug_video.mp4")
    if os.path.exists(debug_video_path):
        return send_file(debug_video_path, mimetype='video/mp4')
    else:
        return jsonify({'error': 'Video not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)

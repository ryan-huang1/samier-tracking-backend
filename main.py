import os
import shutil
import tempfile
import uuid
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import sieve

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
    
    # Get coordinates from the request
    x = request.form.get('x')
    y = request.form.get('y')
    
    if not x or not y:
        return jsonify({'error': 'Coordinates not provided'}), 400
    
    try:
        x = float(x)
        y = float(y)
    except ValueError:
        return jsonify({'error': 'Invalid coordinates'}), 400

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
        frame_interval = 3

        sam2 = sieve.function.get("sieve/sam2")
        output = sam2.run(file, model_type, prompts, debug_masks, multimask_output, bbox_tracking, pixel_confidences, start_frame, end_frame, frame_interval, preview)

        debug_video, files = output

        # Save bbox_tracking data
        bbox_tracking_path = os.path.join(job_dir, "bbox_tracking.json")
        shutil.copy(files['bbox_tracking'].path, bbox_tracking_path)

        # Save debug video
        debug_video_path = os.path.join(job_dir, "debug_video.mp4")
        shutil.copy(debug_video.path, debug_video_path)

    # Clean up the temporary file
    os.unlink(temp_video.name)

    # Return the bbox_tracking data and debug video
    with open(bbox_tracking_path, 'r') as f:
        bbox_data = f.read()

    return jsonify({
        'job_id': job_id,
        'bbox_tracking': bbox_data,
        'debug_video_url': f'/get_debug_video/{job_id}'
    })

@app.route('/get_debug_video/<job_id>', methods=['GET'])
def get_debug_video(job_id):
    debug_video_path = os.path.join(output_dir, job_id, "debug_video.mp4")
    if os.path.exists(debug_video_path):
        return send_file(debug_video_path, mimetype='video/mp4')
    else:
        return jsonify({'error': 'Video not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
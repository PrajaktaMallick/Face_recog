from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import threading
import time
import streamlit as st
# ... your Streamlit code ...
from analysis import analyze_face_features_advanced

app = Flask(__name__)

camera = None
camera_active = False
camera_lock = threading.Lock()

class AdvancedVideoCamera:
    def __init__(self):
        self.video = None
        for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
            try:
                self.video = cv2.VideoCapture(0, backend)
                if self.video.isOpened():
                    break
                self.video.release()
            except:
                continue

        if self.video is None or not self.video.isOpened():
            raise Exception("Could not open camera")

        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video.set(cv2.CAP_PROP_FPS, 30)
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        time.sleep(1)

        ret, frame = self.video.read()
        if not ret:
            raise Exception("Camera opened but cannot read frames")

    def __del__(self):
        if hasattr(self, 'video'):
            self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        if not success:
            return None

        display_image = cv2.resize(image, (640, 480))

        ret, jpeg = cv2.imencode('.jpg', display_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return jpeg.tobytes()

    def capture_high_quality_frame(self):
        success, image = self.video.read()
        if not success:
            return None
        return image

def generate_frames():
    global camera, camera_active

    try:
        while True:
            with camera_lock:
                if camera is None or not camera_active:
                    break

            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033) 
    except Exception as e:
        print(f"Error in generate_frames: {e}")
    finally:
        with camera_lock:
            camera_active = False

@app.route('/')
def index():
    """Main page route"""
    return render_template('index.html')

@app.route('/camera_feed')
def camera_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/initialize_camera', methods=['POST'])
def initialize_camera():
    global camera, camera_active
    try:
        with camera_lock:
            if camera is not None:
                try:
                    camera.video.release()
                except:
                    pass
            camera = AdvancedVideoCamera()
            camera_active = True
        return jsonify({'success': True, 'message': 'Camera initialized'})
    except Exception as e:
        print(f"Camera initialization error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/capture_and_analyze', methods=['POST'])
def capture_and_analyze():
    global camera

    try:
        data = request.get_json()
        selected_features = data.get('features', [])

        if not selected_features:
            return jsonify({
                'success': False,
                'error': 'Please select at least one feature to analyze'
            })

        with camera_lock:
            if camera is None:
                return jsonify({'success': False, 'error': 'Camera not initialized'})

            frame = camera.capture_high_quality_frame()
            if frame is None:
                return jsonify({'success': False, 'error': 'Failed to capture frame'})

        results = analyze_face_features_advanced(frame, selected_features)

        if results['success']:
            return jsonify({
                'success': True,
                'results': results['data'],
                'metadata': results.get('metadata', {})
            })
        else:
            return jsonify({
                'success': False,
                'error': results['error']
            })

    except Exception as e:
        print(f"Error in capture_and_analyze: {e}")
        return jsonify({'success': False, 'error': f'Analysis failed: {str(e)}'})

@app.route('/cleanup_camera', methods=['POST'])
def cleanup_camera():
    global camera, camera_active
    try:
        with camera_lock:
            camera_active = False
            if camera:
                del camera
                camera = None
        return jsonify({'success': True, 'message': 'Camera cleaned up'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'message': 'Pehechan AI - Advanced Facial Analysis System'})

import atexit

def cleanup_on_exit():
    global camera, camera_active
    try:
        with camera_lock:
            camera_active = False
            if camera:
                del camera
                camera = None
    except:
        pass

atexit.register(cleanup_on_exit)


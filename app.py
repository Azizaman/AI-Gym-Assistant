
import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from dotenv import load_dotenv
import os
import boto3
from botocore.config import Config
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, jsonify, g
from werkzeug.utils import secure_filename
import logging
import uuid
from flask_cors import CORS
from pymongo import MongoClient
from firebase_admin import auth, credentials
import firebase_admin
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Load environment variables and initialize app (unchanged)
print("Imports done")
load_dotenv()
print("Environment variables loaded")
app = Flask(__name__)
CORS(app)

# Firebase, MongoDB, Wasabi initialization (unchanged)
try:
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not set")
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Google Gemini API configured successfully")
    FIREBASE_CREDENTIALS_PATH = os.getenv('FIREBASE_CREDENTIALS_PATH')
    if not FIREBASE_CREDENTIALS_PATH or not os.path.exists(FIREBASE_CREDENTIALS_PATH):
        raise FileNotFoundError(f"Firebase credentials file not found at: {FIREBASE_CREDENTIALS_PATH}")
    cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
    firebase_admin.initialize_app(cred)
    logger.info("Firebase Admin SDK initialized successfully")
except Exception as e:
    logger.error(f"Firebase initialization failed: {e}")
    raise

try:
    mongo_client = MongoClient(os.getenv('MONGODB_URI'))
    mongo_client.server_info()
    logger.info("MongoDB connection successful")
except Exception as e:
    logger.error(f"MongoDB connection failed: {e}")
    raise
db = mongo_client['ai_fitness_db']
videos_collection = db['videos']
users_collection = db['users']
print("MongoDB collections initialized")

try:
    WASABI_ACCESS_KEY = os.getenv('WASABI_ACCESS_KEY')
    WASABI_SECRET_KEY = os.getenv('WASABI_SECRET_KEY')
    WASABI_REGION = os.getenv('WASABI_REGION', 'ap-southeast-1')
    WASABI_ENDPOINT_URL = os.getenv('WASABI_ENDPOINT_URL', 'https://s3.ap-southeast-1.wasabisys.com')
    WASABI_BUCKET_NAME = os.getenv('WASABI_BUCKET_NAME', 'ai-fitness-videos')
    if not WASABI_ACCESS_KEY or not WASABI_SECRET_KEY:
        raise ValueError("Wasabi credentials not found")
    s3_client = boto3.client(
        's3',
        aws_access_key_id=WASABI_ACCESS_KEY,
        aws_secret_access_key=WASABI_SECRET_KEY,
        region_name=WASABI_REGION,
        endpoint_url=WASABI_ENDPOINT_URL,
        config=Config(signature_version='s3v4')
    )
    s3_client.head_bucket(Bucket=WASABI_BUCKET_NAME)
    logger.info("Wasabi S3 bucket initialized successfully")
except Exception as e:
    logger.error(f"Wasabi setup failed: {e}")
    raise

UPLOAD_FOLDER = 'Uploads'
OUTPUT_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# MediaPipe initialization (unchanged)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

EXERCISE_CLASSES = ['squat', 'push_up', 'bicep_curl', 'plank', 'jumping_jack']
EXERCISE_THRESHOLD = 0.7
SIDEBAR_WIDTH = 0

# Unchanged helper functions: get_signed_url, firebase_required, calculate_angle, is_starting_position
def get_signed_url(file_name: str, valid_duration_seconds: int = 604800):
    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': WASABI_BUCKET_NAME,
                'Key': file_name,
                'ResponseContentType': 'video/mp4',
                'ResponseContentDisposition': 'inline'
            },
            ExpiresIn=valid_duration_seconds
        )
        logger.info(f"Generated signed URL for {file_name}")
        return url
    except Exception as e:
        logger.error(f"Failed to generate signed URL for '{file_name}': {e}")
        return None

def firebase_required(f):
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        try:
            decoded_token = auth.verify_id_token(token)
            g.user_id = decoded_token['uid']
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Invalid Firebase token: {e}")
            return jsonify({'error': 'Invalid or missing token'}), 401
    decorated_function.__name__ = f.__name__
    return decorated_function

def calculate_angle(a, b, c):
    try:
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle
    except Exception as e:
        raise ValueError(f"Error calculating angle: {e}")

def is_starting_position(landmarks, exercise, width, height):
    try:
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * width, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * height]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * height]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * width, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * height]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * height]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * width, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * height]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * height]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * width, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * height]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * height]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * width, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * height]

        if exercise == 'squat':
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            hip_height = (left_hip[1] + right_hip[1]) / 2
            shoulder_height = (left_shoulder[1] + right_shoulder[1]) / 2
            return (left_knee_angle > 160 and right_knee_angle > 160 and
                    abs(hip_height - shoulder_height) < 120)
        elif exercise == 'push_up':
            elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            return elbow_angle > 150
        elif exercise == 'bicep_curl':
            elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            return elbow_angle > 150
        elif exercise == 'plank':
            body_angle = calculate_angle(left_shoulder, left_hip, left_ankle)
            return body_angle > 160
        elif exercise == 'jumping_jack':
            arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
            return arm_angle < 70 and leg_angle > 150
        return False
    except Exception as e:
        logger.error(f"Error detecting starting position: {e}")
        return False

# Updated form-checking functions with feedback and logging
def check_squat_form(landmarks, width, height):
    feedback = []
    error_points = []
    knee_angle = None
    hip_angle = None
    issue_detected = False
    try:
        if is_starting_position(landmarks, 'squat', width, height):
            feedback.append("Starting position detected")
            logger.info("Squat: Starting position detected, no feedback assigned")
            return feedback, None, None, error_points
            
        def mp_point(part):
            p = landmarks[mp_pose.PoseLandmark[part].value]
            return [p.x * width, p.y * height]
            
        left_hip = mp_point("LEFT_HIP")
        right_hip = mp_point("RIGHT_HIP")
        left_knee = mp_point("LEFT_KNEE")
        right_knee = mp_point("RIGHT_KNEE")
        left_ankle = mp_point("LEFT_ANKLE")
        right_ankle = mp_point("RIGHT_ANKLE")
        left_shoulder = mp_point("LEFT_SHOULDER")
        right_shoulder = mp_point("RIGHT_SHOULDER")
        left_foot = mp_point("LEFT_HEEL")
        right_foot = mp_point("RIGHT_HEEL")
        
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        knee_angle = (left_knee_angle + right_knee_angle) / 2
        
        avg_hip = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]
        avg_shoulder = [(left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2]
        avg_knee = [(left_knee[0] + right_knee[0]) / 2, (left_knee[1] + right_knee[1]) / 2]
        hip_angle = calculate_angle(avg_shoulder, avg_hip, avg_knee)
        
        back_angle = calculate_angle(
            [(left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2],
            [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2],
            [(left_ankle[0] + right_ankle[0]) / 2, (left_ankle[1] + right_ankle[1]) / 2]
        )
        
        if knee_angle < 60:
            feedback.append("Don't go too deep — aim for ~90° knee angle.")
            error_points.extend([(int(left_knee[0]), int(left_knee[1])), (int(right_knee[0]), int(right_knee[1]))])
            issue_detected = True
            logger.info(f"Squat: Issue detected - Knee angle too deep ({knee_angle:.1f}°)")
        elif knee_angle > 130:
            feedback.append("Squat lower to engage quads and glutes.")
            error_points.extend([(int(left_knee[0]), int(left_knee[1])), (int(right_knee[0]), int(right_knee[1]))])
            issue_detected = True
            logger.info(f"Squat: Issue detected - Knee angle too shallow ({knee_angle:.1f}°)")
            
        if back_angle < 140:
            feedback.append("Keep your back straight to avoid strain.")
            error_points.extend([(int(left_hip[0]), int(left_hip[1])), (int(right_hip[0]), int(right_hip[1]))])
            issue_detected = True
            logger.info(f"Squat: Issue detected - Back angle incorrect ({back_angle:.1f}°)")
            
        hip_y = (left_hip[1] + right_hip[1]) / 2
        knee_y = (left_knee[1] + right_knee[1]) / 2
        if hip_y < knee_y - 70:
            feedback.append("Lower your hips further for a full squat.")
            error_points.extend([(int(left_hip[0]), int(left_hip[1])), (int(right_hip[0]), int(right_hip[1]))])
            issue_detected = True
            logger.info(f"Squat: Issue detected - Hips too high (hip_y={hip_y:.1f}, knee_y={knee_y:.1f})")
            
        foot_distance = abs(left_foot[0] - right_foot[0])
        shoulder_distance = abs(left_shoulder[0] - right_shoulder[0])
        if foot_distance < shoulder_distance * 0.7:
            feedback.append("Place feet shoulder-width apart.")
            error_points.extend([(int(left_foot[0]), int(left_foot[1])), (int(right_foot[0]), int(right_foot[1]))])
            issue_detected = True
            logger.info(f"Squat: Issue detected - Feet too close (foot_distance={foot_distance:.1f})")
            
        if abs(left_knee[0] - left_ankle[0]) > 60 or abs(right_knee[0] - right_ankle[0]) > 60:
            feedback.append("Keep knees over ankles to avoid strain.")
            error_points.extend([(int(left_knee[0]), int(left_knee[1])), (int(right_knee[0]), int(right_knee[1]))])
            issue_detected = True
            logger.info(f"Squat: Issue detected - Knees misaligned")
            
        if not issue_detected:
            feedback.append("Correct squat form")
            logger.info(f"Squat: Correct form detected (knee_angle={knee_angle:.1f}°, hip_angle={hip_angle:.1f}°)")
            
        logger.debug(f"Squat: Feedback assigned: {feedback}")
        return feedback, knee_angle, hip_angle, error_points
    except Exception as e:
        logger.error(f"Error checking squat form: {e}")
        feedback.append("Unable to analyze form")
        logger.info(f"Squat: Feedback due to error: {feedback}")
        return feedback, knee_angle, hip_angle, error_points

def check_push_up_form(landmarks, width, height):
    feedback = []
    error_points = []
    elbow_angle = None
    issue_detected = False
    try:
        if is_starting_position(landmarks, 'push_up', width, height):
            feedback.append("Starting position detected")
            logger.info("Push-up: Starting position detected, no feedback assigned")
            return feedback, None, None, error_points
        def mp_point(part):
            p = landmarks[mp_pose.PoseLandmark[part].value]
            return [p.x * width, p.y * height]
        left_shoulder = mp_point("LEFT_SHOULDER")
        left_elbow = mp_point("LEFT_ELBOW")
        left_wrist = mp_point("LEFT_WRIST")
        right_shoulder = mp_point("RIGHT_SHOULDER")
        left_hip = mp_point("LEFT_HIP")
        left_ankle = mp_point("LEFT_ANKLE")
        elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        body_angle = calculate_angle(left_shoulder, left_hip, left_ankle)
        shoulder_alignment = abs(left_shoulder[1] - right_shoulder[1])
        
        if elbow_angle > 120:
            feedback.append("Lower your chest closer to the ground.")
            error_points.append((int(left_elbow[0]), int(left_elbow[1])))
            issue_detected = True
            logger.info(f"Push-up: Issue detected - Elbow angle too wide ({elbow_angle:.1f}°)")
        elif elbow_angle < 30:
            feedback.append("Don't go too low; keep chest above ground.")
            error_points.append((int(left_elbow[0]), int(left_elbow[1])))
            issue_detected = True
            logger.info(f"Push-up: Issue detected - Elbow angle too tight ({elbow_angle:.1f}°)")
        if body_angle < 150:
            feedback.append("Keep your body in a straight line.")
            error_points.append((int(left_hip[0]), int(left_hip[1])))
            issue_detected = True
            logger.info(f"Push-up: Issue detected - Body angle incorrect ({body_angle:.1f}°)")
        if shoulder_alignment > 40:
            feedback.append("Keep shoulders level for balanced form.")
            error_points.extend([(int(left_shoulder[0]), int(left_shoulder[1])), (int(right_shoulder[0]), int(right_shoulder[1]))])
            issue_detected = True
            logger.info(f"Push-up: Issue detected - Shoulders misaligned (alignment={shoulder_alignment:.1f})")
        elbow_shoulder_angle = calculate_angle(left_wrist, left_elbow, left_shoulder)
        if elbow_shoulder_angle > 70:
            feedback.append("Tuck elbows closer to body (~45° angle).")
            error_points.append((int(left_elbow[0]), int(left_elbow[1])))
            issue_detected = True
            logger.info(f"Push-up: Issue detected - Elbow-shoulder angle too wide ({elbow_shoulder_angle:.1f}°)")
        if not issue_detected:
            feedback.append("Correct push-up form")
            logger.info(f"Push-up: Correct form detected (elbow_angle={elbow_angle:.1f}°)")
        logger.debug(f"Push-up: Feedback assigned: {feedback}")
        return feedback, elbow_angle, None, error_points
    except Exception as e:
        logger.error(f"Error checking push-up form: {e}")
        feedback.append("Unable to analyze form")
        logger.info(f"Push-up: Feedback due to error: {feedback}")
        return feedback, elbow_angle, None, error_points

def check_bicep_curl_form(landmarks, width, height):
    feedback = []
    error_points = []
    elbow_angle = None
    issue_detected = False
    try:
        if is_starting_position(landmarks, 'bicep_curl', width, height):
            feedback.append("Starting position detected")
            logger.info("Bicep curl: Starting position detected, no feedback assigned")
            return feedback, None, None, error_points
        def mp_point(part):
            p = landmarks[mp_pose.PoseLandmark[part].value]
            return [p.x * width, p.y * height]
        shoulder = mp_point("LEFT_SHOULDER")
        elbow = mp_point("LEFT_ELBOW")
        wrist = mp_point("LEFT_WRIST")
        hip = mp_point("LEFT_HIP")
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        torso_angle = calculate_angle(shoulder, hip, [hip[0], hip[1] + 100])
        if elbow_angle > 170:
            feedback.append("Avoid fully locking your arm.")
            error_points.append((int(elbow[0]), int(elbow[1])))
            issue_detected = True
            logger.info(f"Bicep curl: Issue detected - Elbow angle too extended ({elbow_angle:.1f}°)")
        if elbow_angle > 120:
            feedback.append("Curl higher for full range of motion.")
            error_points.append((int(elbow[0]), int(elbow[1])))
            issue_detected = True
            logger.info(f"Bicep curl: Issue detected - Elbow angle too wide ({elbow_angle:.1f}°)")
        if torso_angle < 60:
            feedback.append("Keep torso upright; avoid swinging.")
            error_points.append((int(hip[0]), int(hip[1])))
            issue_detected = True
            logger.info(f"Bicep curl: Issue detected - Torso angle incorrect ({torso_angle:.1f}°)")
        if abs(elbow[0] - shoulder[0]) > 60:
            feedback.append("Keep elbow close to torso.")
            error_points.append((int(elbow[0]), int(elbow[1])))
            issue_detected = True
            logger.info(f"Bicep curl: Issue detected - Elbow misaligned")
        wrist_elbow_angle = calculate_angle(elbow, wrist, [wrist[0] + 100, wrist[1]])
        if wrist_elbow_angle > 30:
            feedback.append("Keep wrist straight during curl.")
            error_points.append((int(wrist[0]), int(wrist[1])))
            issue_detected = True
            logger.info(f"Bicep curl: Issue detected - Wrist angle incorrect ({wrist_elbow_angle:.1f}°)")
        if not issue_detected:
            feedback.append("Correct bicep curl form")
            logger.info(f"Bicep curl: Correct form detected (elbow_angle={elbow_angle:.1f}°)")
        logger.debug(f"Bicep curl: Feedback assigned: {feedback}")
        return feedback, elbow_angle, None, error_points
    except Exception as e:
        logger.error(f"Error checking bicep curl form: {e}")
        feedback.append("Unable to analyze form")
        logger.info(f"Bicep curl: Feedback due to error: {feedback}")
        return feedback, elbow_angle, None, error_points

def check_plank_form(landmarks, width, height):
    feedback = []
    error_points = []
    body_angle = None
    issue_detected = False
    try:
        if is_starting_position(landmarks, 'plank', width, height):
            feedback.append("Starting position detected")
            logger.info("Plank: Starting position detected, no feedback assigned")
            return feedback, None, None, error_points
        def mp_point(part):
            p = landmarks[mp_pose.PoseLandmark[part].value]
            return [p.x * width, p.y * height]
        left_shoulder = mp_point("LEFT_SHOULDER")
        right_shoulder = mp_point("RIGHT_SHOULDER")
        left_hip = mp_point("LEFT_HIP")
        left_ankle = mp_point("LEFT_ANKLE")
        body_angle = calculate_angle(left_shoulder, left_hip, left_ankle)
        shoulder_alignment = abs(left_shoulder[1] - right_shoulder[1])
        if body_angle < 150:
            feedback.append("Keep your body in a straight line.")
            error_points.append((int(left_hip[0]), int(left_hip[1])))
            issue_detected = True
            logger.info(f"Plank: Issue detected - Body angle incorrect ({body_angle:.1f}°)")
        if shoulder_alignment > 40:
            feedback.append("Keep shoulders level with hips.")
            error_points.extend([(int(left_shoulder[0]), int(left_shoulder[1])), (int(right_shoulder[0]), int(right_shoulder[1]))])
            issue_detected = True
            logger.info(f"Plank: Issue detected - Shoulders misaligned (alignment={shoulder_alignment:.1f})")
        head = mp_point("NOSE")
        head_shoulder_angle = calculate_angle(head, left_shoulder, left_hip)
        if head_shoulder_angle < 150:
            feedback.append("Keep head in line with spine.")
            error_points.append((int(head[0]), int(head[1])))
            issue_detected = True
            logger.info(f"Plank: Issue detected - Head angle incorrect ({head_shoulder_angle:.1f}°)")
        if not issue_detected:
            feedback.append("Correct plank form")
            logger.info(f"Plank: Correct form detected (body_angle={body_angle:.1f}°)")
        logger.debug(f"Plank: Feedback assigned: {feedback}")
        return feedback, body_angle, None, error_points
    except Exception as e:
        logger.error(f"Error checking plank form: {e}")
        feedback.append("Unable to analyze form")
        logger.info(f"Plank: Feedback due to error: {feedback}")
        return feedback, body_angle, None, error_points

def check_jumping_jack_form(landmarks, width, height):
    feedback = []
    error_points = []
    arm_angle = None
    issue_detected = False
    try:
        if is_starting_position(landmarks, 'jumping_jack', width, height):
            feedback.append("Starting position detected")
            logger.info("Jumping jack: Starting position detected, no feedback assigned")
            return feedback, None, None, error_points
        def mp_point(part):
            p = landmarks[mp_pose.PoseLandmark[part].value]
            return [p.x * width, p.y * height]
        left_shoulder = mp_point("LEFT_SHOULDER")
        left_elbow = mp_point("LEFT_ELBOW")
        left_wrist = mp_point("LEFT_WRIST")
        left_hip = mp_point("LEFT_HIP")
        left_knee = mp_point("LEFT_KNEE")
        left_ankle = mp_point("LEFT_ANKLE")
        arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
        if arm_angle < 50:
            feedback.append("Raise arms higher, almost overhead.")
            error_points.append((int(left_elbow[0]), int(left_elbow[1])))
            issue_detected = True
            logger.info(f"Jumping jack: Issue detected - Arm angle too low ({arm_angle:.1f}°)")
        if leg_angle < 130:
            feedback.append("Spread legs wider for full range.")
            error_points.extend([(int(left_knee[0]), int(left_knee[1])), (int(left_ankle[0]), int(left_ankle[1]))])
            issue_detected = True
            logger.info(f"Jumping jack: Issue detected - Leg angle too narrow ({leg_angle:.1f}°)")
        right_arm_angle = calculate_angle(mp_point("RIGHT_SHOULDER"), mp_point("RIGHT_ELBOW"), mp_point("RIGHT_WRIST"))
        if abs(arm_angle - right_arm_angle) > 30:
            feedback.append("Keep arm movements symmetrical.")
            error_points.extend([(int(left_elbow[0]), int(left_elbow[1])), (int(mp_point("RIGHT_ELBOW")[0]), int(mp_point("RIGHT_ELBOW")[1]))])
            issue_detected = True
            logger.info(f"Jumping jack: Issue detected - Arms asymmetrical (left={arm_angle:.1f}°, right={right_arm_angle:.1f}°)")
        if not issue_detected:
            feedback.append("Correct jumping jack form")
            logger.info(f"Jumping jack: Correct form detected (arm_angle={arm_angle:.1f}°)")
        logger.debug(f"Jumping jack: Feedback assigned: {feedback}")
        return feedback, arm_angle, None, error_points
    except Exception as e:
        logger.error(f"Error checking jumping jack form: {e}")
        feedback.append("Unable to analyze form")
        logger.info(f"Jumping jack: Feedback due to error: {feedback}")
        return feedback, arm_angle, None, error_points

# Unchanged functions: count_reps, allowed_file, process_frame (omitted for brevity)
def count_reps(exercise, angles, fps, total_frames):
    try:
        correct_reps = 0
        incorrect_reps = 0
        in_position = False
        if exercise == 'squat':
            for angle in angles:
                if angle is None:
                    continue
                if angle < 110 and not in_position:
                    in_position = True
                elif angle > 160 and in_position:
                    correct_reps += 1
                    in_position = False
        elif exercise == 'push_up':
            for angle in angles:
                if angle is None:
                    continue
                if angle < 100 and not in_position:
                    in_position = True
                elif angle > 140 and in_position:
                    correct_reps += 1
                    in_position = False
        elif exercise == 'bicep_curl':
            for angle in angles:
                if angle is None:
                    continue
                if angle < 60 and not in_position:
                    in_position = True
                elif angle > 130 and in_position:
                    correct_reps += 1
                    in_position = False
        elif exercise == 'plank':
            correct_reps = sum(1 for angle in angles if angle is not None and 160 <= angle <= 180) // fps
        elif exercise == 'jumping_jack':
            for angle in angles:
                if angle is None:
                    continue
                if angle > 140 and not in_position:
                    in_position = True
                elif angle < 70 and in_position:
                    correct_reps += 1
                    in_position = False
        return correct_reps, incorrect_reps
    except Exception as e:
        logger.error(f"Error counting reps: {e}")
        return 0, 0

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_frame(frame_data, pose, frame_timestamp):
    frame, exercise, width, height, process_width, process_height, frame_count = frame_data
    try:
        if frame is None or frame.size == 0:
            logger.error(f"Frame {frame_count}: Empty or invalid frame")
            return {
                'frame': frame.copy() if frame is not None else np.zeros((height, width, 3), dtype=np.uint8),
                'feedback': ["Invalid frame"],
                'angle': None,
                'error_points': [],
                'frame_count': frame_count,
                'landmarks': None
            }

        aspect_ratio = width / height
        target_aspect = process_width / process_height
        if aspect_ratio > target_aspect:
            new_width = process_width
            new_height = int(process_width / aspect_ratio)
        else:
            new_height = process_height
            new_width = int(process_height * aspect_ratio)
        
        small_frame = cv2.resize(frame, (new_width, new_height))
        pad_w = process_width - new_width
        pad_h = process_height - new_height
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        small_frame = cv2.copyMakeBorder(
            small_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        
        frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = pose.process(frame_rgb)
        frame_rgb.flags.writeable = True
        feedback = []
        knee_angle = None
        hip_angle = None
        error_points = []
        new_frame = frame.copy()

        if results.pose_landmarks:
            logger.debug(f"Frame {frame_count}: Person detected! Found {len(results.pose_landmarks.landmark)} landmarks for exercise {exercise}")
            for lm in results.pose_landmarks.landmark:
                scaled_x = (lm.x * process_width - left) * (width / new_width)
                scaled_y = (lm.y * process_height - top) * (height / new_height)
                lm.x = min(max(scaled_x, 0), width)
                lm.y = min(max(scaled_y, 0), height)
                if lm.visibility < 0.5:
                    logger.debug(f"Frame {frame_count}: Low visibility ({lm.visibility:.2f}) for landmark at scaled ({lm.x:.2f}, {lm.y:.2f})")
            
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            logger.debug(f"Frame {frame_count}: Left shoulder at ({left_shoulder.x:.2f}, {left_shoulder.y:.2f}), Right shoulder at ({right_shoulder.x:.2f}, {right_shoulder.y:.2f})")
            
            if exercise == 'squat':
                feedback, knee_angle, hip_angle, error_points = check_squat_form(landmarks, width, height)
            elif exercise == 'push_up':
                feedback, knee_angle, hip_angle, error_points = check_push_up_form(landmarks, width, height)
            elif exercise == 'bicep_curl':
                feedback, knee_angle, hip_angle, error_points = check_bicep_curl_form(landmarks, width, height)
            elif exercise == 'plank':
                feedback, knee_angle, hip_angle, error_points = check_plank_form(landmarks, width, height)
            elif exercise == 'jumping_jack':
                feedback, knee_angle, hip_angle, error_points = check_jumping_jack_form(landmarks, width, height)
            
            line_color = (255, 0, 0) if not error_points else (0, 0, 255)
            mp_drawing.draw_landmarks(
                new_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 255),
                    thickness=20,
                    circle_radius=8
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=line_color,
                    thickness=5
                )
            )
            
            if exercise == 'squat' and knee_angle is not None and hip_angle is not None:
                avg_knee_x = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * width + 
                             landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * width) / 2
                avg_knee_y = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * height + 
                             landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * height) / 2
                cv2.putText(new_frame, f"{int(knee_angle)}°", (int(avg_knee_x), int(avg_knee_y) - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                avg_hip_x = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * width + 
                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * width) / 2
                avg_hip_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * height + 
                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * height) / 2
                cv2.putText(new_frame, f"{int(hip_angle)}°", (int(avg_hip_x), int(avg_hip_y) - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            for point in error_points:
                cv2.circle(new_frame, point, 15, (0, 0, 255), -1)
            
            logger.debug(f"Frame {frame_count}: Drew landmarks, line_color={line_color}, error_points={len(error_points)}")
        else:
            logger.warning(f"Frame {frame_count}: No pose landmarks detected for exercise {exercise}")
            feedback.append("No person detected in frame")
            cv2.putText(new_frame, "No pose detected - Drawing placeholder", (int(width/2)-150, int(height/2)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.line(new_frame, (0, height//2), (width, height//2), (0, 0, 255), 5)
        
        return {
            'frame': new_frame,
            'feedback': feedback,
            'angle': knee_angle,
            'error_points': error_points,
            'frame_count': frame_count,
            'landmarks': results.pose_landmarks
        }
    except Exception as e:
        logger.error(f"Error processing frame {frame_count}: {e}")
        new_frame = frame.copy() if frame is not None else np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(new_frame, f"Processing error: {str(e)[:30]}", (30, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return {
            'frame': new_frame,
            'feedback': ["Frame processing error"],
            'angle': None,
            'error_points': [],
            'frame_count': frame_count,
            'landmarks': None
        }

# Updated process_video with minimal feedback handling
def process_video(video_data, filename, exercise, user_id):
    start_time = time.time()
    temp_file_path = None
    temp_raw_path = None
    temp_output_path = None
    temp_resized_path = None
    cap = None
    out = None
    pose = None
    try:
        ffmpeg_check = subprocess.run(
            ['ffmpeg', '-codecs'],
            capture_output=True,
            text=True,
            check=True
        )
        if 'libx264' not in ffmpeg_check.stdout:
            logger.warning("FFmpeg does not support libx264; will try h264 encoder")
        
        pose = mp_pose.Pose(
            min_detection_confidence=0.05,
            min_tracking_confidence=0.05,
            model_complexity=2,
            static_image_mode=False
        )
        logger.info(f"Initialized MediaPipe Pose for video processing (exercise: {exercise})")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(video_data)
            temp_file_path = temp_file.name

        temp_resized_path = os.path.join(tempfile.gettempdir(), f"resized_{uuid.uuid4()}.mp4")
        ffmpeg_resize_command = [
            'ffmpeg', '-i', temp_file_path,
            '-vf', 'scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-c:a', 'copy', '-y', temp_resized_path
        ]
        try:
            subprocess.run(
                ffmpeg_resize_command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            logger.info(f"Resized video to {temp_resized_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg resize failed: {e.stderr}")
            raise ValueError("Failed to resize video")

        cap = cv2.VideoCapture(temp_resized_path)
        if not cap.isOpened():
            raise ValueError("Could not open resized video file")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0 or fps == 0:
            raise ValueError("Invalid video: zero frames or FPS")
        logger.info(f"Resized video: {width}x{height}, {fps} FPS, {total_frames} frames")

        if width > 1280 or height > 720:
            raise ValueError("Video resolution exceeds maximum allowed (1280x720)")
        if total_frames / fps > 300:
            raise ValueError("Video duration exceeds maximum allowed (5 minutes)")

        FRAME_SKIP = 2
        process_width, process_height = 640, 480

        frames = []
        frame_count = 0
        frame_timestamp = 0
        timestamp_increment = int(1e6 / fps)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % FRAME_SKIP == 0:
                frames.append((frame, exercise, width, height, process_width, process_height, frame_count))
            frame_count += 1
            frame_timestamp += timestamp_increment
        cap.release()
        cap = None

        relevant_angles = []
        processed_frames = []
        
        max_workers = 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_frame = {
                executor.submit(process_frame, frame_data, pose, frame_timestamp + i * timestamp_increment): frame_data 
                for i, frame_data in enumerate(frames)
            }
            for future in as_completed(future_to_frame):
                try:
                    result = future.result()
                    processed_frames.append(result)
                    if result['angle'] is not None:
                        relevant_angles.append(result['angle'])
                except Exception as e:
                    logger.error(f"Error processing frame batch: {e}")

        processed_frames.sort(key=lambda x: x['frame_count'])

        correct_reps, incorrect_reps = count_reps(exercise, relevant_angles, fps, total_frames)

        all_frames = []
        last_valid_landmarks = None
        last_error_points = []
        
        cap = cv2.VideoCapture(temp_resized_path)
        if not cap.isOpened():
            raise ValueError("Could not reopen resized video file for interpolation")
            
        for i in range(total_frames):
            ret, raw_frame = cap.read()
            if not ret:
                continue
                
            new_frame = raw_frame.copy() if raw_frame is not None else np.zeros((height, width, 3), dtype=np.uint8)
            matching_frame = next((f for f in processed_frames if f['frame_count'] == i), None)
            
            if matching_frame and matching_frame['landmarks']:
                last_valid_landmarks = matching_frame['landmarks']
                last_error_points = matching_frame['error_points']
                new_frame = matching_frame['frame']
                logger.debug(f"Frame {i}: Using processed frame with landmarks")
            elif last_valid_landmarks:
                line_color = (255, 0, 0) if not last_error_points else (0, 0, 255)
                for lm in last_valid_landmarks.landmark:
                    lm.x = min(max(lm.x, 0), width)
                    lm.y = min(max(lm.y, 0), height)
                mp_drawing.draw_landmarks(
                    new_frame,
                    last_valid_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 255, 255),
                        thickness=10,
                        circle_radius=5
                    ),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=line_color,
                        thickness=5
                    )
                )
                for point in last_error_points:
                    cv2.circle(new_frame, point, 15, (0, 0, 255), -1)
                logger.debug(f"Frame {i}: Interpolated landmarks")
            else:
                cv2.putText(new_frame, "Waiting for pose detection...", (int(width/2)-150, int(height/2)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.line(new_frame, (0, height//2), (width, height//2), (0, 0, 255), 5)
                logger.debug(f"Frame {i}: No landmarks available - drew placeholder")
            
            cv2.putText(new_frame, f"Frame: {i}", (width-120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            all_frames.append({
                'frame': new_frame,
                'feedback': matching_frame['feedback'] if matching_frame else [],
                'angle': matching_frame['angle'] if matching_frame else None,
                'error_points': matching_frame['error_points'] if matching_frame else [],
                'frame_count': i,
                'landmarks': matching_frame['landmarks'] if matching_frame else None
            })
        
        cap.release()
        cap = None

        output_filename = f"{uuid.uuid4()}.mp4"
        temp_raw_path = os.path.join(tempfile.gettempdir(), f"raw_{output_filename}")
        temp_output_path = os.path.join(tempfile.gettempdir(), output_filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_raw_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise ValueError("Could not open VideoWriter for raw output")

        most_common_feedback = {}
        
        for result in all_frames:
            frame = result['frame']
            fb_list = result['feedback']
            
            for fb in fb_list:
                if fb != "Starting position detected":  # Exclude starting position feedback
                    most_common_feedback[fb] = most_common_feedback.get(fb, 0) + 1
            
            out.write(frame)
            logger.debug(f"Wrote frame {result['frame_count']}")
        
        out.release()
        out = None

        if not os.path.exists(temp_raw_path) or os.path.getsize(temp_raw_path) == 0:
            raise ValueError("Raw video file is empty or missing")

        raw_s3_key = f"{user_id}/raw_{output_filename}"
        try:
            with open(temp_raw_path, 'rb') as raw_file:
                s3_client.upload_fileobj(
                    raw_file,
                    WASABI_BUCKET_NAME,
                    raw_s3_key,
                    ExtraArgs={'ContentType': 'video/mp4'}
                )
            logger.info(f"Intermediate video uploaded to Wasabi S3: {raw_s3_key}")
            response = s3_client.head_object(Bucket=WASABI_BUCKET_NAME, Key=raw_s3_key)
            uploaded_size = response['ContentLength']
            local_size = os.path.getsize(temp_raw_path)
            if uploaded_size != local_size:
                logger.error(f"Intermediate upload size mismatch: local={local_size}, uploaded={uploaded_size}")
                raise ValueError("Intermediate upload size mismatch")
        except Exception as e:
            logger.error(f"Error uploading intermediate video to Wasabi: {e}")
            raise

        raw_url = get_signed_url(raw_s3_key)
        if not raw_url:
            raise ValueError("Failed to generate signed URL for intermediate video")
        logger.info(f"Generated signed URL for intermediate video: {raw_url}")

        try:
            ffmpeg_command = [
                'ffmpeg', '-i', temp_raw_path,
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                '-vf', 'format=yuv420p',
                '-c:a', 'aac', '-b:a', '128k', '-filter:a', 'volume=0',
                '-y', temp_output_path
            ]
            result = subprocess.run(
                ffmpeg_command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            logger.info(f"FFmpeg stdout: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"libx264 failed: {e.stderr}. Retrying with h264 encoder...")
            ffmpeg_command = [
                'ffmpeg', '-i', temp_raw_path,
                '-c:v', 'h264', '-preset', 'medium', '-crf', '18',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                '-vf', 'format=yuv420p',
                '-c:a', 'aac', '-b:a', '128k', '-filter:a', 'volume=0',
                '-y', temp_output_path
            ]
            result = subprocess.run(
                ffmpeg_command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            logger.info(f"FFmpeg (h264) stdout: {result.stdout}")

        if not os.path.exists(temp_output_path) or os.path.getsize(temp_output_path) == 0:
            logger.error("FFmpeg produced empty or missing output file")
            raise ValueError("FFmpeg conversion failed: empty output")
        
        ffprobe_command = ['ffprobe', '-v', 'error', '-show_format', '-show_streams', temp_output_path]
        ffprobe_result = subprocess.run(
            ffprobe_command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info(f"Output video validated, size: {os.path.getsize(temp_output_path)} bytes")

        s3_key = f"{user_id}/{output_filename}"
        try:
            with open(temp_output_path, 'rb') as video_file:
                s3_client.upload_fileobj(
                    video_file,
                    WASABI_BUCKET_NAME,
                    s3_key,
                    ExtraArgs={'ContentType': 'video/mp4'}
                )
            logger.info(f"Final video uploaded to Wasabi S3: {s3_key}")
            response = s3_client.head_object(Bucket=WASABI_BUCKET_NAME, Key=s3_key)
            uploaded_size = response['ContentLength']
            local_size = os.path.getsize(temp_output_path)
            if uploaded_size != local_size:
                logger.error(f"Upload size mismatch: local={local_size}, uploaded={uploaded_size}")
                raise ValueError("Upload size mismatch")
        except Exception as e:
            logger.error(f"Error uploading final video to Wasabi: {e}")
            raise

        video_url = get_signed_url(s3_key)
        if not video_url:
            raise ValueError("Failed to generate signed URL for final video")
        logger.info(f"Generated signed URL for final video: {video_url}")

        sorted_feedback = sorted(most_common_feedback.items(), key=lambda x: x[1], reverse=True)
        top_feedback = [fb for fb, count in sorted_feedback[:5] if fb != "Correct form" or count > len(all_frames) * 0.5]
        if not top_feedback and "Correct form" in most_common_feedback:
            top_feedback = ["Correct form"]
        logger.info(f"Final feedback for {exercise}: {top_feedback}")

        video_record = {
            'user_id': user_id,
            'exercise': exercise,
            'filename': filename,
            's3_key': s3_key,
            'video_url': video_url,
            'correct_reps': correct_reps,
            'incorrect_reps': incorrect_reps,
            'feedback': top_feedback,
            'uploaded_at': time.time(),
            'duration': total_frames / fps
        }
        videos_collection.insert_one(video_record)

        return {
            'video_url': video_url,
            'raw_url': raw_url,
            'exercise': exercise,
            'correct_reps': correct_reps,
            'incorrect_reps': incorrect_reps,
            'top_feedback': top_feedback,
            'duration': total_frames / fps,
            'processing_time': time.time() - start_time
        }
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        if isinstance(e, cv2.error) and "Insufficient memory" in str(e):
            raise ValueError("Video processing failed due to insufficient memory. Please upload a smaller video (max 1280x720, 5 minutes).")
        raise
    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        if pose is not None:
            pose.close()
            logger.info("Closed MediaPipe Pose instance")
        for temp_file in [temp_file_path, temp_resized_path, temp_raw_path, temp_output_path]:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                    logger.debug(f"Removed temporary file: {temp_file}")
                except Exception as clean_error:
                    logger.error(f"Error removing temporary file {temp_file}: {clean_error}")

# Unchanged routes: upload, get_user_videos, get_video, health_check, chat
@app.route('/upload', methods=['POST'])
@firebase_required
def upload_file():
    try:
        user_id = g.user_id
        
        user = users_collection.find_one({'user_id': user_id})
        if not user:
            user_data = {
                'user_id': user_id,
                'created_at': time.time(),
                'last_active': time.time(),
                'video_count': 0
            }
            users_collection.insert_one(user_data)
            logger.info(f"Created new user record for {user_id}")
        else:
            users_collection.update_one(
                {'user_id': user_id},
                {'$set': {'last_active': time.time()},
                 '$inc': {'video_count': 1}}
            )
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        exercise = request.form.get('exercise', '').lower()
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
            
        if exercise not in EXERCISE_CLASSES:
            return jsonify({'error': f'Exercise type not supported. Choose from: {", ".join(EXERCISE_CLASSES)}'}), 400
        
        video_data = file.read()
        filename = secure_filename(file.filename)
        
        result = process_video(video_data, filename, exercise, user_id)
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/videos', methods=['GET'])
@firebase_required
def get_user_videos():
    try:
        user_id = g.user_id
        logger.info(f"Fetching videos for user_id: {user_id}")
        limit = int(request.args.get('limit', 10))
        offset = int(request.args.get('offset', 0))
        
        user_videos = list(videos_collection.find(
            {'user_id': user_id},
            {'_id': 0}
        ).sort('uploaded_at', -1).skip(offset).limit(limit))
        
        logger.info(f"Found {len(user_videos)} videos for user_id: {user_id}")
        valid_videos = []
        for video in user_videos:
            if 's3_key' not in video:
                logger.warning(f"Skipping video missing s3_key: {video}")
                continue
            if 'video_url' not in video or not video['video_url']:
                video['video_url'] = get_signed_url(video['s3_key'])
                if video['video_url']:
                    videos_collection.update_one(
                        {'s3_key': video['s3_key'], 'user_id': user_id},
                        {'$set': {'video_url': video['video_url']}}
                    )
                    logger.info(f"Updated video {video['s3_key']} with new video_url")
                else:
                    logger.warning(f"Failed to generate signed URL for s3_key: {video['s3_key']}")
                    continue
            valid_videos.append(video)
        
        total = videos_collection.count_documents({'user_id': user_id})
        
        return jsonify({
            'videos': valid_videos,
            'total': total,
            'limit': limit,
            'offset': offset
        }), 200
        
    except Exception as e:
        logger.error(f"Error retrieving videos: {type(e).__name__}: {str(e)}")
        return jsonify({'error': f"Server error: {str(e)}"}), 500

@app.route('/video/<video_id>', methods=['GET'])
@firebase_required
def get_video(video_id):
    try:
        user_id = g.user_id
        video = videos_collection.find_one(
            {'_id': video_id, 'user_id': user_id},
            {'_id': 0}
        )
        
        if not video:
            return jsonify({'error': 'Video not found'}), 404
            
        if 's3_key' not in video:
            logger.warning(f"Video missing s3_key: {video}")
            return jsonify({'error': 'Invalid video data'}), 400
            
        if 'video_url' not in video or not video['video_url']:
            video['video_url'] = get_signed_url(video['s3_key'])
            if video['video_url']:
                videos_collection.update_one(
                    {'s3_key': video['s3_key'], 'user_id': user_id},
                    {'$set': {'video_url': video['video_url']}}
                )
                logger.info(f"Updated video {video['s3_key']} with new video_url")
            else:
                logger.warning(f"Failed to generate signed URL for s3_key: {video['s3_key']}")
                return jsonify({'error': 'Failed to generate video URL'}), 500
        
        return jsonify(video), 200
        
    except Exception as e:
        logger.error(f"Error retrieving video: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'timestamp': time.time()}), 200

@app.route('/chat', methods=['POST'])
@firebase_required
def chat():
    try:
        user_id = g.user_id
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        user_message = data['message']
        if not user_message.strip():
            return jsonify({'error': 'Empty message provided'}), 400
        
        response_text = query_gemini(user_message, user_id)
        
        return jsonify({
            'response': response_text,
            'timestamp': time.time()
        }), 200
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': str(e)}), 500

def query_gemini(prompt, user_id):
    try:
        recent_videos = list(videos_collection.find(
            {'user_id': user_id},
            {'exercise': 1, 'correct_reps': 1, 'feedback': 1, '_id': 0}
        ).sort('uploaded_at', -1).limit(5))
        
        context = "User's recent exercise history:\n"
        if recent_videos:
            for video in recent_videos:
                context += f"- {video['exercise'].replace('_', ' ').title()}: {video['correct_reps']} reps, Feedback: {', '.join(video['feedback'])}\n"
        else:
            context += "- No recent exercise data available.\n"
        
        full_prompt = f"""
        You are a fitness coach AI. Provide personalized exercise and diet suggestions based on the user's query and their exercise history. Keep responses concise, practical, and encouraging. Suggest specific exercises (e.g., squats, push-ups) and simple diet tips (e.g., high-protein meals). Use bullet points for clarity. Avoid using markdown formatting (e.g., no **bold** or *italics*).

        User Query: {prompt}
        {context}
        """
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info(f"Querying Gemini for user {user_id} with prompt: {prompt[:50]}...")
        
        response = model.generate_content(
            full_prompt,
            generation_config={
                'max_output_tokens': 500,
                'temperature': 0.7
            }
        )
        
        logger.info(f"Gemini response received for user {user_id}")
        return response.text.strip()
    
    except Exception as e:
        logger.error(f"Error querying Gemini API: {e}")
        raise ValueError(f"Failed to generate response: {e}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

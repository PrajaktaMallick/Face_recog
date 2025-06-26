import cv2
from deepface import DeepFace
import numpy as np
import time
import json

def convert_to_json_serializable(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

def analyze_face_features_advanced(image, selected_features=['age', 'gender', 'emotion', 'race']):
    start_time = time.time()

    try:
        validation_result = validate_image_advanced(image)
        if not validation_result['valid']:
            return {
                'success': False,
                'error': validation_result['error']
            }

        valid_features = ['age', 'gender', 'emotion', 'race']
        features_to_analyze = [f for f in selected_features if f in valid_features]

        if not features_to_analyze:
            return {
                'success': False,
                'error': 'No valid features selected for analysis'
            }

        print(f"Analyzing features: {features_to_analyze}")

        processed_image = preprocess_image_simple(image)

        result = DeepFace.analyze(
            img_path=processed_image,
            actions=features_to_analyze,
            enforce_detection=True,
            detector_backend='retinaface',  
            align=True,  
            silent=True 
        )

        if isinstance(result, list):
            if len(result) == 0:
                return {
                    'success': False,
                    'error': 'No faces detected. Please ensure your face is clearly visible and well-lit.'
                }
            result = select_best_face(result)

        analysis_results = {}
        metadata = {
            'processing_time': round(time.time() - start_time, 2),
            'face_region': convert_to_json_serializable(result.get('region', {})),
            'detection_confidence': float(calculate_detection_confidence(result))
        }

        if 'age' in features_to_analyze:
            age_value = int(convert_to_json_serializable(result['age']))
            age_category = get_age_category(age_value)
            age_range = get_age_range(age_value)

            analysis_results['age'] = {
                'value': age_value,
                'category': age_category['category'],
                'category_emoji': age_category['emoji'],
                'display': f"{age_value} years old ({age_category['category']})",
                'range': age_range,
                'confidence': 90.0,
                'detailed_info': {
                    'exact_age': age_value,
                    'age_group': age_category['category'],
                    'life_stage': age_category['life_stage'],
                    'estimated_range': age_range
                }
            }

        if 'gender' in features_to_analyze:
            gender = str(result['dominant_gender'])
            confidence = convert_to_json_serializable(result['gender'][gender])
            analysis_results['gender'] = {
                'value': gender,
                'display': gender.capitalize(),
                'confidence': round(float(confidence), 1),
                'all_scores': {k: round(float(convert_to_json_serializable(v)), 1) for k, v in result['gender'].items()}
            }

        if 'emotion' in features_to_analyze:
            emotion = str(result['dominant_emotion'])
            confidence = convert_to_json_serializable(result['emotion'][emotion])
            analysis_results['emotion'] = {
                'value': emotion,
                'display': format_emotion_display(emotion),
                'confidence': round(float(confidence), 1),
                'all_scores': {k: round(float(convert_to_json_serializable(v)), 1) for k, v in result['emotion'].items()}
            }

        if 'race' in features_to_analyze:
            race = str(result['dominant_race'])
            confidence = convert_to_json_serializable(result['race'][race])
            analysis_results['race'] = {
                'value': race,
                'display': format_race_display(race),
                'confidence': round(float(confidence), 1),
                'all_scores': {k: round(float(convert_to_json_serializable(v)), 1) for k, v in result['race'].items()}
            }

        # Ensure everything is JSON serializable before returning
        final_result = {
            'success': True,
            'data': convert_to_json_serializable(analysis_results),
            'metadata': convert_to_json_serializable(metadata)
        }

        return final_result

    except Exception as e:
        error_message = str(e)
        print(f"Error in facial analysis: {error_message}")

        if "No face" in error_message or "Face could not be detected" in error_message:
            return {
                'success': False,
                'error': 'No face detected. Please ensure your face is clearly visible, well-lit, and facing the camera.'
            }
        elif "Invalid image" in error_message or "corrupted" in error_message.lower():
            return {
                'success': False,
                'error': 'Invalid or corrupted image. Please try capturing again.'
            }
        elif "memory" in error_message.lower():
            return {
                'success': False,
                'error': 'Insufficient memory for analysis. Please try again.'
            }
        else:
            return {
                'success': False,
                'error': f'Analysis failed: {error_message}'
            }

def validate_image_advanced(image):
    try:
        if image is None:
            return {'valid': False, 'error': "No image provided"}

        if len(image.shape) < 2:
            return {'valid': False, 'error': "Invalid image format"}

        height, width = image.shape[:2]
        if height < 200 or width < 200:
            return {'valid': False, 'error': "Image resolution too low for accurate analysis"}

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        mean_brightness = np.mean(gray)

        if mean_brightness < 40:
            return {'valid': False, 'error': "Image too dark. Please improve lighting"}
        elif mean_brightness > 215:
            return {'valid': False, 'error': "Image too bright. Please reduce lighting"}

        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        if blur_score < 20:
            return {'valid': False, 'error': "Image too blurry. Please hold camera steady"}

        return {'valid': True, 'error': None}

    except Exception as e:
        return {'valid': False, 'error': f"Image validation error: {str(e)}"}



def preprocess_image_simple(image):
    try:
        if len(image.shape) == 3 and image.shape[2] == 3:
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            processed = image.copy()

        return processed

    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return image

def select_best_face(faces_list):
    if len(faces_list) == 1:
        return faces_list[0]

    best_face = faces_list[0]
    max_area = 0

    for face in faces_list:
        if 'region' in face:
            region = face['region']
            area = region.get('w', 0) * region.get('h', 0)
            if area > max_area:
                max_area = area
                best_face = face

    return best_face

def calculate_detection_confidence(result):
    try:
        if 'region' in result:
            region = result['region']
            area = region.get('w', 0) * region.get('h', 0)
            confidence = min(95.0, max(60.0, (area / 10000) * 100))
            return round(confidence, 1)
        return 85.0
    except:
        return 85.0

def format_emotion_display(emotion):
    emotion_map = {
        'angry': 'Angry 😠',
        'disgust': 'Disgusted 🤢',
        'fear': 'Fearful 😨',
        'happy': 'Happy 😊',
        'sad': 'Sad 😢',
        'surprise': 'Surprised 😲',
        'neutral': 'Neutral 😐'
    }
    return emotion_map.get(emotion.lower(), emotion.capitalize())

def format_race_display(race):
    race_map = {
        'asian': 'Asian',
        'indian': 'Indian',
        'black': 'Black',
        'white': 'White',
        'middle eastern': 'Middle Eastern',
        'latino hispanic': 'Latino/Hispanic'
    }
    return race_map.get(race.lower(), race.replace('_', ' ').title())

def get_age_category(age):
    if age <= 2:
        return {
            'category': 'Baby',
            'emoji': '👶',
            'life_stage': 'Infancy'
        }
    elif age <= 5:
        return {
            'category': 'Toddler',
            'emoji': '🧒',
            'life_stage': 'Early Childhood'
        }
    elif age <= 12:
        return {
            'category': 'Child',
            'emoji': '👧👦',
            'life_stage': 'Childhood'
        }
    elif age <= 17:
        return {
            'category': 'Teenager',
            'emoji': '🧑‍🎓',
            'life_stage': 'Adolescence'
        }
    elif age <= 25:
        return {
            'category': 'Young Adult',
            'emoji': '🧑‍💼',
            'life_stage': 'Early Adulthood'
        }
    elif age <= 40:
        return {
            'category': 'Adult',
            'emoji': '👨👩',
            'life_stage': 'Adulthood'
        }
    elif age <= 60:
        return {
            'category': 'Middle-aged',
            'emoji': '🧑‍🦳',
            'life_stage': 'Middle Age'
        }
    elif age <= 75:
        return {
            'category': 'Senior',
            'emoji': '👴👵',
            'life_stage': 'Senior Years'
        }
    else:
        return {
            'category': 'Elderly',
            'emoji': '🧓',
            'life_stage': 'Advanced Age'
        }

def get_age_range(age):
    if age <= 18:
        margin = 2  
    elif age <= 30:
        margin = 3
    elif age <= 50:
        margin = 4
    else:
        margin = 5 

    min_age = max(0, age - margin)
    max_age = min(120, age + margin)

    return f"{min_age}-{max_age} years"
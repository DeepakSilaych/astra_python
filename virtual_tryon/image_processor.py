import openai
from PIL import Image, ImageFilter, ImageOps, ImageDraw, ImageEnhance
import base64
import io
import cv2
import mediapipe as mp
import numpy as np
import os
import glob
import re
import asyncio
import aiohttp
import tempfile
from typing import Optional, Dict, Any, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

openai.api_key = os.getenv("OPENAI_API_KEY")

def remove_background_rembg(image_path: str) -> Optional[Image.Image]:
    try:
        from rembg import remove
        img = Image.open(image_path).convert("RGBA")
        result = remove(img)
        return result
    except ImportError:
        print("rembg library not available, falling back to edge detection")
        return None
    except Exception as e:
        print(f"rembg background removal failed: {str(e)}")
        return None

def remove_background_color_based(image_path: str) -> Optional[Image.Image]:
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        masks = []
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        masks.append(white_mask)
        black_lower = np.array([0, 0, 0])
        black_upper = np.array([180, 255, 50])
        black_mask = cv2.inRange(hsv, black_lower, black_upper)
        masks.append(black_mask)
        gray_lower = np.array([0, 0, 100])
        gray_upper = np.array([180, 30, 200])
        gray_mask = cv2.inRange(hsv, gray_lower, gray_upper)
        masks.append(gray_mask)
        background_mask = cv2.bitwise_or(cv2.bitwise_or(white_mask, black_mask), gray_mask)
        foreground_mask = cv2.bitwise_not(background_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
        result = cv2.bitwise_and(img, img, mask=foreground_mask)
        result_rgba = cv2.cvtColor(result, cv2.COLOR_BGR2RGBA)
        result_rgba[:, :, 3] = foreground_mask
        pil_result = Image.fromarray(result_rgba)
        return pil_result
    except Exception as e:
        print(f"Color-based background removal failed: {str(e)}")
        return None

def remove_background_improved_edge(image_path: str) -> Optional[Image.Image]:
    try:
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        edges_canny = cv2.Canny(gray, 30, 100)
        edges_sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        edges_sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges_sobel = np.sqrt(edges_sobel_x**2 + edges_sobel_y**2)
        edges_sobel = np.uint8(edges_sobel * 255 / np.max(edges_sobel))
        combined_edges = cv2.bitwise_or(edges_canny, edges_sobel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges_dilated = cv2.dilate(combined_edges, kernel, iterations=2)
        edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [largest_contour], 255)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        result = cv2.bitwise_and(img_array, img_array, mask=mask)
        result_rgba = cv2.cvtColor(result, cv2.COLOR_RGB2RGBA)
        result_rgba[:, :, 3] = mask
        pil_result = Image.fromarray(result_rgba)
        return pil_result
    except Exception as e:
        print(f"Improved edge detection failed: {str(e)}")
        return None

def remove_background_adaptive(image_path: str) -> Optional[Image.Image]:
    methods = [
        ("rembg", remove_background_rembg),
        ("color_based", remove_background_color_based),
        ("improved_edge", remove_background_improved_edge)
    ]
    for method_name, method_func in methods:
        try:
            result = method_func(image_path)
            if result:
                print(f"Background removal successful using {method_name}")
                return result
        except Exception as e:
            print(f"Method {method_name} failed: {str(e)}")
            continue
    print("All background removal methods failed, using original image")
    return None

def crop_jewelry_image(image_path, jewelry_type):
    if image_path is None or not os.path.exists(image_path):
        return None
    try:
        img = Image.open(image_path).convert("RGB")
        original_size = img.size
        if img.mode == "RGBA":
            alpha = img.split()[-1]
            alpha_array = np.array(alpha)
            non_zero = np.nonzero(alpha_array)
            if len(non_zero[0]) > 0:
                y_min, y_max = np.min(non_zero[0]), np.max(non_zero[0])
                x_min, x_max = np.min(non_zero[1]), np.max(non_zero[1])
                padding = max(5, min(x_max - x_min, y_max - y_min) // 20)
                left = max(0, x_min - padding)
                top = max(0, y_min - padding)
                right = min(original_size[0], x_max + padding)
                bottom = min(original_size[1], y_max + padding)
                cropped_img = img.crop((left, top, right, bottom))
            else:
                cropped_img = img
        else:
            img_array = np.array(img)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            kernel = np.ones((3, 3), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=2)
            contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                padding = max(5, min(w, h) // 20)
                left = max(0, x - padding)
                top = max(0, y - padding)
                right = min(original_size[0], x + w + padding)
                bottom = min(original_size[1], y + h + padding)
                cropped_img = img.crop((left, top, right, bottom))
            else:
                cropped_img = img
        return cropped_img
    except Exception as e:
        print(f"Error in crop_jewelry_image: {str(e)}")
        return None

def enhance_jewelry_image(image_path: str) -> Optional[Image.Image]:
    try:
        img = Image.open(image_path).convert("RGBA")
        img_sharp = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        enhancer = ImageEnhance.Contrast(img_sharp)
        img_enhanced = enhancer.enhance(1.1)
        return img_enhanced
    except Exception as e:
        print(f"Image enhancement failed: {str(e)}")
        return None

def enhance_jewelry_image_from_pil(img: Image.Image) -> Optional[Image.Image]:
    try:
        img_rgba = img.convert("RGBA")
        img_sharp = img_rgba.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        enhancer = ImageEnhance.Contrast(img_sharp)
        img_enhanced = enhancer.enhance(1.1)
        return img_enhanced
    except Exception as e:
        print(f"Image enhancement failed: {str(e)}")
        return None

def get_image_height_cm_mediapipe_fallback(model_image_path, jewelry_type):
    """Estimate image height using MediaPipe landmark distances as fallback"""
    try:
        img = cv2.imread(model_image_path)
        if img is None:
            return None
        
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        
        # Standard human proportions (in cm) for a 170cm person
        standard_eye_distance = 6.5  # cm between eyes
        standard_shoulder_width = 45  # cm shoulder width
        standard_nose_to_lip = 2.5  # cm nose to lip
        standard_ear_to_ear = 15  # cm ear to ear
        
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh, mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.3
        ) as pose:
            
            face_results = face_mesh.process(rgb)
            pose_results = pose.process(rgb)
            
            measurements = []
            
            # Face measurements
            if face_results.multi_face_landmarks:
                landmarks = face_results.multi_face_landmarks[0].landmark
                
                # Eye distance (landmarks 33 and 263)
                left_eye = landmarks[33]
                right_eye = landmarks[263]
                eye_distance_px = np.sqrt((left_eye.x - right_eye.x)**2 + (left_eye.y - right_eye.y)**2) * width
                if eye_distance_px > 10:  # Minimum threshold
                    eye_scale = standard_eye_distance / eye_distance_px
                    measurements.append(eye_scale)
                
                # Nose to lip distance (landmarks 1 and 13)
                nose_tip = landmarks[1]
                upper_lip = landmarks[13]
                nose_lip_distance_px = np.sqrt((nose_tip.x - upper_lip.x)**2 + (nose_tip.y - upper_lip.y)**2) * height
                if nose_lip_distance_px > 5:  # Minimum threshold
                    nose_lip_scale = standard_nose_to_lip / nose_lip_distance_px
                    measurements.append(nose_lip_scale)
                
                # Ear to ear distance (landmarks 234 and 454)
                left_ear = landmarks[234]
                right_ear = landmarks[454]
                ear_distance_px = np.sqrt((left_ear.x - right_ear.x)**2 + (left_ear.y - right_ear.y)**2) * width
                if ear_distance_px > 20:  # Minimum threshold
                    ear_scale = standard_ear_to_ear / ear_distance_px
                    measurements.append(ear_scale)
            
            # Body measurements
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                
                # Shoulder width (landmarks 11 and 12)
                left_shoulder = landmarks[11]
                right_shoulder = landmarks[12]
                if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
                    shoulder_distance_px = np.sqrt((left_shoulder.x - right_shoulder.x)**2 + (left_shoulder.y - right_shoulder.y)**2) * width
                    if shoulder_distance_px > 30:  # Minimum threshold
                        shoulder_scale = standard_shoulder_width / shoulder_distance_px
                        measurements.append(shoulder_scale)
            
            if measurements:
                # Use median scale to avoid outliers
                median_scale = np.median(measurements)
                image_height_cm = height * median_scale
                
                print(f"[MediaPipe] Estimated image height: {image_height_cm:.1f}cm using {len(measurements)} measurements")
                print(f"[MediaPipe] Scale factors: {[f'{m:.3f}' for m in measurements]}")
                
                return image_height_cm
        
        return None
        
    except Exception as e:
        print(f"MediaPipe height estimation failed: {str(e)}")
        return None

def get_image_height_cm(model_image_path, jewelry_type):
    """Estimate model image height using OpenAI o3 vision model with MediaPipe fallback"""
    try:
        return get_image_height_cm_mediapipe_fallback(model_image_path, jewelry_type)
        with open(model_image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        prompt = f"""`You are an Export Forensic Expert. 
        Analyze the model image, postion and spacing of body landmarks.
        Estimate image of the image by visible person.


        Chain of thought:
        1. Assume model is 140 cm tall.
        2. Analyze the model image, postion and spacing of body landmarks.
        3. Estimate size of the image by the amount of visible person and space it takes in the image.

        return this format separated by ||
        "thought process" || height in cm
        Do not include any other text in your response. 
`
        Example:
        "Distance between neck and shoulder is 10 cm. Neck is 10 cm from the top of the image. So, height of the person is 140 cm. Since she is visible till shoulder then the height of the image should be 40 cm. " || "40"

        Do not include any other text in your response. 
        """

        response = openai.chat.completions.create(
            model="o3",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                    ]
                }
            ],
        )
        
        
        height_str = response.choices[0].message.content.strip()
        print(f"Model Image Height: {height_str}")

        height_value = height_str.split("||")[1].strip()
        if height_value.startswith('"') and height_value.endswith('"'):
            height_value = height_value[1:-1]
        height_cm = float(height_value)
        
        return height_cm
    except Exception as e:
        print(f"OpenAI height estimation failed: {str(e)}")
        
        # Try MediaPipe fallback first
        mediapipe_height = get_image_height_cm_mediapipe_fallback(model_image_path, jewelry_type)
        if mediapipe_height:
            return mediapipe_height
        
        # Geometric fallback as last resort
        img = Image.open(model_image_path)
        width, height = img.size
        
        jewelry_type_lower = jewelry_type.lower()
        
        if jewelry_type_lower in ['earring', 'earrings']:
            if height > width:
                return 25
            else:
                return 20
        elif jewelry_type_lower in ['necklace', 'necklaces', 'pendant']:
            if height > width:
                return 35
            else:
                return 30
        elif jewelry_type_lower in ['bracelet', 'bracelets', 'bangles']:
            if height > width:
                return 25
            else:
                return 20
        elif jewelry_type_lower in ['ring', 'rings']:
            if height > width:
                return 20
            else:
                return 15
        else:
            return 25

def get_jewelry_size_cm(jewelry_size_input, jewelry_image_path=None, jewelry_type="jewelry"):
    """Get jewelry size in cm from user input or predict from image using OpenAI"""
    if jewelry_size_input:
        try:
            prompt = (
                f"Given the following description for a {jewelry_type}, extract the vertical height of this jewelry in centimeters."
                f"If no units are mentioned, assume mm."
                f"Return only the number (in cm). Description: "
                f"{jewelry_size_input}"
            )
            response = openai.chat.completions.create(
                model="o3",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt}
                        ]
                    }
                ],
            )
            height_str = response.choices[0].message.content.strip()
            number_match = re.search(r'(\d+\.?\d*)', height_str)
            if number_match:
                return float(number_match.group(1))
        except Exception as e:
            pass
    
    if jewelry_image_path and os.path.exists(jewelry_image_path):
        try:
            with open(jewelry_image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
    
            prompt = """Estimate the vertical height of this jewelry in centimeters. Return only the number."""
    
            response = openai.chat.completions.create(
                model="o3",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                        ]
                    }
                ],
            )
            
            height_str = response.choices[0].message.content.strip()
            height_cm = float(''.join(filter(str.isdigit, height_str.split('.')[0])))
            return height_cm
        except:
            pass
    
    return 2.0

def calculate_jewelry_size_mm(jewelry_size_str):
    """Parse jewelry size string and convert to millimeters"""
    if not jewelry_size_str:
        return 20
    
    size_lower = jewelry_size_str.lower().strip()
    if 'cm' in size_lower:
        try:
            number_match = re.search(r'(\d+\.?\d*)', size_lower)
            if number_match:
                return float(number_match.group(1)) * 10
            else:
                return 20
        except:
            return 20
    elif 'mm' in size_lower:
        try:
            number_match = re.search(r'(\d+\.?\d*)', size_lower)
            if number_match:
                return float(number_match.group(1))
            else:
                return 20
        except:
            return 20
    else:
        return 20

def detect_neck_mediapipe(model_image_path):
    """Detect neck position using MediaPipe BlazePose"""
    img = cv2.imread(model_image_path)
    if img is None:
        return None, None
    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    
    try:
        with mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.3
        ) as pose:
            pose_results = pose.process(rgb)
            
            if not pose_results.pose_landmarks:
                return None, None
            
            landmarks = pose_results.pose_landmarks.landmark
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
        
            shoulder_center_x = int((left_shoulder.x + right_shoulder.x) * width / 2)
            shoulder_center_y = int((left_shoulder.y + right_shoulder.y) * height / 2)
            necklace_offset = int(0.08 * height)
        
            position = {
                'x': shoulder_center_x, 
                'y': shoulder_center_y - necklace_offset, 
                'confidence': 0.9, 
                'method': 'mediapipe_blazepose'
            }
            
            landmarks_data = {'pose_results': pose_results, 'jewelry_position': position}
            return position, landmarks_data
            
    except Exception as e:
        print(f"MediaPipe neck detection failed: {str(e)}")
        return None, None

def detect_wrist_mediapipe(model_image_path):
    """Detect wrist position using MediaPipe BlazePose and Hands"""
    img = cv2.imread(model_image_path)
    if img is None:
        return None, None
    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    
    try:
        with mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.3
        ) as pose, mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        ) as hands:
            
            pose_results = pose.process(rgb)
            hand_results = hands.process(rgb)
            
            position = None
            
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                left_wrist = landmarks[15]
                right_wrist = landmarks[16]
                
                if left_wrist.visibility > right_wrist.visibility and left_wrist.visibility > 0.5:
                    position = {
                        'x': int(left_wrist.x * width), 
                        'y': int(left_wrist.y * height), 
                        'confidence': float(left_wrist.visibility), 
                        'method': 'mediapipe_blazepose'
                    }
                elif right_wrist.visibility > 0.5:
                    position = {
                        'x': int(right_wrist.x * width), 
                        'y': int(right_wrist.y * height), 
                        'confidence': float(right_wrist.visibility), 
                        'method': 'mediapipe_blazepose'
                    }
            
            if not position and hand_results.multi_hand_landmarks:
                hand_landmarks = hand_results.multi_hand_landmarks[0]
                wrist = hand_landmarks.landmark[0]
                position = {
                    'x': int(wrist.x * width), 
                    'y': int(wrist.y * height), 
                    'confidence': 0.7, 
                    'method': 'mediapipe_hands'
                }
            
            if position:
                landmarks_data = {
                    'pose_results': pose_results, 
                    'hand_results': hand_results, 
                    'jewelry_position': position
                }
                return position, landmarks_data
            else:
                return None, None
                
    except Exception as e:
        print(f"MediaPipe wrist detection failed: {str(e)}")
        return None, None

def detect_finger_mediapipe(model_image_path):
    img = cv2.imread(model_image_path)
    if img is None:
        return None, None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    try:
        with mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        ) as hands:
            hand_results = hands.process(rgb)
            if not hand_results.multi_hand_landmarks:
                return None, None
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            ring_finger_mcp = hand_landmarks.landmark[13]
            ring_finger_tip = hand_landmarks.landmark[16]
            ring_x = int((ring_finger_mcp.x + ring_finger_tip.x) * width / 2)
            ring_y = int((ring_finger_mcp.y + ring_finger_tip.y) * height / 2)
            position = {
                'x': ring_x, 
                'y': ring_y, 
                'confidence': 0.8, 
                'method': 'mediapipe_hands'
            }
            landmarks_data = {'hand_results': hand_results, 'jewelry_position': position}
            return position, landmarks_data
    except Exception as e:
        print(f"MediaPipe finger detection failed: {str(e)}")
        return None, None

def convert_numpy_types(obj):
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def get_jewelry_placement_position(model_image_path, jewelry_type):
    position = None
    landmarks_data = None
    
    jewelry_type_lower = jewelry_type.lower()
    
    if jewelry_type_lower in ['earring', 'earrings']:
        position = get_earring_position_haar(model_image_path)
        if position:
            print(f"[Landmark] Ear detected using Haar cascade - {position}")
            return position
        
        position = get_earring_position_mediapipe(model_image_path)
        if position:
            print(f"[Landmark] Ear detected using MediaPipe Face Mesh - {position}")
            return position
            
    elif jewelry_type_lower in ['necklace', 'necklaces', 'pendant']:
        position, landmarks_data = detect_neck_mediapipe(model_image_path)
    elif jewelry_type_lower in ['bracelet', 'bracelets', 'bangles']:
        position, landmarks_data = detect_wrist_mediapipe(model_image_path)
    elif jewelry_type_lower in ['ring', 'rings']:
        position, landmarks_data = detect_finger_mediapipe(model_image_path)
    
    if position:
        return position
    
    # Geometric fallback
    try:
        model_img = Image.open(model_image_path)
        width, height = model_img.size

        if jewelry_type_lower in ['necklace', 'necklaces', 'pendant']:
            fallback_pos = {'x': width // 2, 'y': height // 3, 'confidence': 0.3, 'method': 'geometric_fallback'}
        elif jewelry_type_lower in ['earring', 'earrings']:
            fallback_pos = {'x': width // 4, 'y': height // 4, 'confidence': 0.3, 'method': 'geometric_fallback'}
        elif jewelry_type_lower in ['bracelet', 'bracelets', 'bangles']:
            fallback_pos = {'x': width // 2, 'y': int(height * 0.7), 'confidence': 0.3, 'method': 'geometric_fallback'}
        elif jewelry_type_lower in ['ring', 'rings']:
            fallback_pos = {'x': int(width * 0.6), 'y': int(height * 0.8), 'confidence': 0.3, 'method': 'geometric_fallback'}
        else:
            fallback_pos = {'x': width // 2, 'y': height // 2, 'confidence': 0.2, 'method': 'geometric_fallback'}
    
        print(f"[Landmark] Using geometric fallback - {fallback_pos}")
        return fallback_pos
        
    except Exception as e:
        print(f"Geometric fallback error: {e}")
        return None

def calculate_zoom_factor(model_img, jewelry_img, landmark_position, jewelry_type):
    """Calculate optimal zoom factor based on jewelry size relative to model image"""
    model_width, model_height = model_img.size
    jewelry_width, jewelry_height = jewelry_img.size
    
    model_area = model_width * model_height
    jewelry_area = jewelry_width * jewelry_height
    
    jewelry_ratio = jewelry_area / model_area
    
    max_zoom_ratio = 0.05
    
    if jewelry_ratio >= max_zoom_ratio:
        zoom_factor = 0.8
        print(f"[Zoom] Jewelry ratio {jewelry_ratio:.3f} >= {max_zoom_ratio}, applying max zoom")
    else:
        zoom_factor = 1.0 - (jewelry_ratio * 10)
        zoom_factor = max(0.8, min(0.95, zoom_factor))
        print(f"[Zoom] Jewelry ratio {jewelry_ratio:.3f} < {max_zoom_ratio}, calculated zoom: {zoom_factor:.3f}")
    
    return zoom_factor

def apply_zoom_to_landmark(model_img, landmark_position, zoom_factor):
    """Apply zoom transformation towards the landmark position"""
    original_width, original_height = model_img.size
    zoom_center_x = landmark_position['x']
    zoom_center_y = landmark_position['y']
    
    # Calculate new dimensions after zoom
    new_width = int(original_width * zoom_factor)
    new_height = int(original_height * zoom_factor)
    
    # Calculate crop box to center on landmark
    crop_left = max(0, zoom_center_x - new_width // 2)
    crop_top = max(0, zoom_center_y - new_height // 2)
    crop_right = min(original_width, crop_left + new_width)
    crop_bottom = min(original_height, crop_top + new_height)
    
    # Crop and resize to original size
    cropped = model_img.crop((crop_left, crop_top, crop_right, crop_bottom))
    zoomed_model = cropped.resize((original_width, original_height), Image.BICUBIC)
    
    return zoomed_model

def get_earring_position_mediapipe(image_path):
    """Detect ear position using MediaPipe Face Mesh and return ear lobe position for earring placement"""
    img = cv2.imread(image_path)
    if img is None:
                return None
    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    
    try:
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            results = face_mesh.process(rgb)
            
            if not results.multi_face_landmarks:
                return None
    
            landmarks = results.multi_face_landmarks[0].landmark
            
            # MediaPipe Face Mesh ear lobe landmarks
            # Left ear lobe: 234, Right ear lobe: 454
            left_ear_lobe = landmarks[234]
            right_ear_lobe = landmarks[454]
            
            # Convert to pixel coordinates
            left_x = int(left_ear_lobe.x * width)
            left_y = int(left_ear_lobe.y * height)
            right_x = int(right_ear_lobe.x * width)
            right_y = int(right_ear_lobe.y * height)
            
            # Choose the ear that's more visible (closer to center)
            center_x = width // 2
            left_distance = abs(left_x - center_x)
            right_distance = abs(right_x - center_x)
            
            if left_distance < right_distance:
                ear_x, ear_y = left_x, left_y
                ear_side = "left"
            else:
                ear_x, ear_y = right_x, right_y
                ear_side = "right"
            
            # Calculate confidence based on landmark visibility
            confidence = 0.8  # MediaPipe landmarks are generally reliable
            
            position = {
                'x': ear_x, 
                'y': ear_y, 
                'confidence': confidence, 
                'method': 'mediapipe_face_mesh'
            }
            
            return position
            
    except Exception as e:
        print(f"MediaPipe ear detection error: {e}")
        return None

def get_earring_position_haar(image_path):
    """Detect ear position using Haar cascades and return bottom of ear for earring placement"""
    cascade_dir = "virtual_tryon/cascade_files"
    left_ear_cascade_path = os.path.join(cascade_dir, 'haarcascade_mcs_leftear.xml')
    right_ear_cascade_path = os.path.join(cascade_dir, 'haarcascade_mcs_rightear.xml')
    
    if not os.path.exists(cascade_dir):
        os.makedirs(cascade_dir)
    
    if not os.path.exists(left_ear_cascade_path) or not os.path.exists(right_ear_cascade_path):
        print("Haar cascade files not found. Please download from OpenCV repository.")
        return None
    
    try:
        left_ear_cascade = cv2.CascadeClassifier(left_ear_cascade_path)
        right_ear_cascade = cv2.CascadeClassifier(right_ear_cascade_path)
        
        if left_ear_cascade.empty() or right_ear_cascade.empty():
            return None
        
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape[:2]
        
        left_ears = left_ear_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        right_ears = right_ear_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        best_ear = None
        best_confidence = 0
        ear_side = None
        
        for (x, y, w, h) in left_ears:
            ear_area = w * h
            confidence = ear_area / (width * height)
            
            if confidence > best_confidence:
                best_confidence = confidence
                ear_x = x + int(w * 0.75)
                ear_y = y + h - 10
                best_ear = {'x': ear_x, 'y': ear_y, 'confidence': confidence, 'method': 'haar_cascade'}
                ear_side = "left"
        
        for (x, y, w, h) in right_ears:
            ear_area = w * h
            confidence = ear_area / (width * height)
            
            if confidence > best_confidence:
                best_confidence = confidence
                ear_x = x + int(w * 0.25)
                ear_y = y + h - 10
                best_ear = {'x': ear_x, 'y': ear_y, 'confidence': confidence, 'method': 'haar_cascade'}
                ear_side = "right"
        
        if best_ear:
            return best_ear
        else:
            return None
            
    except Exception as e:
        print(f"Haar cascade detection error: {e}")
        return None

class ImageProcessor:
    def __init__(self):
        pass

    async def process_virtual_try_on(
        self,
        jewelry_image_url: str,
        model_image_url: str,
        jewelry_type: str,
        jewelry_subtype: str,
        jewelry_size: Optional[Dict[str, Any]] = None
    ) -> dict:
        try:
            print(f"[ImageProcessor] Processing {jewelry_type}/{jewelry_subtype} virtual try-on")
            if jewelry_image_url.startswith('file://'):
                jewelry_image_path = jewelry_image_url[7:]
                with open(jewelry_image_path, 'rb') as f:
                    jewelry_data = f.read()
            else:
                async with aiohttp.ClientSession() as session:
                    async with session.get(jewelry_image_url) as response:
                        if response.status != 200:
                            raise Exception(f"Failed to download jewelry image: {response.status}")
                        jewelry_data = await response.read()
            
            if model_image_url.startswith('file://'):
                model_image_path = model_image_url[7:]
                with open(model_image_path, 'rb') as f:
                    model_data = f.read()
            else:
                async with aiohttp.ClientSession() as session:
                    async with session.get(model_image_url) as response:
                        if response.status != 200:
                            raise Exception(f"Failed to download model image: {response.status}")
                        model_data = await response.read()
            print(f"[ImageProcessor] Step 1: Downloaded images - jewelry: {len(jewelry_data)} bytes, model: {len(model_data)} bytes")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as jewelry_temp:
                jewelry_temp.write(jewelry_data)
                jewelry_temp_path = jewelry_temp.name
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as model_temp:
                model_temp.write(model_data)
                model_temp_path = model_temp.name
            print(f"[ImageProcessor] Step 2: Created temp files - jewelry: {jewelry_temp_path}, model: {model_temp_path}")
            
            model_img = Image.open(model_temp_path)
            original_width, original_height = model_img.size
            landmark_position = get_jewelry_placement_position(model_temp_path, jewelry_type)
            print(f"[ImageProcessor] Step 3: Detected landmark position - {landmark_position}")
            
            cropped_jewelry = crop_jewelry_image(jewelry_temp_path, jewelry_type)
            if cropped_jewelry is None:
                cropped_jewelry = Image.open(jewelry_temp_path).convert("RGB")
            print(f"[ImageProcessor] Step 4: Initial jewelry processing - size: {cropped_jewelry.size}")
            
            initial_zoom_factor = 0.7
            initial_zoomed_model = apply_zoom_to_landmark(model_img, landmark_position, initial_zoom_factor)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as initial_zoomed_temp:
                initial_zoomed_model.save(initial_zoomed_temp.name, format='PNG')
                initial_zoomed_temp_path = initial_zoomed_temp.name
            print(f"[ImageProcessor] Step 5: Applied initial 20% zoom - {original_width}x{original_height} -> {initial_zoomed_model.size} (zoom factor: {initial_zoom_factor:.1f})")
            
            model_height_cm = get_image_height_cm(initial_zoomed_temp_path, jewelry_type)
            jewelry_size_cm = get_jewelry_size_cm(jewelry_size, jewelry_temp_path, jewelry_type)
            print(f"[ImageProcessor] Step 6: Calculated sizes - model height: {model_height_cm}cm, jewelry size: {jewelry_size_cm}cm")
            
            initial_zoomed_model_img = Image.open(initial_zoomed_temp_path)
            model_width_px, model_height_px = initial_zoomed_model_img.size
            jewelry_width_px, jewelry_height_px = cropped_jewelry.size
            model_cm_to_px = model_height_cm / model_height_px
            jewelry_cm_to_px = jewelry_size_cm / jewelry_height_px
            scale_factor = jewelry_cm_to_px / model_cm_to_px
            print(f"[ImageProcessor] Step 7: Initial jewelry scaling, model cm to px: {model_cm_to_px}, jewelry cm to px: {jewelry_cm_to_px}, scale factor: {scale_factor:.3f}")
            
            new_jewelry_width = int(jewelry_width_px * scale_factor)
            new_jewelry_height = int(jewelry_height_px * scale_factor)
            resized_jewelry = cropped_jewelry.resize((new_jewelry_width, new_jewelry_height), Image.BICUBIC)
            print(f"[ImageProcessor] Step 8: Resized jewelry - {jewelry_width_px}x{jewelry_height_px} -> {new_jewelry_width}x{new_jewelry_height}")
            
        
            landmark_position = get_jewelry_placement_position(initial_zoomed_temp_path, jewelry_type)
            if not isinstance(landmark_position, dict):
                model_img = Image.open(initial_zoomed_temp_path)
                width, height = model_img.size
            print(f"[ImageProcessor] Step 11: Detected landmark position - {landmark_position}")
            
            processed_jewelry_url = None
            processed_jewelry_with_bg_url = None
            
            try:
                import base64
                
                jewelry_bytes = io.BytesIO()
                resized_jewelry.save(jewelry_bytes, format='PNG')
                jewelry_bytes.seek(0)
                jewelry_base64 = base64.b64encode(jewelry_bytes.getvalue()).decode('utf-8')
                
                processed_jewelry_url = f"data:image/png;base64,{jewelry_base64}"
                processed_jewelry_with_bg_url = f"data:image/png;base64,{jewelry_base64}"
                
                model_bytes = io.BytesIO()
                initial_zoomed_model_img.save(model_bytes, format='PNG')
                model_bytes.seek(0)
                model_base64 = base64.b64encode(model_bytes.getvalue()).decode('utf-8')
                model_image_url = f"data:image/png;base64,{model_base64}"
                
                print(f"[ImageProcessor] Step 12: Created base64 URLs - jewelry: {len(jewelry_base64)} chars, model: {len(model_base64)} chars")
                
            except Exception as e:
                print(f"[ImageProcessor] Error creating base64 data URLs: {str(e)}")
                print("[ImageProcessor] Using original image URLs as fallback")
                processed_jewelry_url = jewelry_image_url
                processed_jewelry_with_bg_url = jewelry_image_url
                model_image_url = model_image_url

            # Clean up temp files
            temp_files = []
            for temp_path in [jewelry_temp_path, model_temp_path]:
                if temp_path and os.path.exists(temp_path):
                    temp_files.append(temp_path)
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass
            
            # Add zoomed model temp file if it exists
            if 'zoomed_model_temp_path' in locals() and zoomed_model_temp_path and os.path.exists(zoomed_model_temp_path):
                temp_files.append(zoomed_model_temp_path)
                try:
                    os.unlink(zoomed_model_temp_path)
                except Exception:
                    pass
            
            # Add resized jewelry temp file if it exists
            if 'resized_jewelry_temp_path' in locals() and resized_jewelry_temp_path and os.path.exists(resized_jewelry_temp_path):
                temp_files.append(resized_jewelry_temp_path)
                try:
                    os.unlink(resized_jewelry_temp_path)
                except Exception:
                    pass

            print(f"[ImageProcessor] Step 15: Cleaned temp files - {len(temp_files)} files removed")

            result = convert_numpy_types({
                'processed_jewelry_url': processed_jewelry_url,
                'processed_jewelry_with_bg_url': processed_jewelry_with_bg_url,
                'model_image_url': model_image_url,
                'landmark_position': landmark_position,
                'jewelry_type': jewelry_type,
                'jewelry_subtype': jewelry_subtype,
                'processing_info': {
                    'original_jewelry_size': len(jewelry_data),
                    'model_size': len(model_data),
                    'model_height_cm': model_height_cm,
                    'jewelry_size_cm': jewelry_size_cm,
                    'scale_factor': scale_factor
                }
            })

            if result:
                print(f"\n✅ Success!")
                print(f"Processed jewelry URL: {result['processed_jewelry_url'][:100]}...")
                print(f"Model image URL: {result['model_image_url'][:100]}...")
                print(f"Landmark position: {result['landmark_position']}")
                print(f"Processing info: {result['processing_info']}")
            
            try:
                jewelry_base64 = result['processed_jewelry_url'].split(',')[1]
                model_base64 = result['model_image_url'].split(',')[1]
                
                jewelry_data = base64.b64decode(jewelry_base64)
                model_data = base64.b64decode(model_base64)
                
                jewelry_img = Image.open(io.BytesIO(jewelry_data))
                model_img = Image.open(io.BytesIO(model_data))
                
                position = result['landmark_position']
                x, y = position['x'], position['y']
                
                jewelry_width, jewelry_height = jewelry_img.size
                x_offset = x - jewelry_width // 2
                y_offset = y - jewelry_height // 10
                
                # x_offset = x - jewelry_width // 2
                # y_offset = y - jewelry_height // 2
                
                result_img = model_img.copy()
                if jewelry_img.mode == 'RGBA':
                    result_img.paste(jewelry_img, (x_offset, y_offset), jewelry_img)
                else:
                    result_img.paste(jewelry_img, (x_offset, y_offset))
                
                result_img.save('virtual_tryon_result.png')
                print(f"✅ Overlay image saved as 'virtual_tryon_result.png'")
            except Exception as e:
                print(f"\n❌ Failed: No result returned")
            
            print(f"[ImageProcessor] Virtual try-on processing completed successfully")
            
            if not isinstance(result['landmark_position'], dict):
                result['landmark_position'] = convert_numpy_types({
                    'x': 512,
                    'y': 256,
                    'confidence': 0.3,
                    'method': 'critical_error_fallback'
                })
            
            return result
            
        except Exception as e:
            print(f"[ImageProcessor] Virtual try-on processing failed: {str(e)}")
            return None
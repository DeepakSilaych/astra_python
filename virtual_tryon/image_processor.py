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

def get_image_height_mm(model_image_path, jewelry_type):
    """Estimate model image height using OpenAI o3 vision model"""
    with open(model_image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
    
    prompt = """Place jewelry on model. Estimate image height in mm. Return only the number."""
    
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
    
    try:
        height_str = response.choices[0].message.content.strip()
        height_mm = float(''.join(filter(str.isdigit, height_str.split('.')[0])))
        return height_mm
    except:
        fallback_heights = {
            "necklaces": 300,
            "earrings": 200,
            "bracelets": 150,
            "rings": 100
        }
        fallback_height = fallback_heights.get(jewelry_type, 200)
        return fallback_height

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
    if jewelry_type in "earrings":
        position = get_earring_position_haar(model_image_path)
    elif jewelry_type in "necklaces" or jewelry_type in "pendants":
        position, landmarks_data = detect_neck_mediapipe(model_image_path)
    elif jewelry_type in "bracelets" or jewelry_type in "bangles":
        position, landmarks_data = detect_wrist_mediapipe(model_image_path)
    elif jewelry_type in "rings":
        position, landmarks_data = detect_finger_mediapipe(model_image_path)
    if position:
        return position
    try:
        model_img = Image.open(model_image_path)
        width, height = model_img.size
        jewelry_type_lower = jewelry_type.lower()
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
        return fallback_pos
    except Exception as e:
        print(f"Geometric fallback error: {e}")
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
                ear_y = y + h
                best_ear = {'x': ear_x, 'y': ear_y, 'confidence': confidence, 'method': 'haar_cascade'}
                ear_side = "left"
        
        for (x, y, w, h) in right_ears:
            ear_area = w * h
            confidence = ear_area / (width * height)
            
            if confidence > best_confidence:
                best_confidence = confidence
                ear_x = x + int(w * 0.25)
                ear_y = y + h
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
            async with aiohttp.ClientSession() as session:
                async with session.get(jewelry_image_url) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to download jewelry image: {response.status}")
                    jewelry_data = await response.read()
                async with session.get(model_image_url) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to download model image: {response.status}")
                    model_data = await response.read()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as jewelry_temp:
                jewelry_temp.write(jewelry_data)
                jewelry_temp_path = jewelry_temp.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as model_temp:
                model_temp.write(model_data)
                model_temp_path = model_temp.name
            
            print(f"[ImageProcessor] Downloaded images: jewelry ({len(jewelry_data)} bytes), model ({len(model_data)} bytes)")
            
            cropped_jewelry = crop_jewelry_image(jewelry_temp_path, jewelry_type)
            if cropped_jewelry is None:
                cropped_jewelry = Image.open(jewelry_temp_path).convert("RGB")
            
            enhanced_jewelry = enhance_jewelry_image_from_pil(cropped_jewelry)
            if enhanced_jewelry:
                final_jewelry = enhanced_jewelry
            else:
                final_jewelry = cropped_jewelry

            model_height_mm = get_image_height_mm(model_temp_path, jewelry_type)
            
            if jewelry_size and isinstance(jewelry_size, dict):
                jewelry_size_str = jewelry_size.get('size', '20mm')
            elif jewelry_size and isinstance(jewelry_size, str):
                jewelry_size_str = jewelry_size
            else:
                jewelry_size_str = '20mm'
            
            jewelry_size_mm = calculate_jewelry_size_mm(jewelry_size_str)
            jewelry_width_px, jewelry_height_px = final_jewelry.size

            model_img = Image.open(model_temp_path)
            model_width_px, model_height_px = model_img.size
        
            target_jewelry_height_px = (jewelry_size_mm * model_height_px) / model_height_mm
            scale_factor = target_jewelry_height_px / jewelry_height_px
            
            if scale_factor < 0.1 or scale_factor > 10:
                jewelry_type_lower = jewelry_type.lower()
                if jewelry_type_lower in ['earring', 'earrings']:
                    target_percentage = 0.08
                elif jewelry_type_lower in ['necklace', 'pendant']:
                    target_percentage = 0.15
                elif jewelry_type_lower in ['bracelet', 'bangles']:
                    target_percentage = 0.12
                elif jewelry_type_lower in ['ring', 'rings']:
                    target_percentage = 0.06
                else:
                    target_percentage = 0.10
                
                target_jewelry_height_px = model_height_px * target_percentage
                scale_factor = target_jewelry_height_px / jewelry_height_px
            
            new_jewelry_width = int(jewelry_width_px * scale_factor)
            new_jewelry_height = int(jewelry_height_px * scale_factor)
            
            resized_jewelry = final_jewelry.resize((new_jewelry_width, new_jewelry_height), Image.Resampling.LANCZOS)
            
            print(f"[ImageProcessor] Jewelry resized: {jewelry_width_px}x{jewelry_height_px} -> {new_jewelry_width}x{new_jewelry_height} (scale: {scale_factor:.2f})")
            
            landmark_position = get_jewelry_placement_position(model_temp_path, jewelry_type)
            
            if not isinstance(landmark_position, dict):
                model_img = Image.open(model_temp_path)
                width, height = model_img.size
                
                jt = jewelry_type.lower()
                if jt in ['necklace', 'pendant']:
                    landmark_position = {'x': width // 2, 'y': height // 3, 'confidence': 0.5, 'method': 'fallback'}
                elif jt in ['earring', 'earrings']:
                    landmark_position = {'x': width // 4, 'y': height // 4, 'confidence': 0.5, 'method': 'fallback'}
                elif jt in ['bracelet', 'bangles']:
                    landmark_position = {'x': width // 2, 'y': int(height * 0.7), 'confidence': 0.5, 'method': 'fallback'}
                elif jt in ['ring', 'rings']:
                    landmark_position = {'x': int(width * 0.6), 'y': int(height * 0.8), 'confidence': 0.5, 'method': 'fallback'}
                else:
                    landmark_position = {'x': width // 2, 'y': height // 2, 'confidence': 0.3, 'method': 'fallback'}

            processed_jewelry_url = None
            processed_jewelry_with_bg_url = None
            
            try:
                # Convert processed images to base64 data URLs
                import base64
                
                # Convert the processed jewelry image to base64
                jewelry_bytes = io.BytesIO()
                resized_jewelry.save(jewelry_bytes, format='PNG')
                jewelry_bytes.seek(0)
                jewelry_base64 = base64.b64encode(jewelry_bytes.getvalue()).decode('utf-8')
                
                processed_jewelry_url = f"data:image/png;base64,{jewelry_base64}"
                processed_jewelry_with_bg_url = f"data:image/png;base64,{jewelry_base64}"
                
                print(f"[ImageProcessor] Created base64 data URLs for processed images")
                
            except Exception as e:
                print(f"[ImageProcessor] Error creating base64 data URLs: {str(e)}")
                print("[ImageProcessor] Using original image URLs as fallback")
                processed_jewelry_url = jewelry_image_url
                processed_jewelry_with_bg_url = jewelry_image_url

            try:
                temp_files = [jewelry_temp_path, model_temp_path]
                for temp_file in temp_files:
                    if temp_file and os.path.exists(temp_file):
                        os.unlink(temp_file)
            except Exception:
                pass

            result = convert_numpy_types({
                'processed_jewelry_url': processed_jewelry_url,
                'processed_jewelry_with_bg_url': processed_jewelry_with_bg_url,
                'landmark_position': landmark_position,
                'jewelry_type': jewelry_type,
                'jewelry_subtype': jewelry_subtype,
                'processing_info': {
                    'original_jewelry_size': len(jewelry_data),
                    'model_size': len(model_data),
                    'model_height_mm': model_height_mm,
                    'jewelry_size_mm': jewelry_size_mm,
                    'scale_factor': scale_factor
                }
            })
            
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


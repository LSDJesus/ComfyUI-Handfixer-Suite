# utils.py

import warnings
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image

warnings.filterwarnings("ignore")

# Define the landmark indices for specific body parts from MediaPipe's documentation
POSE_LANDMARKS_FEET = [27, 28, 29, 30, 31, 32]
FACE_MESH_LANDMARKS_EYES = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
]
# Landmarks for inner and outer lips
FACE_MESH_LANDMARKS_MOUTH = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308
]

class MediapipeEngine:
    """A class to handle all MediaPipe model loading and processing."""
    def __init__(self):
        self.models = {}

    def _get_model(self, model_type, confidence):
        model_key = f"{model_type}_{confidence}"
        if model_type == 'selfie':
            model_key = "selfie_segmentation"

        if model_key not in self.models:
            if model_type == 'hands':
                self.models[model_key] = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=10, min_detection_confidence=confidence)
            elif model_type == 'pose':
                self.models[model_key] = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=confidence)
            elif model_type == 'face_mesh':
                self.models[model_key] = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=10, min_detection_confidence=confidence)
            elif model_type == 'holistic':
                self.models[model_key] = mp.solutions.holistic.Holistic(static_image_mode=True, min_detection_confidence=confidence)
            elif model_type == 'selfie':
                self.models[model_key] = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
        return self.models[model_key]

    def _get_detections(self, image, model_type, confidence):
        """Runs detection and returns a list of detected instances, each with its landmarks and confidence."""
        model = self._get_model(model_type, confidence)
        results = model.process(image)
        detections = []

        if model_type == 'hands' and results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                score = results.multi_handedness[i].score
                detections.append({'landmarks': hand_landmarks.landmark, 'confidence': score})
        
        elif model_type == 'face_mesh' and results.multi_face_landmarks:
            # FaceMesh doesn't provide a direct confidence score, so we'll use a placeholder
            for face_landmarks in results.multi_face_landmarks:
                detections.append({'landmarks': face_landmarks.landmark, 'confidence': 1.0})
        
        # Add other model detections here as needed for sorting (e.g., pose has no multi-instance)
        # For simplicity, we'll focus on hands and faces for filtering, as they are most common.
        
        return detections

    def _create_mask_from_landmarks(self, image_shape, landmarks, padding, blur):
        """Creates a single dilated and blurred convex hull mask from a set of landmarks."""
        if not landmarks:
            return None

        H, W, _ = image_shape
        points = np.array([(lm.x * W, lm.y * H) for lm in landmarks if lm.x is not None], dtype=np.int32)

        if len(points) < 3:
            return None

        hull = cv2.convexHull(points)
        instance_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(instance_mask, [hull], -1, 255, -1)
        
        if padding > 0:
            kernel = np.ones((padding, padding), np.uint8)
            instance_mask = cv2.dilate(instance_mask, kernel, iterations=1)
        
        if blur > 0:
            # Blur kernel size must be odd
            blur_kernel_size = blur * 2 + 1
            instance_mask = cv2.GaussianBlur(instance_mask, (blur_kernel_size, blur_kernel_size), 0)
        
        return instance_mask

    def process_and_create_mask(self, image, options):
        """The main function to process an image and generate the final combined mask based on user options."""
        final_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        rgb_image = image
        model_type = options.get('model_type')

        # Selfie segmentation is a special case (pixel-based, no landmarks)
        if model_type == 'selfie':
            model = self._get_model('selfie', options.get('confidence'))
            results = model.process(rgb_image)
            condition = results.segmentation_mask > 0.5
            segmentation_mask = np.where(condition, 255, 0).astype(np.uint8)
            return np.maximum(final_mask, segmentation_mask)

        # --- Landmark-based processing ---
        
        # 1. Detect all instances and collect their data
        detected_instances = []
        if model_type in ['hands', 'face_mesh', 'pose', 'holistic', 'feet', 'eyes', 'mouth']:
            # This is a simplified example. A full implementation would handle each model type.
            # For now, let's focus on the 'hands' and 'face_mesh' logic which supports filtering.
            
            raw_detections = []
            if model_type == 'hands':
                raw_detections = self._get_detections(rgb_image, 'hands', options.get('confidence'))
            elif model_type in ['face_mesh', 'eyes', 'mouth']:
                raw_detections = self._get_detections(rgb_image, 'face_mesh', options.get('confidence'))
            
            # Extract specific landmarks if needed (e.g., for eyes/mouth)
            for det in raw_detections:
                landmarks = det['landmarks']
                if model_type == 'eyes':
                    landmarks = [landmarks[i] for i in FACE_MESH_LANDMARKS_EYES]
                elif model_type == 'mouth':
                    landmarks = [landmarks[i] for i in FACE_MESH_LANDMARKS_MOUTH]
                
                # Calculate properties for sorting
                H, W, _ = image.shape
                points = np.array([(lm.x * W, lm.y * H) for lm in landmarks], dtype=np.int32)
                if len(points) < 3: continue
                
                area = cv2.contourArea(cv2.convexHull(points))
                M = cv2.moments(points)
                center_x = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
                
                detected_instances.append({
                    'landmarks': landmarks,
                    'confidence': det['confidence'],
                    'area': area,
                    'center_x': center_x
                })
        
        # 2. Sort the detected instances
        sort_by = options.get('sort_by')
        if sort_by == 'Confidence':
            detected_instances.sort(key=lambda x: x['confidence'], reverse=True)
        elif sort_by == 'Area: Largest to Smallest':
            detected_instances.sort(key=lambda x: x['area'], reverse=True)
        elif sort_by == 'Position: Left to Right':
            detected_instances.sort(key=lambda x: x['center_x'])
        # Add other sorting methods here...

        # 3. Filter the sorted list
        max_objects = options.get('max_objects')
        filtered_instances = detected_instances[:max_objects]

        # 4. Create and combine masks for the filtered instances
        for instance in filtered_instances:
            instance_mask = self._create_mask_from_landmarks(
                image.shape, 
                instance['landmarks'],
                options.get('mask_padding'),
                options.get('mask_blur')
            )
            if instance_mask is not None:
                final_mask = np.maximum(final_mask, instance_mask)
        
        return final_mask
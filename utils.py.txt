import numpy as np
import cv2
import mediapipe as mp
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# This import is now needed for the type check
from mediapipe.framework.formats import landmark_pb2

POSE_LANDMARKS_FEET = [27, 28, 29, 30, 31, 32]
POSE_LANDMARKS_TORSO = [11, 12, 23, 24]
FACE_MESH_LANDMARKS_EYES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
FACE_MESH_LANDMARKS_MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

class MediapipeEngine:
    def __init__(self):
        self.models = {}

    def _get_model(self, model_type, confidence):
        model_name_map = {
            'face': 'face_mesh', 'eyes': 'face_mesh', 'mouth': 'face_mesh',
            'full_body (bbox)': 'pose', 'feet': 'pose', 'torso': 'pose',
            'person (segmentation)': 'selfie', 'hands': 'hands'
        }
        model_name = model_name_map.get(model_type, model_type)
        
        model_key = f"{model_name}_{confidence}"
        if model_name == 'selfie': model_key = "selfie_segmentation"

        if model_key not in self.models:
            if model_name == 'hands': self.models[model_key] = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=10, min_detection_confidence=confidence)
            elif model_name == 'pose': self.models[model_key] = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=confidence)
            elif model_name == 'face_mesh': self.models[model_key] = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=10, min_detection_confidence=confidence)
            elif model_name == 'selfie': self.models[model_key] = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
        return self.models.get(model_key)

    def _create_mask_from_landmarks(self, image_shape, landmarks, padding, blur):
        if not landmarks: return None
        H, W, _ = image_shape

        # --- THE FIX IS HERE ---
        # Check if we received a special MediaPipe LandmarkList object or a plain Python list.
        list_of_points = []
        if isinstance(landmarks, landmark_pb2.NormalizedLandmarkList):
            # If it's the special object, get the list from its '.landmark' attribute.
            list_of_points = landmarks.landmark
        else:
            # Otherwise, assume we already have a plain list of points.
            list_of_points = landmarks
        
        points = np.array([(lm.x * W, lm.y * H) for lm in list_of_points if hasattr(lm, 'x')], dtype=np.int32)
        if len(points) < 3: return None
        
        hull = cv2.convexHull(points)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(mask, [hull], -1, 255, -1)
        
        if padding > 0:
            kernel = np.ones((padding, padding), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        
        if blur > 0:
            blur_kernel_size = blur * 2 + 1
            mask = cv2.GaussianBlur(mask, (blur_kernel_size, blur_kernel_size), 0)
        return mask

    def process_and_create_mask(self, image, options):
        final_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        model_type = options['model_type']
        confidence = options['confidence']
        
        model = self._get_model(model_type, confidence)
        if not model: return final_mask
        
        results = model.process(image)

        if model_type == 'person (segmentation)':
            if hasattr(results, 'segmentation_mask'):
                return np.where(results.segmentation_mask > 0.5, 255, 0).astype(np.uint8)
            return final_mask
        
        landmark_sets = []
        if model_type == 'hands' and results.multi_hand_landmarks: landmark_sets.extend(results.multi_hand_landmarks)
        elif model_type in ['face', 'eyes', 'mouth'] and results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if model_type == 'eyes': landmark_sets.append([face_landmarks.landmark[i] for i in FACE_MESH_LANDMARKS_EYES])
                elif model_type == 'mouth': landmark_sets.append([face_landmarks.landmark[i] for i in FACE_MESH_LANDMARKS_MOUTH])
                else: landmark_sets.append(face_landmarks)
        elif model_type in ['full_body (bbox)', 'feet', 'torso'] and results.pose_landmarks:
            if model_type == 'feet': landmark_sets.append([results.pose_landmarks.landmark[i] for i in POSE_LANDMARKS_FEET])
            elif model_type == 'torso': landmark_sets.append([results.pose_landmarks.landmark[i] for i in POSE_LANDMARKS_TORSO])
            else: landmark_sets.append(results.pose_landmarks)

        for landmarks in landmark_sets:
            instance_mask = self._create_mask_from_landmarks(image.shape, landmarks, options['mask_padding'], options['mask_blur'])
            if instance_mask is not None: final_mask = np.maximum(final_mask, instance_mask)
            
        return final_mask
"""
Logo detectors
Contains SIFT, ORB, BRISK, SURF and other feature matching algorithms
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Callable
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class BaseDetector:
    """Base class for all detectors"""
    
    def __init__(self, logo_path: str, **kwargs):
        self.logo_path = logo_path
        self.template = None
        self.template_kp = None
        self.template_des = None
        self.template_height = 0
        self.template_width = 0
        self._load_template()
    
    def _load_template(self):
        """To be implemented by each detector"""
        raise NotImplementedError
    
    def detect_in_frame(self, frame: np.ndarray, frame_number: int = 0, total_frames: int = 0) -> List[Tuple[int, int, int, int, float]]:
        """To be implemented by each detector"""
        raise NotImplementedError
    
    def process_video(self, video_path: str, output_path: str, progress_callback: Callable = None) -> Dict[str, Any]:
        """Standard video processing"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer with mp4v (will re-encode to H.264 later)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = Path(output_path)  # Convert to Path object
        temp_output = output_path.parent / f"temp_{output_path.name}"
        
        out = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise ValueError(f"Cannot create output video: {temp_output}")
        
        logger.info(f"Using temporary codec: mp4v (will convert to H.264)")
        
        logger.info(f"Processing video with {self.__class__.__name__}: {total_frames} frames")
        
        all_detections = []
        frame_number = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            
            # Detect logo in frame
            detections = self.detect_in_frame(frame, frame_number, total_frames)
            
            if detections:
                all_detections.append({
                    "frame_number": frame_number,
                    "timestamp": frame_number / fps,  # Calculate timestamp
                    "bounding_boxes": [
                        {"x": x, "y": y, "width": w, "height": h, "confidence": conf}
                        for x, y, w, h, conf in detections
                    ]
                })
                
                # Draw bounding boxes
                for x, y, w, h, conf in detections:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{self.__class__.__name__}: {conf:.2f}", (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            out.write(frame)
            
            # Progress callback
            if progress_callback:
                progress = int((frame_number / total_frames) * 100)
                progress_callback(progress, frame_number, total_frames)
        
        cap.release()
        out.release()
        
        # Re-encode to H.264 for browser compatibility
        logger.info("Re-encoding video to H.264 for browser compatibility...")
        import subprocess
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', str(temp_output),
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                '-pix_fmt', 'yuv420p',  # Required for browser compatibility
                str(output_path)
            ], check=True, capture_output=True, text=True)
            
            # Remove temporary file
            temp_output.unlink()
            logger.info("Video successfully re-encoded to H.264")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to re-encode video: {e.stderr}")
            # Fallback: use the temp file as final output
            temp_output.rename(output_path)
            logger.warning("Using original mp4v codec (may not work in all browsers)")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info(f"{self.__class__.__name__} processing completed: {len(all_detections)} frames with detections in {processing_time:.2f}s")
        
        return {
            "total_frames": total_frames,
            "frames_with_logo": len(all_detections),
            "processing_time": processing_time,
            "detections": all_detections
        }


class SIFTDetector(BaseDetector):
    """SIFT detector (Scale-Invariant Feature Transform) with structural validation"""
    
    def __init__(self, logo_path: str, min_matches: int = 15, ratio_threshold: float = 0.75):
        self.min_matches = min_matches
        self.ratio_threshold = ratio_threshold
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        
        super().__init__(logo_path)
    
    def _load_template(self):
        """Load SIFT template"""
        try:
            logo_img = cv2.imread(str(self.logo_path), cv2.IMREAD_GRAYSCALE)
            if logo_img is None:
                raise ValueError(f"Cannot load logo from {self.logo_path}")
            
            self.template = logo_img
            self.template_height, self.template_width = logo_img.shape
            self.template_kp, self.template_des = self.sift.detectAndCompute(logo_img, None)
            
            if self.template_des is None:
                logger.warning("No SIFT features found in logo template")
                return
            
            logger.info(f"Loaded SIFT template: {len(self.template_kp)} features")
            
        except Exception as e:
            logger.error(f"Failed to load SIFT template: {e}")
            raise
    
    def _validate_neurons_structure(self, width, height, frame_shape):
        """Validate structure according to brand guidelines"""
        frame_height, frame_width = frame_shape[:2]
        frame_area = frame_width * frame_height
        detection_area = width * height
        
        # Safe Zone minimum 30px according to guidelines
        if width < 30 or height < 30:
            return False
        
        # Maximum size 25% of the image
        if detection_area > frame_area * 0.25:
            return False
        
        # Aspect ratio for neurons logo (wider than tall)
        aspect_ratio = width / height
        if aspect_ratio < 1.0 or aspect_ratio > 8:
            return False
        
        return True
    
    def detect_in_frame(self, frame: np.ndarray, frame_number: int = 0, total_frames: int = 0) -> List[Tuple[int, int, int, int, float]]:
        """SIFT detection"""
        if self.template_des is None:
            return []
        
        try:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_kp, frame_des = self.sift.detectAndCompute(frame_gray, None)
            
            if frame_des is None or len(frame_kp) == 0:
                return []
            
            matches = self.matcher.knnMatch(self.template_des, frame_des, k=2)
            
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.ratio_threshold * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < self.min_matches:
                return []
            
            src_pts = np.float32([self.template_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if M is None:
                return []
            
            h, w = self.template.shape
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(corners, M)
            
            x_coords = transformed_corners[:, 0, 0]
            y_coords = transformed_corners[:, 0, 1]
            
            x_min, y_min = int(np.min(x_coords)), int(np.min(y_coords))
            x_max, y_max = int(np.max(x_coords)), int(np.max(y_coords))
            
            if x_max <= x_min or y_max <= y_min:
                return []
            
            width = x_max - x_min
            height = y_max - y_min
            
            # Validation according to brand guidelines (structure only)
            if not self._validate_neurons_structure(width, height, frame.shape):
                logger.debug("SIFT: Detection rejected (invalid structure)")
            return []

            confidence = len(good_matches) / len(self.template_kp)
            
            logger.info(f"SIFT found valid neurons logo with {len(good_matches)} matches, confidence {confidence:.3f}")
                return [(x_min, y_min, width, height, confidence)]
            
        except Exception as e:
            logger.warning(f"SIFT detection error: {e}")
            return []


class ORBDetector(BaseDetector):
    """ORB detector (Oriented FAST and Rotated BRIEF)"""
    
    def __init__(self, logo_path: str, min_matches: int = 25, ratio_threshold: float = 0.78):
        self.min_matches = min_matches
        self.ratio_threshold = ratio_threshold
        self.orb = cv2.ORB_create(nfeatures=2000)
        # Use knnMatch instead of crossCheck to apply Lowe's ratio test
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        super().__init__(logo_path)
    
    def _load_template(self):
        """Load ORB template"""
        try:
            logo_img = cv2.imread(str(self.logo_path), cv2.IMREAD_GRAYSCALE)
            if logo_img is None:
                raise ValueError(f"Cannot load logo from {self.logo_path}")
            
            self.template = logo_img
            self.template_height, self.template_width = logo_img.shape
            self.template_kp, self.template_des = self.orb.detectAndCompute(logo_img, None)
            
            if self.template_des is None:
                logger.warning("No ORB features found in logo template")
                return
            
            logger.info(f"Loaded ORB template: {len(self.template_kp)} features")
            
        except Exception as e:
            logger.error(f"Failed to load ORB template: {e}")
            raise
    
    def detect_in_frame(self, frame: np.ndarray, frame_number: int = 0, total_frames: int = 0) -> List[Tuple[int, int, int, int, float]]:
        """ORB detection with Lowe's ratio test and homography validation"""
        if self.template_des is None:
            return []
        
        try:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_kp, frame_des = self.orb.detectAndCompute(frame_gray, None)
            
            if frame_des is None or len(frame_kp) == 0:
                return []
            
            # Use knnMatch to apply Lowe's ratio test
            knn_matches = self.matcher.knnMatch(self.template_des, frame_des, k=2)
            
            # Apply Lowe's ratio test to filter false positives
            good_matches = []
            for match_pair in knn_matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    # Lowe's ratio test: best match must be significantly better than second
                    if m.distance < self.ratio_threshold * n.distance:
                        # Additional filter on absolute distance
                        if m.distance < 45:  # Trade-off between quality and sensitivity
                            good_matches.append(m)
            
            if len(good_matches) < self.min_matches:
                return []
            
            src_pts = np.float32([self.template_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # RANSAC with optimized threshold
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
            
            if M is None:
                return []
            
            # Homography validation: check that it's not too distorted
            try:
                det = np.linalg.det(M[:2, :2])
                # Determinant must be reasonable (no excessive distortion)
                if det < 0.05 or det > 20.0:
                    logger.debug(f"ORB: Homography rejected (determinant={det:.2f})")
                    return []
            except:
                return []
            
            # Count inliers after RANSAC
            inliers = np.sum(mask)
            # Accept if at least 70% of min_matches is met
            if inliers < int(self.min_matches * 0.7):
                return []
            
            h, w = self.template.shape
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(corners, M)
            
            x_coords = transformed_corners[:, 0, 0]
            y_coords = transformed_corners[:, 0, 1]
            
            x_min, y_min = int(np.min(x_coords)), int(np.min(y_coords))
            x_max, y_max = int(np.max(x_coords)), int(np.max(y_coords))
            
            if x_max <= x_min or y_max <= y_min:
                return []
            
            width = x_max - x_min
            height = y_max - y_min
            
            # Check that detected size is reasonable (not too small or too large)
            template_area = w * h
            detected_area = width * height
            area_ratio = detected_area / template_area
            
            # Reject if size is too different from template (wider range)
            if area_ratio < 0.1 or area_ratio > 10.0:
                logger.debug(f"ORB: Detection rejected (size ratio={area_ratio:.2f})")
                return []
            
            # Confidence based on inliers rather than all matches
            confidence = inliers / len(self.template_kp)
            
            return [(x_min, y_min, width, height, confidence)]
            
        except Exception as e:
            logger.warning(f"ORB detection error: {e}")
            return []


class ColorBasedDetector(BaseDetector):
    """Detector based on segmentation of distinctive logo colors"""
    
    def __init__(self, logo_path: str, color_threshold: float = 0.2, min_area: int = 50):
        self.color_threshold = color_threshold
        self.min_area = min_area
        
        # Define color ranges for neurons logo based on brand guidelines
        # Official colors: Tulip #FE839C, Bright Sun #FFD14C, French Sky Blue #85A0FE, Lavender #AA82FF, Dark Indigo #380F57
        self.color_ranges = {
            'tulip': ([340, 100, 80], [360, 255, 255]),      # Tulip #FE839C (pink/red)
            'bright_sun': ([40, 100, 80], [60, 255, 255]),   # Bright Sun #FFD14C (yellow)
            'french_sky_blue': ([220, 100, 80], [240, 255, 255]), # French Sky Blue #85A0FE (blue)
            'lavender': ([260, 100, 80], [280, 255, 255]),   # Lavender #AA82FF (light purple)
            'dark_indigo': ([270, 200, 20], [290, 255, 100]) # Dark Indigo #380F57 (dark purple - text)
        }
        
        super().__init__(logo_path)
    
    def _load_template(self):
        """Load template to analyze colors"""
        try:
            logo_img = cv2.imread(str(self.logo_path))
            if logo_img is None:
                raise ValueError(f"Cannot load logo from {self.logo_path}")
            
            self.template = logo_img
            self.template_height, self.template_width = logo_img.shape[:2]
            
            # Analyze dominant colors in template
            self._analyze_template_colors(logo_img)
            
            logger.info(f"Loaded ColorBased template: {self.template_width}x{self.template_height}")
            
        except Exception as e:
            logger.error(f"Failed to load ColorBased template: {e}")
            raise
    
    def _analyze_template_colors(self, template):
        """Analyze template colors to adjust ranges"""
        hsv_template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
        
        # Create masks for each color and adjust ranges if necessary
        for color_name, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv_template, np.array(lower), np.array(upper))
            pixel_count = np.sum(mask > 0)
            
            if pixel_count > 0:
                logger.info(f"Color {color_name}: {pixel_count} pixels found")
    
    def _segment_colors(self, frame):
        """Segment logo colors in the image"""
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create combined mask for all logo colors
        combined_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
        
        for color_name, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Nettoyer le masque avec des opérations morphologiques plus agressives
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Dilatation pour connecter les parties du logo
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined_mask = cv2.dilate(combined_mask, kernel_dilate, iterations=1)
        
        return combined_mask
    
    def _find_logo_regions(self, mask):
        """Find candidate regions for the logo"""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            
            # Calculate bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (logo approximately rectangular)
            aspect_ratio = w / h
            if 0.5 < aspect_ratio < 3.0:  # Logo can be wider than tall
                # Calculate logo pixel density in this region
                roi_mask = mask[y:y+h, x:x+w]
                density = np.sum(roi_mask > 0) / (w * h)
                
                if density > self.color_threshold:
                    confidence = min(density * 2, 1.0)  # Normalize confidence
                    candidates.append((x, y, w, h, confidence))
        
        return candidates
    
    def detect_in_frame(self, frame: np.ndarray, frame_number: int = 0, total_frames: int = 0) -> List[Tuple[int, int, int, int, float]]:
        """Color-based detection"""
        try:
            # Segment logo colors
            color_mask = self._segment_colors(frame)
            
            # Find candidate regions
            candidates = self._find_logo_regions(color_mask)
            
            if not candidates:
                return []
            
            # Apply non-maximum suppression to avoid duplicates
            final_detections = self._non_maximum_suppression(candidates)
            
            if final_detections:
                logger.info(f"ColorBased found {len(final_detections)} detections in frame {frame_number}")
            
            return final_detections
            
        except Exception as e:
            logger.warning(f"ColorBased detection error: {e}")
            return []
    
    def _non_maximum_suppression(self, detections, overlap_threshold=0.2):
        """Non-maximum suppression to eliminate duplicates"""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x[4], reverse=True)
        
        # Group nearby detections
        groups = []
        used = set()
        
        for i, det in enumerate(detections):
            if i in used:
                continue
            
            group = [det]
            used.add(i)
            
            # Find all nearby detections
            for j, other_det in enumerate(detections):
                if j in used:
                    continue
                
                # Calculate distance between centers
                x1, y1, w1, h1 = det[:4]
                x2, y2, w2, h2 = other_det[:4]
                
                center1 = (x1 + w1/2, y1 + h1/2)
                center2 = (x2 + w2/2, y2 + h2/2)
                
                distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
                
                # If detections are close (within 80 pixels radius)
                if distance < 80:
                    group.append(other_det)
                    used.add(j)
            
            groups.append(group)
        
        # Merge each group into a single detection
        final_detections = []
        for group in groups:
            if len(group) == 1:
                final_detections.append(group[0])
            else:
                # Merge group into a single bounding box
                merged = self._merge_detections(group)
                final_detections.append(merged)
        
        return final_detections
    
    def _merge_detections(self, detections):
        """Merge multiple detections into a single bounding box"""
        if not detections:
            return None
        
        # Calculate bounding box
        x_min = min(det[0] for det in detections)
        y_min = min(det[1] for det in detections)
        x_max = max(det[0] + det[2] for det in detections)
        y_max = max(det[1] + det[3] for det in detections)
        
        width = x_max - x_min
        height = y_max - y_min
        
        # Weighted average confidence by size
        total_area = sum(det[2] * det[3] for det in detections)
        weighted_confidence = sum(det[4] * (det[2] * det[3]) for det in detections) / total_area
        
        return (x_min, y_min, width, height, weighted_confidence)
    
    def _compute_overlap(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = box2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0


class EdgeBasedDetector(BaseDetector):
    """Detector based on contour/edge detection"""
    
    def __init__(self, logo_path: str, edge_threshold1: int = 50, edge_threshold2: int = 150, 
                 min_contour_area: int = 200):
        self.edge_threshold1 = edge_threshold1
        self.edge_threshold2 = edge_threshold2
        self.min_contour_area = min_contour_area
        
        super().__init__(logo_path)
    
    def _load_template(self):
        """Load template and extract its contours"""
        try:
            logo_img = cv2.imread(str(self.logo_path), cv2.IMREAD_GRAYSCALE)
            if logo_img is None:
                raise ValueError(f"Cannot load logo from {self.logo_path}")
            
            self.template = logo_img
            self.template_height, self.template_width = logo_img.shape
            
            # Extract template contours
            self.template_edges = cv2.Canny(logo_img, self.edge_threshold1, self.edge_threshold2)
            
            logger.info(f"Loaded EdgeBased template: {self.template_width}x{self.template_height}")
            
        except Exception as e:
            logger.error(f"Failed to load EdgeBased template: {e}")
            raise
    
    def detect_in_frame(self, frame: np.ndarray, frame_number: int = 0, total_frames: int = 0) -> List[Tuple[int, int, int, int, float]]:
        """Contour-based detection"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce textured background noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Detect contours
            edges = cv2.Canny(blurred, self.edge_threshold1, self.edge_threshold2)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            candidates = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_contour_area:
                    continue
                
                # Calculate bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio
                aspect_ratio = w / h
                if 0.3 < aspect_ratio < 4.0:
                    # Calculate similarity with template (contours)
                    roi_edges = edges[y:y+h, x:x+w]
                    similarity = self._compute_edge_similarity(roi_edges)
                    
                    if similarity > 0.3:  # Similarity threshold
                        candidates.append((x, y, w, h, similarity))
            
            return candidates
        
        except Exception as e:
            logger.warning(f"EdgeBased detection error: {e}")
            return []
        
    def _compute_edge_similarity(self, roi_edges):
        """Calculate similarity between ROI contours and template"""
        try:
            # Resize ROI to match template
            resized_roi = cv2.resize(roi_edges, (self.template_width, self.template_height))
            
            # Calculate correlation between contours
            correlation = cv2.matchTemplate(resized_roi, self.template_edges, cv2.TM_CCOEFF_NORMED)
            
            return float(correlation[0][0])
            
        except:
            return 0.0




class HybridDetector(BaseDetector):
    """Hybrid detector combining multiple approaches for textured backgrounds"""
    
    def __init__(self, logo_path: str, color_weight: float = 0.6, edge_weight: float = 0.4):
        self.color_weight = color_weight
        self.edge_weight = edge_weight
        
        # Initialiser les sous-détecteurs
        self.color_detector = ColorBasedDetector(logo_path)
        self.edge_detector = EdgeBasedDetector(logo_path)
        
        super().__init__(logo_path)
    
    def _load_template(self):
        """Sub-detectors handle their own loading"""
        self.template_height = self.color_detector.template_height
        self.template_width = self.color_detector.template_width
        logger.info(f"Loaded Hybrid detector: {self.template_width}x{self.template_height}")
    
    def detect_in_frame(self, frame: np.ndarray, frame_number: int = 0, total_frames: int = 0) -> List[Tuple[int, int, int, int, float]]:
        """Hybrid detection combining color and contours"""
        try:
            # Color detections
            color_detections = self.color_detector.detect_in_frame(frame, frame_number, total_frames)
            
            # Contour detections
            edge_detections = self.edge_detector.detect_in_frame(frame, frame_number, total_frames)
            
            # Combine results
            all_detections = []
            
            # Add color detections with weight
            for x, y, w, h, conf in color_detections:
                weighted_conf = conf * self.color_weight
                all_detections.append((x, y, w, h, weighted_conf))
            
            # Add contour detections with weight
            for x, y, w, h, conf in edge_detections:
                weighted_conf = conf * self.edge_weight
                all_detections.append((x, y, w, h, weighted_conf))
            
            if not all_detections:
                return []

            # If we have multiple detections, try to merge them intelligently
            if len(all_detections) > 1:
                # Filter detections by size and confidence before fusion
                filtered_detections = self._filter_detections_for_fusion(all_detections, frame.shape)
                
                if len(filtered_detections) > 1:
                    # Calculate bounding box of filtered detections
                    x_min = min(det[0] for det in filtered_detections)
                    y_min = min(det[1] for det in filtered_detections)
                    x_max = max(det[0] + det[2] for det in filtered_detections)
                    y_max = max(det[1] + det[3] for det in filtered_detections)
                    
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    # Check that merged box is reasonable
                    if self._is_valid_merged_detection(width, height, frame.shape, filtered_detections):
                        # Weighted average confidence
                        total_area = sum(det[2] * det[3] for det in filtered_detections)
                        weighted_confidence = sum(det[4] * (det[2] * det[3]) for det in filtered_detections) / total_area
                        
                        logger.info(f"Hybrid merged {len(filtered_detections)} detections into one in frame {frame_number}")
                        return [(x_min, y_min, width, height, weighted_confidence)]
            
            # If fusion is impossible, use normal NMS
            final_detections = self._non_maximum_suppression(all_detections)
            
            if final_detections:
                logger.info(f"Hybrid found {len(final_detections)} detections in frame {frame_number}")
            
            return final_detections
            
        except Exception as e:
            logger.warning(f"Hybrid detection error: {e}")
            return []
    
    def _non_maximum_suppression(self, detections, overlap_threshold=0.5):
        """Non-maximum suppression to merge detections"""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x[4], reverse=True)
        
        # Group nearby detections with wider distance
        groups = []
        used = set()
        
        for i, det in enumerate(detections):
            if i in used:
                continue
            
            group = [det]
            used.add(i)
            
            # Find all nearby detections
            for j, other_det in enumerate(detections):
                if j in used:
                    continue
                
                # Calculate distance between centers
                x1, y1, w1, h1 = det[:4]
                x2, y2, w2, h2 = other_det[:4]
                
                center1 = (x1 + w1/2, y1 + h1/2)
                center2 = (x2 + w2/2, y2 + h2/2)
                
                distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
                
                # Wider distance to capture entire logo (200 pixels)
                if distance < 200:
                    group.append(other_det)
                    used.add(j)
            
            groups.append(group)
        
        # Merge each group into a single detection
        final_detections = []
        for group in groups:
            if len(group) == 1:
                final_detections.append(group[0])
            else:
                # Merge group into a single bounding box
                merged = self._merge_detections(group)
                if merged:  # Check that fusion succeeded
                    final_detections.append(merged)
        
        return final_detections
    
    def _merge_detections(self, detections):
        """Merge multiple detections into a single bounding box"""
        if not detections:
            return None
        
        # Calculate bounding box
        x_min = min(det[0] for det in detections)
        y_min = min(det[1] for det in detections)
        x_max = max(det[0] + det[2] for det in detections)
        y_max = max(det[1] + det[3] for det in detections)
        
        width = x_max - x_min
        height = y_max - y_min
        
        # Weighted average confidence by size
        total_area = sum(det[2] * det[3] for det in detections)
        weighted_confidence = sum(det[4] * (det[2] * det[3]) for det in detections) / total_area
        
        return (x_min, y_min, width, height, weighted_confidence)
    
    def _compute_overlap(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = box2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _filter_detections_for_fusion(self, detections, frame_shape):
        """Filter detections to avoid merging irrelevant elements"""
        if not detections:
            return []
        
        filtered = []
        frame_height, frame_width = frame_shape[:2]
        
        for det in detections:
            x, y, w, h, conf = det
            
            # Filter by size - avoid detections too small or too large
            area = w * h
            frame_area = frame_width * frame_height
            
            # Minimum size: at least 0.1% of image
            # Maximum size: no more than 30% of image
            if area < frame_area * 0.001 or area > frame_area * 0.3:
                continue
            
            # Filter by aspect ratio - avoid shapes too elongated
            aspect_ratio = w / h
            if aspect_ratio < 0.1 or aspect_ratio > 10:
                continue
            
            # Filter by minimum confidence
            if conf < 0.3:
                continue
            
            # Filter detections too close to edges (often artifacts)
            margin = min(frame_width, frame_height) * 0.05  # 5% margin
            if x < margin or y < margin or x + w > frame_width - margin or y + h > frame_height - margin:
                continue
            
            filtered.append(det)
        
        return filtered
    
    def _is_valid_merged_detection(self, width, height, frame_shape, detections):
        """Check if merged detection is valid according to brand guidelines"""
        frame_height, frame_width = frame_shape[:2]
        frame_area = frame_width * frame_height
        merged_area = width * height
        
        # Merged box should not be too large (max 25% of image)
        if merged_area > frame_area * 0.25:
            return False
        
        # Aspect ratio specific to neurons logo (icon + text)
        # Logo is generally wider than tall (triangular icon + "neurons" text)
        aspect_ratio = width / height
        if aspect_ratio < 1.5 or aspect_ratio > 6:  # Neurons logo typically 2-4x wider
            return False
        
        # Check that original detections are consistent
        if len(detections) < 2:
            return False
        
        # Calculate detection dispersion
        centers = [(det[0] + det[2]/2, det[1] + det[3]/2) for det in detections]
        center_x = sum(c[0] for c in centers) / len(centers)
        center_y = sum(c[1] for c in centers) / len(centers)
        
        # Calculate standard deviation of positions
        std_x = (sum((c[0] - center_x)**2 for c in centers) / len(centers))**0.5
        std_y = (sum((c[1] - center_y)**2 for c in centers) / len(centers))**0.5
        
        # Detections must be relatively grouped (Safe Zone concept)
        max_std = min(frame_width, frame_height) * 0.08  # 8% of image (stricter)
        if std_x > max_std or std_y > max_std:
            return False
        
        # Check minimum size (Safe Zone minimum 30px according to guidelines)
        min_size = 30
        if width < min_size or height < min_size:
            return False
        
        return True


# Factory to create detectors
def create_detector(detector_type: str, logo_path: str, **kwargs):
    """Factory to create detectors"""
    detectors = {
        "sift": SIFTDetector,
        "orb": ORBDetector,
        "color_based": ColorBasedDetector,
        "edge_based": EdgeBasedDetector,
        "hybrid": HybridDetector
    }
    
    if detector_type not in detectors:
        raise ValueError(f"Unknown detector type: {detector_type}. Available: {list(detectors.keys())}")
    
    try:
        return detectors[detector_type](logo_path, **kwargs)
    except Exception as e:
        logger.error(f"Failed to create {detector_type} detector: {e}")
        raise


def get_available_detectors():
    """Return list of available detectors"""
    available = []
    
    # Test each detector
    detectors_to_test = {
        "sift": SIFTDetector,
        "orb": ORBDetector,
        "color_based": ColorBasedDetector,
        "edge_based": EdgeBasedDetector,
        "hybrid": HybridDetector
    }
    
    for name, detector_class in detectors_to_test.items():
        try:
            available.append(name)
        except:
            logger.warning(f"Detector {name} not available")
    
    return available


def benchmark_detectors(video_path: str, logo_path: str, sample_frames: int = 50):
    """Benchmark all available detectors"""
    available_detectors = get_available_detectors()
    results = {}
    
    print(f"🔍 Benchmarking {len(available_detectors)} detectors on {sample_frames} frames")
    print("=" * 60)
    
    # Extract sample frames
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    sample_indices = np.linspace(0, total_frames-1, sample_frames, dtype=int)
    test_frames = []
    
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            test_frames.append((idx, frame))
    
    cap.release()
    
    # Test each detector
    for detector_name in available_detectors:
        print(f"\n🧪 Testing {detector_name.upper()}...")
        
        try:
            detector = create_detector(detector_name, logo_path)
            
            detections_count = 0
            start_time = time.time()
            
            for frame_idx, frame in test_frames:
                detections = detector.detect_in_frame(frame)
                if detections:
                    detections_count += 1
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            detection_rate = detections_count / len(test_frames) * 100
            fps = len(test_frames) / processing_time
            
            results[detector_name] = {
                "detection_rate": detection_rate,
                "processing_time": processing_time,
                "fps": fps,
                "detections": detections_count,
                "frames_tested": len(test_frames)
            }
            
            print(f"  ✅ {detection_rate:.1f}% detection rate, {fps:.1f} fps")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[detector_name] = {"error": str(e)}
    
    # Final results
    print(f"\n🏆 BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Detector':<10} {'Detection%':<12} {'FPS':<8} {'Time(s)':<10} {'Detections':<12}")
    print("-" * 80)
    
    for name, result in results.items():
        if "error" in result:
            print(f"{name.upper():<10} {'ERROR':<12} {'-':<8} {'-':<10} {'-':<12}")
        else:
            print(f"{name.upper():<10} {result['detection_rate']:.1f}%{'':<8} {result['fps']:.1f}{'':<5} {result['processing_time']:.2f}{'':<7} {result['detections']:<12}")
    
    return results

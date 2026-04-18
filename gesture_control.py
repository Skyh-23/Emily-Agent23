"""
Gesture Control Module — Emily Voice Assistant
================================================
MediaPipe 21-point hand landmark detection with:
- Hand calibration phase
- Volume control (thumb-index pinch)
- Brightness control (thumb-index pinch)
- General hand skeleton visualization

Usage:
    from gesture_control import start_gesture, stop_gesture
    start_gesture("volume")    # or "brightness" or "general"
    stop_gesture()
"""

import cv2
import threading
import math
import time
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ===== MEDIAPIPE IMPORT =====
MEDIAPIPE_AVAILABLE = False
mp_hands = None
mp_drawing = None

try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    MEDIAPIPE_AVAILABLE = True
    logger.info("✅ MediaPipe loaded successfully (21-point hand detection)")
except Exception as e:
    logger.error(f"❌ MediaPipe not available: {e}")

# ===== PYCAW (Volume Control) =====
PYCAW_AVAILABLE = False
volume_interface = None

try:
    from ctypes import POINTER, cast
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    
    devices = AudioUtilities.GetSpeakers()
    if hasattr(devices, '_dev'):
        interface = devices._dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    else:
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    
    volume_interface = cast(interface, POINTER(IAudioEndpointVolume))
    current_volume = volume_interface.GetMasterVolumeLevelScalar()
    PYCAW_AVAILABLE = True
    logger.info(f"✅ Pycaw loaded (Current volume: {int(current_volume * 100)}%)")
except Exception as e:
    logger.error(f"❌ Pycaw not available: {e}")

# ===== SCREEN BRIGHTNESS =====
SBC_AVAILABLE = False
sbc = None

try:
    import screen_brightness_control as sbc
    SBC_AVAILABLE = True
    logger.info("✅ Screen Brightness Control loaded")
except Exception as e:
    logger.error(f"❌ Screen Brightness Control not available: {e}")



# ===== GLOBAL VARIABLES =====
gesture_active = False
gesture_mode = "none"
gesture_thread = None

# ===== HAND LANDMARK INDICES =====
# Reference: https://google.github.io/mediapipe/solutions/hands.html
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# Finger connections for drawing skeleton
FINGER_CONNECTIONS = [
    # Thumb
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    # Index
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    # Middle
    (WRIST, MIDDLE_MCP), (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    # Ring
    (WRIST, RING_MCP), (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    # Pinky
    (WRIST, PINKY_MCP), (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
    # Palm connections
    (INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, RING_MCP), (RING_MCP, PINKY_MCP)
]

def _map(value, in_min, in_max, out_min, out_max):
    """Map value from one range to another"""
    if in_max - in_min == 0:
        return out_min
    ratio = (value - in_min) / (in_max - in_min)
    ratio = max(0.0, min(1.0, ratio))
    return out_min + (out_max - out_min) * ratio

def draw_hand_landmarks(frame, landmarks, w, h):
    """Draw 21 hand landmarks with connections like MediaPipe style"""
    points = []
    
    # Convert normalized landmarks to pixel coordinates
    for lm in landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        points.append((x, y))
    
    # Draw connections (skeleton lines)
    for start_idx, end_idx in FINGER_CONNECTIONS:
        start = points[start_idx]
        end = points[end_idx]
        cv2.line(frame, start, end, (0, 255, 0), 2)
    
    # Draw landmark points
    for i, point in enumerate(points):
        # Different colors for different parts
        if i == WRIST:
            color = (255, 0, 0)  # Blue for wrist
            size = 10
        elif i in [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]:
            color = (0, 255, 255)  # Yellow for fingertips
            size = 12
        elif i in [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]:
            color = (0, 255, 0)  # Green for MCP joints
            size = 8
        else:
            color = (255, 0, 255)  # Magenta for other joints
            size = 6
        
        cv2.circle(frame, point, size, color, cv2.FILLED)
        cv2.circle(frame, point, size + 2, (255, 255, 255), 1)
    
    return points

def _gesture_loop_mediapipe(mode="general"):
    """Gesture control using MediaPipe 21-point hand detection"""
    global gesture_active, gesture_mode, volume_interface

    if not MEDIAPIPE_AVAILABLE:
        logger.error("❌ MediaPipe not available")
        gesture_active = False
        return

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("❌ Camera not available")
        gesture_active = False
        return

    logger.info(f"✅ Gesture Mode ON: {mode.upper()} (MediaPipe)")
    
    # Colors
    COLOR_GREEN = (0, 255, 0)
    COLOR_CYAN = (255, 255, 0)
    COLOR_WHITE = (255, 255, 255)
    COLOR_RED = (0, 0, 255)
    COLOR_DARK_BG = (40, 40, 40)
    
    # Calibration variables
    calibrating = True
    calibration_frames = 0
    CALIBRATION_REQUIRED = 30  # 30 frames for calibration
    # IMPROVED: Wider default range for smoother control
    min_distance = 40   # Minimum finger distance (fingers close together)
    max_distance = 250  # Maximum finger distance (fingers spread apart)
    MIN_RANGE_SIZE = 150  # Minimum range size to prevent too-sensitive control
    
    last_ratio = 0.5
    
    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    ) as hands:
        
        while gesture_active and gesture_mode == mode:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Convert to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            value_text = "..."
            hand_detected = False
            
            if results.multi_hand_landmarks:
                hand_detected = True
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Draw landmarks
                points = draw_hand_landmarks(frame, hand_landmarks, w, h)
                
                # Get thumb and index fingertip positions
                thumb_tip = points[THUMB_TIP]
                index_tip = points[INDEX_TIP]
                
                # Calculate distance
                distance = math.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])
                
                # Draw line between thumb and index
                cv2.line(frame, thumb_tip, index_tip, (255, 0, 255), 3)
                
                # Calibration phase
                if calibrating:
                    calibration_frames += 1
                    progress = int((calibration_frames / CALIBRATION_REQUIRED) * 100)
                    
                    # Show calibration message
                    cv2.rectangle(frame, (w//2 - 200, h//2 - 40), (w//2 + 200, h//2 + 60), COLOR_DARK_BG, cv2.FILLED)
                    cv2.putText(frame, "Scanning your hand...", (w//2 - 150, h//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_CYAN, 2)
                    cv2.putText(frame, f"Please wait: {progress}%", (w//2 - 100, h//2 + 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
                    
                    # Update calibration bounds
                    if calibration_frames == 1:
                        min_distance = distance
                        max_distance = distance
                    else:
                        min_distance = min(min_distance, distance)
                        max_distance = max(max_distance, distance)
                    
                    if calibration_frames >= CALIBRATION_REQUIRED:
                        calibrating = False
                        # Expand range slightly
                        range_size = max_distance - min_distance
                        
                        # IMPROVED: Enforce minimum range size for smoother control
                        if range_size < MIN_RANGE_SIZE:
                            # Expand the range to at least MIN_RANGE_SIZE pixels
                            center = (min_distance + max_distance) / 2
                            min_distance = max(30, center - MIN_RANGE_SIZE / 2)
                            max_distance = center + MIN_RANGE_SIZE / 2
                            logger.info(f"⚠️ Range too narrow, expanded to: {min_distance:.0f} - {max_distance:.0f} px")
                        else:
                            # Normal expansion
                            min_distance = max(30, min_distance - range_size * 0.2)
                            max_distance = min(350, max_distance + range_size * 0.2)
                        
                        logger.info(f"✅ Calibration complete: {min_distance:.0f} - {max_distance:.0f} px (range: {max_distance - min_distance:.0f}px)")
                
                else:
                    # Normal operation - control volume/brightness
                    ratio = _map(distance, min_distance, max_distance, 0, 1)
                    
                    # Smooth the value
                    ratio = 0.7 * ratio + 0.3 * last_ratio
                    last_ratio = ratio
                    
                    if mode == "volume" and volume_interface:
                        try:
                            volume_interface.SetMasterVolumeLevelScalar(float(ratio), None)
                            current_vol = volume_interface.GetMasterVolumeLevelScalar()
                            value_text = f"{int(current_vol * 100)}%"
                        except Exception:
                            value_text = "Error"
                    
                    elif mode == "brightness" and SBC_AVAILABLE:
                        bright_val = int(ratio * 100)
                        try:
                            sbc.set_brightness(bright_val)
                            value_text = f"{bright_val}%"
                        except Exception:
                            value_text = "Error"
                    
                    else:
                        value_text = f"{int(ratio * 100)}%"
            
            else:
                # No hand detected
                if calibrating:
                    cv2.putText(frame, "Show your hand to start", (w//2 - 150, h//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_RED, 2)
                else:
                    cv2.putText(frame, "Hand lost - show your hand", (w//2 - 170, h//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_RED, 2)
            
            # ===== SIDE PANEL =====
            if not calibrating:
                panel_width = 120
                panel_x = w - panel_width
                
                overlay = frame.copy()
                cv2.rectangle(overlay, (panel_x, 0), (w, h), COLOR_DARK_BG, cv2.FILLED)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                # Mode label
                mode_label = mode.upper() + " MODE"
                cv2.putText(frame, mode_label, (panel_x + 5, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CYAN, 2)
                
                # Vertical bar
                bar_x = panel_x + 40
                bar_y = 60
                bar_w = 40
                bar_h = h - 120
                
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), cv2.FILLED)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COLOR_WHITE, 2)
                
                current_bar_h = int(bar_h * last_ratio)
                fill_color = (0, int(200 * last_ratio + 55), 0)
                cv2.rectangle(frame, (bar_x + 2, bar_y + bar_h - current_bar_h), 
                             (bar_x + bar_w - 2, bar_y + bar_h - 2), fill_color, cv2.FILLED)
                
                cv2.putText(frame, value_text, (panel_x + 25, bar_y + bar_h + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2)
            
            # ===== BOTTOM BAR =====
            cv2.rectangle(frame, (0, h - 40), (w - 120 if not calibrating else w, h), COLOR_DARK_BG, cv2.FILLED)
            cv2.putText(frame, "HAND DETECTION & GESTURE RECOGNITION", (10, h - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
            cv2.putText(frame, "Press Q to exit", (w - 250 if calibrating else w - 370, h - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
            
            cv2.imshow("Emily - Gesture Control (MediaPipe)", frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
    # Cleanup - must be done in same thread that created the window
    try:
        cap.release()
    except:
        pass
    
    try:
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Process the destroy
    except:
        pass
    
    gesture_active = False
    gesture_mode = "none"
    logger.info("🛑 Gesture mode OFF")

def _start_thread(mode):
    """Start gesture control thread (internal)."""
    global gesture_active, gesture_mode, gesture_thread

    if not MEDIAPIPE_AVAILABLE:
        return "MediaPipe not available boss. Install it with: pip install mediapipe"

    if gesture_active:
        return "Gesture mode is already running boss. Say stop gesture to turn it off first."

    if mode == "volume" and (not PYCAW_AVAILABLE or volume_interface is None):
        return "Volume control unavailable boss. Install pycaw first."

    if mode == "brightness" and not SBC_AVAILABLE:
        return "Brightness control unavailable boss. Install screen-brightness-control first."

    gesture_active = True
    gesture_mode = mode
    gesture_thread = threading.Thread(target=_gesture_loop_mediapipe, args=(mode,), daemon=True)
    gesture_thread.start()

    return f"{mode.capitalize()} gesture mode starting boss! Show your hand to the camera for calibration."


# ══════════════════════════════════════════════════════════════════════
# Public API — used by LLMHandler and main.py
# ══════════════════════════════════════════════════════════════════════

def start_gesture(mode: str = "general") -> str:
    """
    Start gesture control.
    mode: 'general' | 'volume' | 'brightness'
    """
    mode = mode.lower().strip()
    if mode not in ("general", "volume", "brightness"):
        mode = "general"
    return _start_thread(mode)


def stop_gesture() -> str:
    """Stop any active gesture control mode."""
    global gesture_active, gesture_mode

    if not gesture_active:
        return "No gesture mode is running right now boss."

    # Just set flags - the gesture thread will handle cleanup
    gesture_active = False
    gesture_mode = "none"
    
    # Wait for thread to finish (it will close OpenCV window itself)
    time.sleep(0.5)
    return "Gesture mode stopped boss."


def is_gesture_active() -> bool:
    """Check if gesture control is currently running."""
    return gesture_active


# ── quick test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing gesture control...")
    print(f"MediaPipe available: {MEDIAPIPE_AVAILABLE}")
    print(f"Pycaw available: {PYCAW_AVAILABLE}")
    print(f"Screen brightness available: {SBC_AVAILABLE}")
    if MEDIAPIPE_AVAILABLE:
        result = start_gesture("general")
        print(result)
        input("Press Enter to stop...")
        print(stop_gesture())
    else:
        print("Install mediapipe first: pip install mediapipe")
import base64
import json
import math
import re
from pathlib import Path
from typing import List, Optional

import cv2
import httpx
import numpy as np
import pytesseract
from fastapi import HTTPException
from pydantic import BaseModel
from shapely.geometry import LineString, Point, Polygon
from openai import OpenAI

from config import api_key
from general import app

# Ensure pytesseract points to the installed tesseract binary on Windows hosts safely
try:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except Exception:
    pass

# --- CONSTANTS & CALIBRATION ---
PIXELS_PER_METER = 10.0
CABLE_BUFFER_MULTIPLIER = 1.15
VISION_MODEL = "google/gemma-4-31B-turbo-TEE"

# --- PYDANTIC REQUEST/RESPONSE CONTRACT SCHEMAS ---

class CameraArrangementItem(BaseModel):
    id: int
    zone: str
    position: str
    purpose: str

class FloorPlanAnalysisRequest(BaseModel):
    floor_plan_b64: str  # Supports data URL prefix or raw base64 string configurations
    camera_arrangement: Optional[List[CameraArrangementItem]] = None
    camera_count: Optional[int] = None

class FloorPlanImageResponse(BaseModel):
    analysis_image_b64: str
    total_cable_meters: float

# --- ORIGINAL CORE GEOMETRIC & OCR VISUAL PROCESSING PIPELINE ---

def detect_walls_and_border(img, text_boxes=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if text_boxes:
        for b in text_boxes:
            x, y, w, h = b['box']
            padding = 10
            cv2.rectangle(
                gray, 
                (max(0, x - padding), max(0, y - padding)), 
                (min(gray.shape[1], x + w + padding), min(gray.shape[0], y + h + padding)), 
                255, 
                -1
            )

    _, binary = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    eroded = cv2.erode(binary, kernel, iterations=2)
    
    thick_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    dilated = cv2.dilate(eroded, thick_kernel, iterations=3)
    
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, thick_kernel, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No structural walls could be isolated from layout visuals.")
        
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    outer = contours_sorted[0]
    outer_poly = Polygon([tuple(pt[0]) for pt in outer])

    h, w = img.shape[:2]
    max_dim = max(h, w)
    
    min_line_len = int(max_dim * 0.05)
    hough_thresh = max(50, int(max_dim * 0.03))
    
    lines = cv2.HoughLinesP(closed, 1, math.pi / 180, hough_thresh, minLineLength=min_line_len, maxLineGap=50)
    wall_segments = []
    
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            wall_segments.append(((x1, y1), (x2, y2)))

    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            pts = [tuple(pt[0]) for pt in cnt]
            for i in range(len(pts)):
                wall_segments.append((pts[i], pts[(i + 1) % len(pts)]))

    return outer_poly, wall_segments


def extract_text_boxes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    boxes = []
    n = len(data['text'])
    for i in range(n):
        txt = data['text'][i].strip()
        conf = int(data['conf'][i]) if data['conf'][i] != '-1' else -1
        if txt and conf > 30:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            cx, cy = x + w / 2, y + h / 2
            boxes.append({'index': i, 'text': txt, 'conf': conf, 'box': (x, y, w, h), 'center': (cx, cy)})
    return boxes


def find_label_center(boxes, label, img_shape=None, used_indices=None):
    used = set(used_indices or [])
    lab = label.lower().strip()
    words = [w for w in lab.split() if w]
    if not words:
        return None

    spatial_hint = None
    for hint in ['top', 'bottom', 'left', 'right']:
        if hint in words:
            spatial_hint = hint
            words.remove(hint)
            break
    
    clean_label = ' '.join(words)
    candidates = []
    
    for b in boxes:
        if b['index'] in used:
            continue
        if clean_label in b['text'].lower() or any(w in b['text'].lower() for w in words):
            candidates.append(b)
            
    if not candidates:
        return None

    if spatial_hint and img_shape:
        if spatial_hint == 'top':
            candidates.sort(key=lambda b: b['center'][1])
        elif spatial_hint == 'bottom':
            candidates.sort(key=lambda b: b['center'][1], reverse=True)
        elif spatial_hint == 'left':
            candidates.sort(key=lambda b: b['center'][0])
        elif spatial_hint == 'right':
            candidates.sort(key=lambda b: b['center'][0], reverse=True)
    
    selected_box = candidates[0]
    selected_group = [selected_box]
    
    for b in candidates[1:]:
        if abs(b['center'][1] - selected_box['center'][1]) < 16 and abs(b['center'][0] - selected_box['center'][0]) < 150:
            selected_group.append(b)

    cx = sum(bb['center'][0] for bb in selected_group) / len(selected_group)
    cy = sum(bb['center'][1] for bb in selected_group) / len(selected_group)
    
    return (cx, cy), [bb['index'] for bb in selected_group]


def nearest_point_on_segments(pt, segments):
    px, py = pt
    best = None
    best_dist = float('inf')
    for a, b in segments:
        ax, ay = a
        bx, by = b
        vx, vy = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        denom = vx * vx + vy * vy
        if denom == 0:
            continue
            
        t = (wx * vx + wy * vy) / denom
        t = max(0, min(1, t))
        projx = ax + t * vx
        projy = ay + t * vy
        d = math.hypot(px - projx, py - projy)
        if d < best_dist:
            best_dist = d
            best = (projx, projy)
    return best


def nearest_point_with_min_distance(pt, segments, existing_points, min_dist=40.0):
    px, py = pt
    candidates = []
    
    for a, b in segments:
        ax, ay = a
        bx, by = b
        vx, vy = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        denom = vx * vx + vy * vy
        if denom == 0:
            continue
            
        t = (wx * vx + wy * vy) / denom
        t = max(0, min(1, t))
        projx = ax + t * vx
        projy = ay + t * vy
        
        base_dist = math.hypot(px - projx, py - projy)
        too_close = False
        
        for ex, ey in existing_points:
            if math.hypot(projx - ex, projy - ey) < min_dist:
                too_close = True
                break
        
        if too_close:
            segment_len = math.hypot(vx, vy)
            if segment_len > 0:
                for shift in [min_dist, -min_dist]:
                    t_shifted = t + (shift / segment_len)
                    t_shifted = max(0, min(1, t_shifted))
                    sx = ax + t_shifted * vx
                    sy = ay + t_shifted * vy
                    
                    still_too_close = False
                    for ex, ey in existing_points:
                        if math.hypot(sx - ex, sy - ey) < min_dist:
                            still_too_close = True
                            break
                    if not still_too_close:
                        d_shifted = math.hypot(px - sx, py - sy)
                        candidates.append((d_shifted, (sx, sy)))
        else:
            candidates.append((base_dist, (projx, projy)))
            
    if not candidates:
        return nearest_point_on_segments(pt, segments)
        
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def generate_bounded_fov_points(cam_pos, target_pos, wall_segments, fov_deg=110, max_dist=450, steps=60):
    cx, cy = cam_pos
    tx, ty = target_pos
    
    base_angle = math.atan2(ty - cy, tx - cx)
    half_fov = math.radians(fov_deg) / 2
    
    start_angle = base_angle - half_fov
    end_angle = base_angle + half_fov
    
    shapely_walls = [LineString([a, b]) for a, b in wall_segments if math.hypot(b[0]-a[0], b[1]-a[1]) > 2]
    cam_point = Point(cx, cy)
    points = [(int(round(cx)), int(round(cy)))]
    
    for i in range(steps + 1):
        angle = start_angle + i * (end_angle - start_angle) / steps
        rx = cx + math.cos(angle) * max_dist
        ry = cy + math.sin(angle) * max_dist
        
        ray_line = LineString([(cx, cy), (rx, ry)])
        closest_hit_dist = max_dist
        
        for wall in shapely_walls:
            if wall.distance(cam_point) < 5.0:
                continue
                
            if ray_line.intersects(wall):
                intersection = ray_line.intersection(wall)
                if intersection.geom_type == 'Point':
                    d = cam_point.distance(intersection)
                    if 5.0 < d < closest_hit_dist:
                        closest_hit_dist = d
                elif intersection.geom_type == 'MultiPoint':
                    for p in intersection.geoms:
                        d = cam_point.distance(p)
                        if 5.0 < d < closest_hit_dist:
                            closest_hit_dist = d

        final_x = cx + math.cos(angle) * closest_hit_dist
        final_y = cy + math.sin(angle) * closest_hit_dist
        points.append((int(round(final_x)), int(round(final_y))))
        
    return points


def calculate_geometric_median(points, eps=1e-5, max_iter=200):
    pts = np.array(points, dtype=np.float64)
    if len(pts) == 0:
        return (0.0, 0.0)
        
    median = np.mean(pts, axis=0)
    for _ in range(max_iter):
        distances = np.linalg.norm(pts - median, axis=1)
        distances = np.where(distances < eps, eps, distances)
        
        weights = 1.0 / distances
        total_weight = np.sum(weights)
        
        next_median = np.sum(pts * weights[:, np.newaxis], axis=0) / total_weight
        if np.linalg.norm(next_median - median) < eps:
            break
        median = next_median
        
    return float(median[0]), float(median[1])

# --- MODIFIED RENDERED COGNITIVE DESIGNER ENGINE ---

def draw_results_to_bytes(img, cameras, recorder_pos, buffered_meters) -> str:
    """Renders all elements in memory and encodes them back straight to a base64 string."""
    overlay = img.copy()
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]  # BGR Matrix configurations
    alpha = 0.35

    # 1. Render FOV sectors
    for i, cam in enumerate(cameras):
        color = colors[i % len(colors)]
        fov_pts = cam.get('fov_points', [])
        if len(fov_pts) >= 3:
            cv2.fillPoly(overlay, [np.array(fov_pts, dtype=np.int32)], color=color)

    blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    rx, ry = int(round(recorder_pos[0])), int(round(recorder_pos[1]))

    # 2. Draw routing links from cameras straight across to NVR terminals
    for i, cam in enumerate(cameras):
        wp = cam.get('wall_point')
        if wp:
            cx, cy = int(round(wp[0])), int(round(wp[1]))
            cv2.line(blended, (cx, cy), (rx, ry), (110, 110, 110), 1, cv2.LINE_AA)

    # 3. Render camera structural layout placements
    for i, cam in enumerate(cameras):
        color = colors[i % len(colors)]
        
        lc = cam.get('label_center')
        if lc:
            cv2.circle(blended, (int(round(lc[0])), int(round(lc[1]))), 4, (0, 0, 0), -1)
            cv2.circle(blended, (int(round(lc[0])), int(round(lc[1]))), 2, (255, 255, 255), -1)

        wp = cam.get('wall_point')
        if wp:
            wx, wy = int(round(wp[0])), int(round(wp[1]))
            cv2.circle(blended, (wx, wy), 7, (0, 0, 0), -1)
            cv2.circle(blended, (wx, wy), 5, color, -1)

            tx, ty = wx + 12, wy - 12
            label = cam.get('name', f'CAM{i+1}')
            cv2.putText(blended, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(blended, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # 4. Render optimized centralized Recorder hub node
    cv2.circle(blended, (rx, ry), 9, (0, 0, 0), -1)
    cv2.circle(blended, (rx, ry), 6, (0, 165, 255), -1)  # Vivid Orange Node
    
    cv2.putText(blended, "RECORDER", (rx + 14, ry + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(blended, "RECORDER", (rx + 14, ry + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

    # 5. Compile metric telemetry tracking boxes
    summary_text = f"Cable Needed (+15% Buffer): {buffered_meters:.2f} m"
    cv2.putText(blended, summary_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(blended, summary_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

    # Convert binary buffer mapping into text-safe b64
    _, buffer = cv2.imencode('.png', blended)
    return base64.b64encode(buffer).decode('utf-8')

# --- EXTERNAL VISION SUPPORT PROMPTS ---

ROOM_EXTRACTION_PROMPT = """/no_think
You are a floor plan analysis system. Your only task is to identify every distinct room or zone in the floor plan image and return their bounding coordinates.

COORDINATE SYSTEM (all values are fractions of the full image dimensions, range 0.0–1.0):
- x=0.0 is the LEFT edge,  x=1.0 is the RIGHT edge
- y=0.0 is the TOP edge,   y=1.0 is the BOTTOM edge
- x,y is the CENTER of the room; width,height are its approximate span

Return ONLY valid JSON — no explanation, no markdown:
{
  "rooms": [
    {"name": "living room", "x": 0.25, "y": 0.40, "width": 0.30, "height": 0.25}
  ]
}

Rules:
- Include ALL visible rooms/zones, even unlabelled ones (name them descriptively, e.g. "corridor", "unlabelled room 1").
- Estimate coordinates by mentally dividing the image into a fine grid.
- Do NOT include cameras — rooms and zones only."""

# --- VISION INTERFACE ROUTINES ---

def _call_vision(image_url: str, prompt: str) -> str:
    client = OpenAI(base_url="https://llm.chutes.ai/v1", api_key=api_key)
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ],
        }],
        temperature=0.1,
        max_tokens=1500,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL)
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]
    return raw.strip()

# --- REFACTORED WORKSPACE ENTRY ROUTINE ---

@app.post("/analyze-floor-plan", response_model=FloorPlanImageResponse)
async def analyze_floor_plan(request: FloorPlanAnalysisRequest) -> FloorPlanImageResponse:
    # 1. Normalize Base64 input syntax variants safely
    b64_data = request.floor_plan_b64
    if "," in b64_data:
        b64_data = b64_data.split(",")[1]
        
    try:
        img_bytes = base64.b64decode(b64_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid format matrix received on layout channel.")

    # 2. Extract OCR string components via local tesseract engines
    boxes = extract_text_boxes(img)
    
    # 3. Detect system layouts 
    try:
        outer_poly, wall_segments = detect_walls_and_border(img, text_boxes=boxes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Structural analysis pipeline failed: {str(e)}")

    # 4. Normalize incoming targeting strings safely
    target_cams = []
    if request.camera_arrangement:
        for item in request.camera_arrangement:
            target_cams.append({'name': f"CAM {item.id}", 'label': item.zone})
    else:
        # Step 2b Fallback Integration: Ask the Vision LLM to resolve room configurations automatically
        print("[/analyze-floor-plan] No structural configuration sent. Falling back to dynamic vision lookups...")
        image_url = request.floor_plan_b64 if request.floor_plan_b64.startswith("data:") else f"data:image/jpeg;base64,{request.floor_plan_b64}"
        try:
            raw_rooms = _call_vision(image_url, ROOM_EXTRACTION_PROMPT)
            parsed_rooms = json.loads(raw_rooms).get("rooms", [])
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Dynamic room discovery fallback failed: {str(e)}")
            
        limit = request.camera_count if request.camera_count else len(parsed_rooms)
        for idx, rm in enumerate(parsed_rooms[:limit]):
            target_cams.append({'name': f"CAM {idx+1}", 'label': rm.get("name", "Zone")})

    # 5. Calculate deployment mapping targets onto localized coordinate slots
    placed = []
    used_indices = set()
    camera_coordinates = []

    for cam in target_cams:
        label = cam['label']
        match = find_label_center(boxes, label, img_shape=img.shape, used_indices=used_indices)
        if match is None:
            print(f"Warning: Room identifier tag allocation fallback skipped: '{label}'")
            continue
            
        center, matched_indices = match
        used_indices.update(matched_indices)
        
        nearest = nearest_point_on_segments(center, wall_segments)
        if nearest is None:
            continue
            
        fov_points = generate_bounded_fov_points(
            cam_pos=nearest, 
            target_pos=center, 
            wall_segments=wall_segments,
            fov_deg=110, 
            max_dist=350
        )

        placed.append({
            'name': cam['name'],
            'label': cam['label'],
            'wall_point': nearest,
            'label_center': center,
            'fov_points': fov_points
        })
        camera_coordinates.append(nearest)

    if not placed:
        raise HTTPException(status_code=422, detail="No specific layout marker strings could be linked to physical rooms.")

    # 6. Extract spatial network medians
    raw_med_x, raw_med_y = calculate_geometric_median(camera_coordinates)
    recorder_snapped = nearest_point_with_min_distance(
        (raw_med_x, raw_med_y), 
        wall_segments, 
        camera_coordinates, 
        min_dist=40.0
    )

    # 7. Aggregate scale parameters
    total_pixel_distance = 0.0
    for c in placed:
        cx, cy = c['wall_point']
        dist = math.hypot(cx - recorder_snapped[0], cy - recorder_snapped[1])
        total_pixel_distance += dist

    total_base_meters = total_pixel_distance / PIXELS_PER_METER
    total_buffered_meters = total_base_meters * CABLE_BUFFER_MULTIPLIER

    # 8. Render visualization changes straight to data blocks
    result_b64 = draw_results_to_bytes(img, placed, recorder_snapped, total_buffered_meters)

    return FloorPlanImageResponse(
        analysis_image_b64=f"data:image/png;base64,{result_b64}",
        total_cable_meters=round(total_buffered_meters, 2)
    )
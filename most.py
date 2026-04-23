import cv2
import mediapipe as mp
import numpy as np
import math
import time
import random

# ══════════════════════════════════════════════════════════════════════
# INIT MEDIAPIPE (partagé)
# ══════════════════════════════════════════════════════════════════════
mp_face_mesh = mp.solutions.face_mesh
mp_hands     = mp.solutions.hands
mp_pose      = mp.solutions.pose
mp_draw      = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)
hands_detector = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2,
    min_detection_confidence=0.6, min_tracking_confidence=0.6
)
pose_detector = mp_pose.Pose(
    static_image_mode=False, model_complexity=1,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# ══════════════════════════════════════════════════════════════════════
# UTILITAIRES COMMUNS
# ══════════════════════════════════════════════════════════════════════
def lm_pt(landmarks, idx, w, h):
    l = landmarks.landmark[idx]
    return np.array([int(l.x * w), int(l.y * h)], dtype=np.int32)

def lm_ptf(landmarks, idx, w, h):
    l = landmarks.landmark[idx]
    return np.array([l.x * w, l.y * h], dtype=np.float32)

def dist2d(p1, p2):
    return float(np.linalg.norm(np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)))

def perpendicular_unit(p1, p2):
    v = np.array([p2[0]-p1[0], p2[1]-p1[1]], dtype=np.float32)
    n = np.linalg.norm(v)
    if n < 1e-6: return np.array([0.,-1.], dtype=np.float32)
    v /= n
    return np.array([-v[1], v[0]], dtype=np.float32)

def quadratic_bezier(p0, p1, p2, n=28):
    pts = []
    for t in np.linspace(0., 1., n):
        pt = (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2
        pts.append(pt.astype(np.int32))
    return np.array(pts, dtype=np.int32)

def get_body_points(pose_lm, W, H):
    if pose_lm is None: return []
    ids = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,  mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW,     mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST,     mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_HIP,       mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,      mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.NOSE,
    ]
    pts = []
    for pid in ids:
        p = pose_lm.landmark[pid]
        if p.visibility > 0.3:
            pts.append([int(p.x * W), int(p.y * H)])
    return pts

# ══════════════════════════════════════════════════════════════════════
# DÉTECTION GESTES
# ══════════════════════════════════════════════════════════════════════
def is_crossed_fingers(lm):
    """✌ Sharingan : index + majeur levés, majeur croisé sur index"""
    index_up  = lm.landmark[8].y  < lm.landmark[6].y
    middle_up = lm.landmark[12].y < lm.landmark[10].y
    if not (index_up and middle_up): return False
    crossed   = (lm.landmark[12].x - lm.landmark[8].x) < -0.01
    ring_dn   = lm.landmark[16].y > lm.landmark[14].y
    pinky_dn  = lm.landmark[20].y > lm.landmark[18].y
    return crossed and ring_dn and pinky_dn

def is_ily_gesture(lm):
    """🤟 Byakugan : index + auriculaire + pouce levés"""
    lmk = lm.landmark
    return (lmk[8].y  < lmk[6].y  and
            lmk[12].y > lmk[10].y and
            lmk[16].y > lmk[14].y and
            lmk[20].y < lmk[18].y and
            abs(lmk[4].x - lmk[9].x) > abs(lmk[3].x - lmk[9].x))

def is_vulcan_gesture(lm):
    """🖖 Gaara : index + majeur levés, annulaire + auriculaire pliés"""
    lmk = lm.landmark
    return (lmk[8].y  < lmk[6].y  and
            lmk[12].y < lmk[10].y and
            lmk[16].y > lmk[14].y and
            lmk[20].y > lmk[18].y and
            abs(lmk[8].x - lmk[12].x) > 0.04)

def is_fist(lm):
    """✊ Chidori : poing fermé"""
    tips = [8,12,16,20]; pips = [6,10,14,18]
    count = sum(1 for t,p in zip(tips,pips) if lm.landmark[t].y > lm.landmark[p].y)
    thumb_folded = abs(lm.landmark[4].x - lm.landmark[3].x) < 0.08
    return count >= 3 and thumb_folded

def is_index_pointing(lm):
    """☝ Un seul index tendu"""
    lmk = lm.landmark
    return (lmk[8].y < lmk[6].y < lmk[5].y and
            lmk[12].y > lmk[10].y and
            lmk[16].y > lmk[14].y and
            lmk[20].y > lmk[18].y)

def detect_cross_index_two_hands(hand_results, W, H):
    """👆👆 Chakra : deux index qui se croisent (2 mains)"""
    if not hand_results.multi_hand_landmarks: return False, None, None
    if len(hand_results.multi_hand_landmarks) < 2: return False, None, None
    h1, h2 = hand_results.multi_hand_landmarks[0], hand_results.multi_hand_landmarks[1]
    if not (is_index_pointing(h1) and is_index_pointing(h2)): return False, None, None
    t1 = lm_ptf(h1, 8, W, H)
    t2 = lm_ptf(h2, 8, W, H)
    d  = np.linalg.norm(t1 - t2)
    mid = ((t1 + t2) / 2).astype(np.int32)
    if d < W * 0.18: return True, mid, d
    return False, None, None

def detect_circle_two_hands(hand_results, W, H):
    """Rasengan : former un cercle avec 2 mains"""
    if not hand_results.multi_hand_landmarks: return False, 0, 0, 0
    if len(hand_results.multi_hand_landmarks) < 2: return False, 0, 0, 0
    h1, h2 = hand_results.multi_hand_landmarks[0], hand_results.multi_hand_landmarks[1]
    pts = [lm_ptf(h1,4,W,H), lm_ptf(h1,8,W,H), lm_ptf(h2,4,W,H), lm_ptf(h2,8,W,H)]
    all_pts = np.array(pts)
    cx, cy = int(np.mean(all_pts[:,0])), int(np.mean(all_pts[:,1]))
    center = np.array([cx,cy], dtype=np.float32)
    radii = [dist2d(p, center) for p in pts]
    r_mean = float(np.mean(radii))
    r_std  = float(np.std(radii))
    if r_mean < 30 or r_mean > 280: return False,0,0,0
    if r_std / max(r_mean,1) > 0.55: return False,0,0,0
    palm_dist = dist2d(lm_ptf(h1,9,W,H), lm_ptf(h2,9,W,H))
    if palm_dist < 60: return False,0,0,0
    hand_ref = dist2d(lm_ptf(h1,0,W,H), lm_ptf(h1,9,W,H))
    if hand_ref < 1: return False,0,0,0
    if dist2d(pts[0],pts[1])/hand_ref > 0.65: return False,0,0,0
    if dist2d(pts[2],pts[3])/hand_ref > 0.65: return False,0,0,0
    return True, cx, cy, int(r_mean)

def detect_current_mode(hand_results, W, H):
    """
    Retourne le mode actif parmi :
    'sharingan', 'byakugan', 'gaara', 'chidori', 'chakra', 'rasengan', None
    Priorité décroissante.
    """
    if not hand_results.multi_hand_landmarks:
        return None, {}

    hands_list = hand_results.multi_hand_landmarks
    n = len(hands_list)

    # Gestes à 2 mains d'abord
    if n >= 2:
        ok_chakra, mid, d = detect_cross_index_two_hands(hand_results, W, H)
        if ok_chakra:
            return 'chakra', {'mid': mid, 'dist': d}

        ok_ras, cx, cy, r = detect_circle_two_hands(hand_results, W, H)
        if ok_ras:
            return 'rasengan', {'cx': cx, 'cy': cy, 'r': r}

    # Gestes à 1 main (ou plus)
    for lm in hands_list:
        if is_crossed_fingers(lm): return 'sharingan', {'lm': lm}
        if is_ily_gesture(lm):     return 'byakugan',  {'lm': lm}
        if is_vulcan_gesture(lm):  return 'gaara',     {'lm': lm}
        if is_fist(lm):            return 'chidori',   {'lm': lm}

    return None, {}


# ══════════════════════════════════════════════════════════════════════
# ŒILS — géométrie commune (Sharingan + Byakugan)
# ══════════════════════════════════════════════════════════════════════
def build_eye_shape(landmarks, outer_id, inner_id, top_id, bottom_id,
                    iris_center_id, w, h):
    outer  = lm_pt(landmarks, outer_id,  w, h)
    inner  = lm_pt(landmarks, inner_id,  w, h)
    top    = lm_pt(landmarks, top_id,    w, h)
    bottom = lm_pt(landmarks, bottom_id, w, h)
    center = lm_pt(landmarks, iris_center_id, w, h)
    eye_width  = dist2d(outer, inner)
    eye_height = max(8., dist2d(top, bottom))
    perp = perpendicular_unit(outer.astype(float), inner.astype(float))
    mid  = (outer.astype(np.float32) + inner.astype(np.float32)) / 2.
    upper = quadratic_bezier(outer.astype(np.float32), mid - perp * eye_height * 1.05, inner.astype(np.float32))
    lower = quadratic_bezier(inner.astype(np.float32), mid + perp * eye_height * 0.75, outer.astype(np.float32))
    return {'outer': outer, 'inner': inner, 'top': top, 'bottom': bottom,
            'center': center, 'width': eye_width, 'height': eye_height,
            'poly': np.vstack([upper, lower])}

def create_eye_mask(shape_hw, eye_shape):
    h, w = shape_hw[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [eye_shape["poly"]], 255, lineType=cv2.LINE_AA)
    return mask

def clip_layer_to_eye(frame, layer, eye_shape, center, radius, opacity=0.93):
    eye_mask  = create_eye_mask(frame.shape, eye_shape)
    lens_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(lens_mask, center, radius, 255, -1, cv2.LINE_AA)
    final_mask = cv2.bitwise_and(eye_mask, lens_mask)
    mask_f = (final_mask.astype(np.float32) / 255.) * opacity
    blended = frame.astype(np.float32) * (1. - mask_f[:,:,None]) + layer.astype(np.float32) * mask_f[:,:,None]
    frame[:] = np.clip(blended, 0, 255).astype(np.uint8)

# ══════════════════════════════════════════════════════════════════════
# SHARINGAN / RINNEGAN
# ══════════════════════════════════════════════════════════════════════
def radial_gradient(layer, cx, cy, r_max, color_inner, color_outer):
    h, w = layer.shape[:2]
    Y, X = np.ogrid[:h, :w]
    d = np.sqrt((X-cx)**2 + (Y-cy)**2).astype(np.float32)
    t = np.clip(d / max(r_max, 1), 0, 1)
    mask = d <= r_max
    for c in range(3):
        layer[:,:,c] = np.where(mask,
            (color_inner[c]*(1-t) + color_outer[c]*t).astype(np.uint8),
            layer[:,:,c])
    return mask.astype(np.uint8)*255

def draw_iris_fibers(layer, cx, cy, r_iris, r_pupil, color, n=64, alpha=0.18):
    overlay = layer.copy()
    for i in range(n):
        angle = 2*math.pi*i/n + (i%3)*0.08
        noise = 0.85 + 0.15*((i*7+3)%11)/10.
        x0 = int(cx + r_pupil*1.1*math.cos(angle)); y0 = int(cy + r_pupil*1.1*math.sin(angle))
        x1 = int(cx + r_iris*noise*math.cos(angle)); y1 = int(cy + r_iris*noise*math.sin(angle))
        cv2.line(overlay, (x0,y0), (x1,y1), color, 1 if i%4!=0 else 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, layer, 1-alpha, 0, layer)

def draw_corneal_highlight(layer, cx, cy, r):
    hx, hy = int(cx - r*0.28), int(cy - r*0.30)
    cv2.ellipse(layer, (hx,hy), (max(2,int(r*0.14)), max(1,int(r*0.09))), -30, 0, 360, (255,255,255), -1, cv2.LINE_AA)
    cv2.circle(layer, (int(cx+r*0.20), int(cy-r*0.22)), max(1,int(r*0.04)), (220,220,255), -1, cv2.LINE_AA)

def draw_lid_shadow(layer, cx, cy, r, mask_2d):
    shadow = np.zeros_like(layer)
    cv2.ellipse(shadow, (cx, int(cy-r*0.05)), (r, int(r*0.55)), 0, 200, 340, (0,0,0), int(r*0.45), cv2.LINE_AA)
    mask3 = np.stack([mask_2d]*3, axis=-1).astype(np.float32)/255.
    layer[:] = np.clip(layer.astype(np.float32)*(1-mask3*0.22) + shadow.astype(np.float32)*mask3*0.22, 0, 255).astype(np.uint8)

def draw_sharingan_on_layer(layer, center, r, t):
    cx, cy = center
    iris_mask = radial_gradient(layer, cx, cy, r, (40,10,200), (10,5,90))
    cv2.circle(layer, (cx,cy), r, (5,2,40), 2, cv2.LINE_AA)
    draw_iris_fibers(layer, cx, cy, int(r*0.92), int(r*0.18), (60,40,220), 72, 0.22)
    cv2.circle(layer, (cx,cy), int(r*0.70), (0,0,0), 1, cv2.LINE_AA)
    for i in range(3):
        ang = t*1.5 + i*(2*math.pi/3)
        tx, ty = int(cx+r*0.50*math.cos(ang)), int(cy+r*0.50*math.sin(ang))
        tomoe_r = max(2, int(r*0.115))
        cv2.circle(layer, (tx,ty), tomoe_r, (0,0,0), -1, cv2.LINE_AA)
        for k in range(1,7):
            frac = k/7.; tail_ang = ang - frac*0.85
            tail_r = tomoe_r*(1-frac*0.80)
            tx2 = int(cx+(r*0.50-frac*r*0.30)*math.cos(tail_ang))
            ty2 = int(cy+(r*0.50-frac*r*0.30)*math.sin(tail_ang))
            if tail_r >= 1: cv2.circle(layer, (tx2,ty2), int(tail_r), (0,0,0), -1, cv2.LINE_AA)
    pupil_r = max(3, int(r*0.17))
    cv2.circle(layer, (cx,cy), pupil_r+2, (30,10,60), -1, cv2.LINE_AA)
    cv2.circle(layer, (cx,cy), pupil_r,   (0,0,0),    -1, cv2.LINE_AA)
    draw_lid_shadow(layer, cx, cy, r, iris_mask)
    draw_corneal_highlight(layer, cx, cy, r)

def draw_rinnegan_on_layer(layer, center, r):
    cx, cy = center
    iris_mask = radial_gradient(layer, cx, cy, r, (210,185,240), (100,70,140))
    for i in range(1, 9):
        rr = max(1, int(r*i/8))
        ov = layer.copy(); cv2.circle(ov, (cx,cy), rr, (70,45,105), 1, cv2.LINE_AA)
        a = 0.55 if i%2==0 else 0.30; cv2.addWeighted(ov, a, layer, 1-a, 0, layer)
    hex_r = int(r*0.44)
    for i in range(6):
        ang = i*math.pi/3
        px, py = int(cx+hex_r*math.cos(ang)), int(cy+hex_r*math.sin(ang))
        cv2.circle(layer, (px,py), max(1,int(r*0.065)), (70,45,105), -1, cv2.LINE_AA)
        nx, ny = int(cx+r*0.72*math.cos(ang)), int(cy+r*0.72*math.sin(ang))
        ov2 = layer.copy(); cv2.line(ov2, (px,py), (nx,ny), (70,45,105), 1, cv2.LINE_AA)
        cv2.addWeighted(ov2, 0.35, layer, 0.65, 0, layer)
    cv2.circle(layer, (cx,cy), r, (55,30,90), 2, cv2.LINE_AA)
    draw_iris_fibers(layer, cx, cy, int(r*0.90), int(r*0.16), (90,60,140), 56, 0.15)
    pupil_r = max(3, int(r*0.14))
    cv2.circle(layer, (cx,cy), pupil_r+2, (45,25,75), -1, cv2.LINE_AA)
    cv2.circle(layer, (cx,cy), pupil_r,   (8,5,18),   -1, cv2.LINE_AA)
    draw_lid_shadow(layer, cx, cy, r, iris_mask)
    draw_corneal_highlight(layer, cx, cy, r)

def render_sharingan(frame, face_landmarks, w, h, t):
    eyes = [
        dict(outer_id=33,  inner_id=133, top_id=159, bottom_id=145, iris_center_id=468, style='sharingan'),
        dict(outer_id=362, inner_id=263, top_id=386, bottom_id=374, iris_center_id=473, style='rinnegan'),
    ]
    for e in eyes:
        eye_shape = build_eye_shape(face_landmarks, e['outer_id'], e['inner_id'], e['top_id'], e['bottom_id'], e['iris_center_id'], w, h)
        center = tuple(eye_shape['center'])
        r = max(8, int(min(eye_shape['width']*0.21, eye_shape['height']*0.65)))
        layer = np.zeros_like(frame)
        if e['style'] == 'sharingan': draw_sharingan_on_layer(layer, center, r, t)
        else: draw_rinnegan_on_layer(layer, center, r)
        clip_layer_to_eye(frame, layer, eye_shape, center, r)

# ══════════════════════════════════════════════════════════════════════
# BYAKUGAN
# ══════════════════════════════════════════════════════════════════════
def draw_byakugan_on_layer(layer, center, r):
    cx, cy = int(center[0]), int(center[1])
    COLOR_BASE     = (210,200,210); COLOR_MID    = (190,180,195)
    COLOR_RING_OUT = (130,120,140); COLOR_LINES  = (155,145,160)
    COLOR_PUPIL_OUT= (160,150,165); COLOR_PUPIL_IN= (230,225,235)
    COLOR_DOT      = (140,130,148); COLOR_OUTLINE = (80,70,90)
    cv2.circle(layer, (cx,cy), r, COLOR_BASE, -1, cv2.LINE_AA)
    for rr, col, thick in [(int(r*0.85),COLOR_MID,2),(int(r*0.68),COLOR_MID,1),(int(r*0.50),COLOR_BASE,1)]:
        cv2.circle(layer, (cx,cy), rr, col, thick, cv2.LINE_AA)
    for i in range(16):
        ang = 2*math.pi*i/16
        cv2.line(layer, (int(cx+r*0.22*math.cos(ang)), int(cy+r*0.22*math.sin(ang))),
                        (int(cx+r*0.82*math.cos(ang)), int(cy+r*0.82*math.sin(ang))), COLOR_LINES, 1, cv2.LINE_AA)
    for i in range(32):
        if i%2==0: continue
        ang = 2*math.pi*i/32
        cv2.line(layer, (int(cx+r*0.38*math.cos(ang)), int(cy+r*0.38*math.sin(ang))),
                        (int(cx+r*0.72*math.cos(ang)), int(cy+r*0.72*math.sin(ang))), COLOR_LINES, 1, cv2.LINE_AA)
    cv2.circle(layer, (cx,cy), r,          COLOR_RING_OUT, 3, cv2.LINE_AA)
    cv2.circle(layer, (cx,cy), r-3,        COLOR_RING_OUT, 1, cv2.LINE_AA)
    cv2.circle(layer, (cx,cy), int(r*0.88),COLOR_MID,      1, cv2.LINE_AA)
    pupil_r = max(4, int(r*0.28))
    cv2.circle(layer, (cx,cy), pupil_r,           COLOR_PUPIL_OUT, -1, cv2.LINE_AA)
    cv2.circle(layer, (cx,cy), int(pupil_r*0.72), COLOR_PUPIL_IN,  -1, cv2.LINE_AA)
    cv2.circle(layer, (cx,cy), max(1,int(pupil_r*0.22)), COLOR_DOT, -1, cv2.LINE_AA)
    cv2.circle(layer, (cx,cy), r, COLOR_OUTLINE, 2, cv2.LINE_AA)
    cv2.ellipse(layer, (int(cx-r*0.28), int(cy-r*0.30)),
                (max(2,int(r*0.13)), max(1,int(r*0.08))), -30, 0, 360, (255,255,255), -1, cv2.LINE_AA)

def render_byakugan(frame, face_landmarks, w, h):
    eyes = [
        dict(outer_id=33,  inner_id=133, top_id=159, bottom_id=145, iris_center_id=468),
        dict(outer_id=362, inner_id=263, top_id=386, bottom_id=374, iris_center_id=473),
    ]
    for e in eyes:
        eye_shape = build_eye_shape(face_landmarks, e['outer_id'], e['inner_id'], e['top_id'], e['bottom_id'], e['iris_center_id'], w, h)
        center = tuple(eye_shape['center'])
        r = max(8, int(min(eye_shape['width']*0.28, eye_shape['height']*0.78)))
        layer = np.zeros_like(frame)
        draw_byakugan_on_layer(layer, center, r)
        clip_layer_to_eye(frame, layer, eye_shape, center, r, opacity=0.97)

# ══════════════════════════════════════════════════════════════════════
# GAARA — SABLE DORÉ
# ══════════════════════════════════════════════════════════════════════
SAND_COLORS = [(30,140,210),(45,165,230),(60,185,245),(80,205,255),(100,220,255),(120,230,255)]

class SandParticle:
    def __init__(self, W, H):
        self.W=W; self.H=H; self.reset(random.random())
    def reset(self, phase_offset=0.):
        self.x=random.uniform(0,self.W); self.y=random.uniform(self.H*0.75,self.H)
        self.vx=random.uniform(-1.5,1.5); self.vy=random.uniform(-4.5,-1.5)
        self.life=random.uniform(0.3,1.); self.max_life=self.life
        self.size=random.randint(1,4); self.color=random.choice(SAND_COLORS)
        self.angular_v=random.uniform(-0.08,0.08); self.orbit_r=random.uniform(0,30)
        self.orbit_angle=random.uniform(0,math.pi*2)
    def update(self, dt, body_cx, body_cy, strength):
        dx=body_cx-self.x; dy=body_cy-self.y
        d=max(1,math.sqrt(dx*dx+dy*dy)); attract=strength*180./(d+60)
        self.vx+=(dx/d)*attract*dt; self.vy+=(dy/d)*attract*dt
        self.orbit_angle+=self.angular_v*strength*3
        self.x+=self.vx*dt*60+math.cos(self.orbit_angle)*self.orbit_r*strength*dt*2
        self.y+=self.vy*dt*60+math.sin(self.orbit_angle)*self.orbit_r*strength*dt*2+0.15
        self.life-=dt*0.6
    def is_dead(self): return self.life<=0 or self.y<-50
    def draw(self, frame):
        alpha=max(0,self.life/self.max_life); cx,cy=int(self.x),int(self.y)
        H,W=frame.shape[:2]
        if not (0<=cx<W and 0<=cy<H): return
        color=tuple(int(c*alpha) for c in self.color); r=max(1,self.size)
        cv2.circle(frame,(cx,cy),r,color,-1,cv2.LINE_AA)
        if r>=3: cv2.circle(frame,(cx,cy),r+3,tuple(int(c*alpha*0.35) for c in self.color),1,cv2.LINE_AA)

class SpiralParticle:
    def __init__(self, body_cx, body_cy, W, H, index):
        self.W=W; self.H=H; n_spirals=3
        self.spiral_id=index%n_spirals
        self.angle=(2*math.pi*index/25)+(2*math.pi*self.spiral_id/n_spirals)
        self.radius=random.uniform(W*0.08,W*0.28); self.speed=random.uniform(1.2,2.8)
        self.life=random.uniform(0.5,1.5); self.max_life=self.life
        self.size=random.randint(1,5); self.color=random.choice(SAND_COLORS)
        self.body_cx=body_cx; self.body_cy=body_cy
        self.y_offset=random.uniform(-H*0.3,H*0.3); self.drift=random.uniform(-0.5,0.5)
    def update(self, dt, body_cx, body_cy, strength):
        self.body_cx=body_cx; self.body_cy=body_cy
        self.angle+=self.speed*dt*strength*2.5
        self.radius=max(20,self.radius-dt*15*strength)
        self.y_offset+=self.drift*dt*10; self.life-=dt*0.4
    def is_dead(self): return self.life<=0 or self.radius<20
    def get_pos(self):
        return (int(self.body_cx+self.radius*math.cos(self.angle)),
                int(self.body_cy+self.y_offset+self.radius*math.sin(self.angle)*0.45))
    def draw(self, frame):
        alpha=max(0,self.life/self.max_life); cx,cy=self.get_pos()
        H,W=frame.shape[:2]
        if not (0<=cx<W and 0<=cy<H): return
        color=tuple(int(c*alpha) for c in self.color); r=max(1,int(self.size*(0.5+0.5*alpha)))
        cv2.circle(frame,(cx,cy),r,color,-1,cv2.LINE_AA)

class SandWaveParticle:
    def __init__(self, W, H):
        self.W=W; self.H=H; self.reset()
    def reset(self):
        self.x=random.uniform(0,self.W); self.y=self.H+random.uniform(0,40)
        self.vy=random.uniform(-3.,-0.8); self.vx=random.uniform(-0.8,0.8)
        self.life=random.uniform(0.6,1.8); self.max_life=self.life
        self.size=random.randint(1,6); self.color=random.choice(SAND_COLORS)
        self.wave_freq=random.uniform(1.5,4.); self.wave_amp=random.uniform(5,25)
        self.wave_phase=random.uniform(0,math.pi*2); self.t0=time.time()
    def update(self, dt, strength):
        elapsed=time.time()-self.t0
        self.x+=self.vx*dt*60+math.sin(self.wave_freq*elapsed+self.wave_phase)*self.wave_amp*dt
        self.y+=self.vy*dt*60*strength; self.life-=dt*0.45
    def is_dead(self): return self.life<=0 or self.y<-30
    def draw(self, frame):
        alpha=max(0,self.life/self.max_life); cx,cy=int(self.x),int(self.y)
        H,W=frame.shape[:2]
        if not (0<=cx<W and 0<=cy<H): return
        color=tuple(int(c*alpha) for c in self.color); r=max(1,int(self.size*alpha))
        cv2.circle(frame,(cx,cy),r,color,-1,cv2.LINE_AA)

def draw_sand_trails(frame, spiral_particles, strength):
    if len(spiral_particles) < 2: return
    by_spiral = {}
    for p in spiral_particles:
        by_spiral.setdefault(p.spiral_id, []).append(p)
    for sid, group in by_spiral.items():
        group_sorted = sorted(group, key=lambda p: p.angle)
        for i in range(len(group_sorted)-1):
            p1,p2=group_sorted[i],group_sorted[i+1]
            x1,y1=p1.get_pos(); x2,y2=p2.get_pos()
            if math.sqrt((x2-x1)**2+(y2-y1)**2) < 80:
                alpha=min(p1.life/p1.max_life, p2.life/p2.max_life)
                color=tuple(int(c*alpha*strength*0.5) for c in random.choice(SAND_COLORS))
                cv2.line(frame,(x1,y1),(x2,y2),color,1,cv2.LINE_AA)

def draw_gaara_eyes(frame, landmarks, W, H, strength):
    for iris_id, outer_id, inner_id in [(468,33,133),(473,362,263)]:
        iris_lm=landmarks.landmark[iris_id]
        cx,cy=int(iris_lm.x*W),int(iris_lm.y*H)
        outer=lm_ptf(landmarks,outer_id,W,H); inner=lm_ptf(landmarks,inner_id,W,H)
        r=max(5,int(np.linalg.norm(outer-inner)*0.18))
        layer=np.zeros((H,W,3),dtype=np.uint8)
        cv2.circle(layer,(cx,cy),r,          (160,185,60),-1,cv2.LINE_AA)
        cv2.circle(layer,(cx,cy),int(r*0.80),(175,200,80),-1,cv2.LINE_AA)
        cv2.circle(layer,(cx,cy),int(r*0.50),(190,215,100),-1,cv2.LINE_AA)
        cv2.circle(layer,(cx,cy),int(r*0.22),(210,225,180),-1,cv2.LINE_AA)
        cv2.circle(layer,(cx,cy),r,(30,25,20),2,cv2.LINE_AA)
        cv2.circle(layer,(int(cx-r*0.25),int(cy-r*0.28)),max(1,int(r*0.15)),(230,240,255),-1,cv2.LINE_AA)
        mask=np.zeros((H,W),dtype=np.uint8); cv2.circle(mask,(cx,cy),r,255,-1,cv2.LINE_AA)
        mask3=np.stack([mask]*3,axis=-1).astype(np.float32)/255.*strength*0.95
        frame[:]=np.clip(frame.astype(np.float32)*(1-mask3)+layer.astype(np.float32)*mask3,0,255).astype(np.uint8)

def draw_ground_wave(frame, t, strength, W, H):
    layer=np.zeros((H,W,3),dtype=np.uint8)
    base_y=int(H*0.88); wave_height=int(H*0.12*strength)
    for color,opacity,freq,speed,amp_frac in [
        (SAND_COLORS[0],0.9,2.5,1.2,1.0),(SAND_COLORS[2],0.7,3.2,2.0,0.75),(SAND_COLORS[4],0.5,4.0,2.8,0.55)]:
        pts_top=[(x, base_y-int(wave_height*amp_frac*(0.5+0.5*math.sin(freq*x/W*math.pi*2+t*speed)))) for x in range(0,W,3)]
        poly=np.array(pts_top+[(W,H),(0,H)],dtype=np.int32)
        tmp=np.zeros_like(layer); cv2.fillPoly(tmp,[poly],color)
        cv2.addWeighted(layer,1.,tmp,opacity,0,layer)
    cv2.addWeighted(frame,1.,layer,strength*0.75,0,frame)

# ══════════════════════════════════════════════════════════════════════
# CHIDORI
# ══════════════════════════════════════════════════════════════════════
_chidori_bolt_buf = _chidori_aura_buf = _chidori_body_buf = None

def _get_chidori_bufs(fh, fw):
    global _chidori_bolt_buf, _chidori_aura_buf, _chidori_body_buf
    if _chidori_bolt_buf is None or _chidori_bolt_buf.shape[:2] != (fh, fw):
        _chidori_bolt_buf  = np.zeros((fh,fw,3),dtype=np.float32)
        _chidori_aura_buf  = np.zeros((fh,fw,3),dtype=np.float32)
        _chidori_body_buf  = np.zeros((fh,fw,3),dtype=np.float32)
    return _chidori_bolt_buf, _chidori_aura_buf, _chidori_body_buf

def _draw_bolt_tree(buf, x0, y0, angle, length, depth, pw, ph, base_col=(160,220,255), jitter=0.55):
    if depth <= 0 or length < 4: return
    angle += np.random.uniform(-jitter, jitter)
    x1=int(x0+length*math.cos(angle)); y1=int(y0+length*math.sin(angle))
    x1c=np.clip(x1,0,pw-1); y1c=np.clip(y1,0,ph-1)
    thick=max(1,depth-1); bright=0.4+0.6*(depth/5.)
    col=tuple(int(c*bright) for c in base_col)
    cv2.line(buf,(int(x0),int(y0)),(x1c,y1c),col,thick,cv2.LINE_AA)
    _draw_bolt_tree(buf,x1,y1,angle,length*0.72,depth-1,pw,ph,base_col,jitter*0.9)
    if np.random.rand()>0.38:
        _draw_bolt_tree(buf,x1,y1,angle+np.random.uniform(0.4,1.1),length*np.random.uniform(0.35,0.55),depth-1,pw,ph,base_col,jitter)
    if depth>=4 and np.random.rand()>0.72:
        _draw_bolt_tree(buf,x1,y1,angle-np.random.uniform(0.3,0.9),length*np.random.uniform(0.25,0.45),depth-2,pw,ph,base_col,jitter)

def draw_chidori(frame, palm_cx, palm_cy, wrist_x, wrist_y, body_points, t, hand_size):
    fh,fw=frame.shape[:2]
    bolt_buf,aura_buf,body_buf=_get_chidori_bufs(fh,fw)
    R=int(hand_size*1.2)
    pad=min(fh,fw)//2
    px0=max(0,palm_cx-pad); px1=min(fw,palm_cx+pad)
    py0=max(0,palm_cy-pad); py1=min(fh,palm_cy+pad)
    ph_,pw_=py1-py0,px1-px0
    if ph_<=0 or pw_<=0: return
    lcx=palm_cx-px0; lcy=palm_cy-py0
    Y_p,X_p=np.mgrid[py0:py1,px0:px1].astype(np.float32)
    d_c=np.sqrt((X_p-palm_cx)**2+(Y_p-palm_cy)**2)
    pulse=0.85+0.15*math.sin(t*18.)
    glow=(np.exp(-0.5*(d_c/(R*0.30*pulse))**2)*200+
          np.exp(-0.5*(d_c/(R*0.60*pulse))**2)*120+
          np.exp(-0.5*(d_c/(R*1.00*pulse))**2)*60)
    patch=frame[py0:py1,px0:px1].astype(np.float32)
    patch=np.clip(patch+glow[:,:,None]*np.array([255,230,120],dtype=np.float32)/255.,0,255)
    core_r=max(4,int(R*0.18))
    core_mask=np.zeros((ph_,pw_),dtype=np.float32)
    cv2.circle(core_mask,(lcx,lcy),core_r,1.,-1,cv2.LINE_AA)
    patch=np.clip(patch+core_mask[:,:,None]*255,0,255)
    seed=int(t*12)%9999; np.random.seed(seed)
    bl=bolt_buf[py0:py1,px0:px1]; bl[:]=0
    for i in range(14):
        base_angle=2*math.pi*i/14+t*2.5; length=int(R*np.random.uniform(1.2,2.8)); depth=np.random.randint(3,6)
        col=(255,255,255) if i%3==0 else (200,230,255)
        _draw_bolt_tree(bl,lcx,lcy,base_angle,length,depth,pw_,ph_,col,0.5)
    for i in range(8):
        a=np.random.uniform(0,2*math.pi); ln=int(R*np.random.uniform(0.4,0.9))
        _draw_bolt_tree(bl,lcx,lcy,a,ln,3,pw_,ph_,(240,250,255),0.7)
    patch=np.clip(patch+bl.astype(np.float32)*1.1,0,255)
    dx=wrist_x-palm_cx; dy=wrist_y-palm_cy
    arm_angle=math.atan2(dy,dx)
    np.random.seed(seed+1)
    arm_buf=body_buf[py0:py1,px0:px1]; arm_buf[:]=0
    for i in range(6):
        frac=np.random.uniform(0.1,0.9)
        sx=int(lcx+dx*frac); sy=int(lcy+dy*frac)
        length=int(max(dist2d([palm_cx,palm_cy],[wrist_x,wrist_y]),1)*np.random.uniform(0.12,0.28))
        perp=arm_angle+math.pi/2+np.random.uniform(-0.5,0.5)
        _draw_bolt_tree(arm_buf,sx,sy,perp,length,3,pw_,ph_,(180,220,255),0.6)
    patch=np.clip(patch+arm_buf.astype(np.float32)*0.85,0,255)
    if body_points:
        np.random.seed(seed+2)
        bl2=aura_buf[py0:py1,px0:px1]; bl2[:]=0
        for (bx,by) in body_points:
            lbx=bx-px0; lby=by-py0
            if not (0<=lbx<pw_ and 0<=lby<ph_): continue
            for _ in range(np.random.randint(2,5)):
                a=np.random.uniform(0,2*math.pi); ln=np.random.randint(20,70)
                _draw_bolt_tree(bl2,lbx,lby,a,ln,3,pw_,ph_,(140,200,255),0.7)
        patch=np.clip(patch+bl2.astype(np.float32)*0.7,0,255)
    frame[py0:py1,px0:px1]=np.clip(patch,0,255).astype(np.uint8)

# ══════════════════════════════════════════════════════════════════════
# CHAKRA (mode 5 — deux index croisés)
# ══════════════════════════════════════════════════════════════════════
def draw_chakra_aura(frame, body_pts, t, intensity=1.):
    if len(body_pts)<2: return
    H,W=frame.shape[:2]; aura_layer=np.zeros((H,W,3),dtype=np.float32)
    pulse=0.6+0.4*math.sin(t*4.); pulse2=0.5+0.5*math.sin(t*6.5+1.2)
    pts=np.array(body_pts,dtype=np.int32)
    if len(pts)>=3:
        hull=cv2.convexHull(pts)
        for expand,color,alpha in [(55,(255,140,20),0.08*pulse*intensity),(40,(255,200,50),0.12*pulse*intensity),
            (28,(200,255,100),0.18*pulse2*intensity),(18,(150,255,180),0.25*pulse*intensity),
            (10,(100,255,220),0.35*pulse2*intensity),(4,(80,255,255),0.50*pulse*intensity)]:
            cx_h=int(np.mean(hull[:,0,0])); cy_h=int(np.mean(hull[:,0,1]))
            expanded=[]
            for pt in hull[:,0]:
                dx,dy=pt[0]-cx_h,pt[1]-cy_h; d=max(1,math.sqrt(dx*dx+dy*dy))
                expanded.append([int(pt[0]+dx/d*expand),int(pt[1]+dy/d*expand)])
            tmp=np.zeros((H,W,3),dtype=np.uint8)
            cv2.fillPoly(tmp,[np.array(expanded,dtype=np.int32)],color)
            aura_layer+=tmp.astype(np.float32)*alpha
    aura_layer=np.clip(aura_layer,0,255).astype(np.uint8)
    cv2.addWeighted(frame,1.,aura_layer,0.6,0,frame)
    pts2=np.array(body_pts,dtype=np.int32)
    cx_b=int(np.mean(pts2[:,0])); cy_b=int(np.mean(pts2[:,1]))
    rx=int(np.std(pts2[:,0]))+40; ry=int(np.std(pts2[:,1]))+40
    for i in range(int(18*intensity)):
        angle=2*math.pi*i/(18*intensity if intensity else 1)+t*2.5
        noise_r=30+25*math.sin(t*3+i)
        sx=int(cx_b+(rx+noise_r)*math.cos(angle)); sy=int(cy_b+(ry+noise_r)*math.sin(angle)*0.85)
        bright=int(180+75*math.sin(t*7+i*1.3))
        cv2.circle(frame,(sx,sy),max(1,int(2+2*math.sin(t*5+i))),(bright,255,255),-1,cv2.LINE_AA)

def draw_chakra_lines(frame, body_pts, t):
    if len(body_pts)<2: return
    pts=np.array(body_pts,dtype=np.int32); cx_b=int(np.mean(pts[:,0]))
    for i in range(8):
        offset_x=int((i-4)*18); speed=1.5+0.4*i; phase=(t*speed+i*0.7)%1.
        x=cx_b+offset_x+int(15*math.sin(t*3+i))
        y_top=int(np.min(pts[:,1]))-30; y_bottom=int(np.max(pts[:,1]))+20
        y=int(y_bottom-phase*(y_bottom-y_top))
        bright=int(100+155*math.sin(math.pi*phase))
        cv2.circle(frame,(x,y),3,(bright,255,200),-1,cv2.LINE_AA)
        cv2.circle(frame,(x,y),6,(bright//2,180,100),1,cv2.LINE_AA)

def draw_orbiting_orbs(frame, body_pts, t, intensity=1.):
    if len(body_pts)<2: return
    pts=np.array(body_pts,dtype=np.int32)
    cx_b=int(np.mean(pts[:,0])); cy_b=int(np.mean(pts[:,1]))
    rx=int(np.std(pts[:,0])*1.8)+60; ry=int(np.std(pts[:,1])*1.4)+50
    for i in range(6):
        angle=2*math.pi*i/6+t*1.5
        ox=int(cx_b+rx*math.cos(angle)); oy=int(cy_b+ry*math.sin(angle)*0.7)
        r_orb=max(6,int(18*intensity))
        cv2.circle(frame,(ox,oy),r_orb+6,(180,140,30),-1,cv2.LINE_AA)
        cv2.circle(frame,(ox,oy),r_orb+3,(100,80,10),-1,cv2.LINE_AA)
        cv2.circle(frame,(ox,oy),r_orb,(30,20,5),-1,cv2.LINE_AA)
        cv2.circle(frame,(ox,oy),max(2,r_orb-4),(10,8,2),-1,cv2.LINE_AA)

def draw_rasengan_chakra(frame, hand_pos, t, size=55):
    cx,cy=int(hand_pos[0]),int(hand_pos[1])
    H,W=frame.shape[:2]; layer=np.zeros((H,W,3),dtype=np.uint8)
    cv2.circle(layer,(cx,cy),size,(200,120,0),-1,cv2.LINE_AA)
    cv2.circle(layer,(cx,cy),int(size*0.85),(230,170,10),-1,cv2.LINE_AA)
    cv2.circle(layer,(cx,cy),int(size*0.68),(255,210,30),-1,cv2.LINE_AA)
    cv2.circle(layer,(cx,cy),int(size*0.48),(255,240,100),-1,cv2.LINE_AA)
    cv2.circle(layer,(cx,cy),int(size*0.28),(255,255,200),-1,cv2.LINE_AA)
    for s in range(3):
        base_offset=2*math.pi*s/3; prev=None
        for step in range(80):
            frac=step/79.; r_sp=size*0.10+size*0.78*frac
            angle=base_offset+t*6.+frac*4.5*math.pi
            px_=int(cx+r_sp*math.cos(angle)); py_=int(cy+r_sp*math.sin(angle))
            bright=int(180+75*frac)
            if prev: cv2.line(layer,prev,(px_,py_),(bright,255,255),2,cv2.LINE_AA)
            prev=(px_,py_)
    pulse=0.5+0.5*math.sin(t*8); halo_r=size+int(20*pulse)
    halo_layer=np.zeros_like(layer)
    cv2.circle(halo_layer,(cx,cy),halo_r,(255,220,50),int(8*pulse)+2,cv2.LINE_AA)
    cv2.addWeighted(layer,1.,halo_layer,0.5,0,layer)
    for i in range(14):
        angle=2*math.pi*i/14+t*3; dist_=size+int(5+12*abs(math.sin(t*4+i)))
        sx=int(cx+dist_*math.cos(angle)); sy=int(cy+dist_*math.sin(angle))
        cv2.circle(layer,(sx,sy),max(1,int(3*pulse)),(255,255,200),-1,cv2.LINE_AA)
    cv2.addWeighted(frame,1.,layer,0.85,0,frame)

# ══════════════════════════════════════════════════════════════════════
# RASENGAN (mode 6 — cercle 2 mains)
# ══════════════════════════════════════════════════════════════════════
class PlasmaTexture:
    def __init__(self, size=128, n_frames=30):
        self.size=size; self.frames=[]; self._precompute(n_frames)
    def _precompute(self, n_frames):
        s=self.size; Y,X=np.mgrid[0:s,0:s].astype(np.float32)
        cx,cy=s/2.,s/2.
        dist_map=np.sqrt((X-cx)**2+(Y-cy)**2); angle_map=np.arctan2(Y-cy,X-cx)
        self.mask=np.clip((cx-dist_map)/(cx*0.15),0,1).astype(np.float32)
        for i in range(n_frames):
            t=i*(2*math.pi/n_frames)
            v=np.sin(X/8.+t*3.)+np.sin(Y/6.+t*2.5)+np.sin((X+Y)/10.+t*2.)
            v+=np.sin(dist_map/7.-t*4.)+np.sin(dist_map/4.-t*6.)+np.sin(angle_map*4+t*3.)*0.8
            v=(v-v.min())/(v.max()-v.min()+1e-6)
            B=np.clip(0.3+0.7*v,0,1); G=np.clip(0.8*(v-0.2),0,1); R=np.clip(0.5*(v-0.55),0,1)
            self.frames.append((np.stack([B,G,R],axis=-1)*255).astype(np.uint8))
    def get(self, t):
        return self.frames[int(t*15)%len(self.frames)], self.mask

print("Précalcul du plasma pour le Rasengan...")
_plasma_gen = PlasmaTexture(size=128, n_frames=30)

_rasengan_layer = _bolt_layer_r = _spiral_layer_r = None

def _get_rasengan_bufs(fh, fw):
    global _rasengan_layer, _bolt_layer_r, _spiral_layer_r
    if _rasengan_layer is None or _rasengan_layer.shape[:2] != (fh,fw):
        _rasengan_layer = np.zeros((fh,fw,3),dtype=np.float32)
        _bolt_layer_r   = np.zeros((fh,fw,3),dtype=np.float32)
        _spiral_layer_r = np.zeros((fh,fw,3),dtype=np.float32)
    return _rasengan_layer, _bolt_layer_r, _spiral_layer_r

def draw_rasengan_full(frame, cx, cy, t, radius=60):
    fh,fw=frame.shape[:2]; R=int(np.clip(radius,40,110))
    layer,bolt_layer,spiral_layer=_get_rasengan_bufs(fh,fw)
    pad=R+90; px0=max(0,cx-pad); px1=min(fw,cx+pad)
    py0=max(0,cy-pad); py1=min(fh,cy+pad); ph,pw=py1-py0,px1-px0
    if ph<=0 or pw<=0: return
    Y_p,X_p=np.mgrid[py0:py1,px0:px1].astype(np.float32)
    d_from_center=np.sqrt((X_p-cx)**2+(Y_p-cy)**2)
    glow=(np.exp(-0.5*(d_from_center/(R*0.55))**2)*60+
          np.exp(-0.5*(d_from_center/(R*0.35))**2)*90+
          np.exp(-0.5*(d_from_center/(R*0.20))**2)*120)
    patch=frame[py0:py1,px0:px1].astype(np.float32)
    patch=np.clip(patch+glow[:,:,None]*np.array([255,200,60],dtype=np.float32)/255.,0,255)
    plasma_frame,plasma_mask=_plasma_gen.get(t)
    diam=R*2
    plasma_r=cv2.resize(plasma_frame,(diam,diam),interpolation=cv2.INTER_NEAREST)
    mask_r=cv2.resize(plasma_mask,(diam,diam),interpolation=cv2.INTER_NEAREST)
    sx0=cx-R-px0; sy0=cy-R-py0; sx1=sx0+diam; sy1=sy0+diam
    csx0=max(0,sx0); csx1=min(pw,sx1); csy0=max(0,sy0); csy1=min(ph,sy1)
    psx0=csx0-sx0; psx1=psx0+(csx1-csx0); psy0=csy0-sy0; psy1=psy0+(csy1-csy0)
    if csx1>csx0 and csy1>csy0:
        a=mask_r[psy0:psy1,psx0:psx1,None].astype(np.float32)*0.88
        sub=patch[csy0:csy1,csx0:csx1]; pla=plasma_r[psy0:psy1,psx0:psx1].astype(np.float32)
        patch[csy0:csy1,csx0:csx1]=sub*(1-a)+pla*a
    sl=spiral_layer[py0:py1,px0:px1]; sl[:]=0
    def draw_spirals_fast(sl,cx,cy,px0,py0,R,t_off,n_arms,col,thick):
        for arm in range(n_arms):
            base=t_off+arm*(2*math.pi/n_arms); fracs=np.linspace(0,1,40)
            r_pts=fracs*R*0.92+np.sin(fracs*12+t_off*3)*R*0.04
            angles=base+fracs*5.*math.pi
            xs=(cx-px0+r_pts*np.cos(angles)).astype(np.int32)
            ys=(cy-py0+r_pts*np.sin(angles)).astype(np.int32)
            alphas=0.2+0.8*fracs
            for k in range(len(xs)-1):
                a=float(alphas[k]); c=tuple(int(v*a) for v in col)
                cv2.line(sl,(xs[k],ys[k]),(xs[k+1],ys[k+1]),c,thick,cv2.LINE_AA)
    draw_spirals_fast(sl,cx,cy,px0,py0,R,t*2.8,3,(255,240,180),2)
    draw_spirals_fast(sl,cx,cy,px0,py0,R,-t*2.2,2,(255,255,255),1)
    draw_spirals_fast(sl,cx,cy,px0,py0,int(R*0.65),t*3.5,3,(180,255,255),1)
    patch=np.clip(patch+sl.astype(np.float32)*0.8,0,255)
    angles_ring=t*4.5+np.arange(24)*(2*math.pi/24)
    r_orbs=R+8+np.sin(angles_ring*2+t*6)*5
    pxs=(cx-px0+r_orbs*np.cos(angles_ring)).astype(np.int32)
    pys=(cy-py0+r_orbs*np.sin(angles_ring)).astype(np.int32)
    brights=0.55+0.45*np.abs(np.sin(np.arange(24)*0.45+t*3))
    for i in range(len(pxs)):
        b=float(brights[i]); size=max(2,int(2+b*2)); px_,py_=int(pxs[i]),int(pys[i])
        if 0<=py_<ph and 0<=px_<pw:
            y0_=max(0,py_-size); y1_=min(ph,py_+size+1); x0_=max(0,px_-size); x1_=min(pw,px_+size+1)
            patch[y0_:y1_,x0_:x1_]=np.clip(patch[y0_:y1_,x0_:x1_]+b*50,0,255)
    bl=bolt_layer[py0:py1,px0:px1]; bl[:]=0
    np.random.seed(int(t*8)%9999)
    def bolt_fast(bl,x0b,y0b,cx_off,cy_off,ang,length,depth=0):
        if depth>2 or length<5: return
        x1b=int(x0b+length*math.cos(ang)); y1b=int(y0b+length*math.sin(ang))
        cv2.line(bl,(x0b-cx_off,y0b-cy_off),(x1b-cx_off,y1b-cy_off),(160,220,255),max(1,2-depth),cv2.LINE_AA)
        if np.random.rand()>0.35: bolt_fast(bl,x1b,y1b,cx_off,cy_off,ang+np.random.uniform(-0.7,0.7),length*np.random.uniform(0.4,0.65),depth+1)
        if np.random.rand()>0.6: bolt_fast(bl,x1b,y1b,cx_off,cy_off,ang+np.random.uniform(-1.,1.),length*np.random.uniform(0.3,0.5),depth+1)
    for _ in range(5):
        a0=np.random.uniform(0,2*math.pi); x0b=int(cx+R*math.cos(a0)); y0b=int(cy+R*math.sin(a0))
        bolt_fast(bl,x0b,y0b,px0,py0,a0,np.random.randint(15,35))
    patch=np.clip(patch+bl.astype(np.float32)*0.9,0,255)
    hx=cx-px0-int(R*0.30); hy=cy-py0-int(R*0.32)
    Y_s,X_s=np.mgrid[0:ph,0:pw].astype(np.float32)
    spec=np.exp(-((X_s-hx)**2/(R*0.28)**2+(Y_s-hy)**2/(R*0.16)**2))*180
    patch=np.clip(patch+spec[:,:,None],0,255)
    inside=np.clip(1.-d_from_center/R,0,1)
    edge_drk=np.clip((d_from_center-R*0.6)/(R*0.4),0,1)*inside
    patch=patch*(1.-edge_drk[:,:,None]*0.45)
    frame[py0:py1,px0:px1]=np.clip(patch,0,255).astype(np.uint8)

# ══════════════════════════════════════════════════════════════════════
# BOUCLE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════
MODE_NAMES = {
    'sharingan': '✌  SHARINGAN / RINNEGAN',
    'byakugan':  '🤟 BYAKUGAN (Neji)',
    'gaara':     '🖖 SABLE DE GAARA',
    'chidori':   '✊  CHIDORI (Kakashi)',
    'chakra':    '👆👆 MODE CHAKRA',
    'rasengan':  '👆👆 RASENGAN',
    None:        'Aucun geste détecté',
}
MODE_COLORS = {
    'sharingan': (0,80,220),
    'byakugan':  (200,180,230),
    'gaara':     (40,200,240),
    'chidori':   (255,220,50),
    'chakra':    (255,200,0),
    'rasengan':  (0,255,120),
    None:        (120,120,120),
}

# États persistants pour Gaara et Chakra (particules + force)
gaara_sand_particles   = []
gaara_spiral_particles = []
gaara_wave_particles   = []
chakra_effect_strength = 0.0
gaara_effect_strength  = 0.0

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Impossible d'ouvrir la caméra. Essai sans CAP_DSHOW...")
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Impossible d'ouvrir la caméra !")
    raise SystemExit

prev_time = time.time()
print("\n" + "="*60)
print("  NARUTO ALL EFFECTS — fichier unifié")
print("="*60)
print("  ✌  Doigts croisés   → Sharingan / Rinnegan")
print("  🤟 ILY               → Byakugan (Neji)")
print("  🖖 Vulcan            → Sable de Gaara")
print("  ✊  Poing fermé       → Chidori (Kakashi)")
print("  👆👆 Index croisés    → Mode Chakra")
print("  ○ Cercle (2 mains)  → Rasengan")
print("  Q / Échap            → Quitter")
print("="*60 + "\n")

while True:
    ok, frame = cap.read()
    if not ok: break

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W  = frame.shape[:2]
    now   = time.time()
    dt    = min(now - prev_time, 0.05)
    prev_time = now
    t = now

    # ── Détections ────────────────────────────────────────────────
    face_results = face_mesh.process(rgb)
    hand_results = hands_detector.process(rgb)
    pose_results = pose_detector.process(rgb)

    mode, extra = detect_current_mode(hand_results, W, H)

    body_pts = get_body_points(
        pose_results.pose_landmarks if pose_results else None, W, H
    )

    # ── Dessin landmarks mains ────────────────────────────────────
    if hand_results.multi_hand_landmarks:
        for hl in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(180,180,255),thickness=1,circle_radius=2),
                mp_draw.DrawingSpec(color=(120,120,200),thickness=1))

    # ── SHARINGAN ─────────────────────────────────────────────────
    if mode == 'sharingan':
        if face_results.multi_face_landmarks:
            render_sharingan(frame, face_results.multi_face_landmarks[0], W, H, t)

    # ── BYAKUGAN ──────────────────────────────────────────────────
    elif mode == 'byakugan':
        if face_results.multi_face_landmarks:
            render_byakugan(frame, face_results.multi_face_landmarks[0], W, H)

    # ── GAARA ─────────────────────────────────────────────────────
    elif mode == 'gaara':
        gaara_effect_strength = min(1., gaara_effect_strength + 0.04)
        if body_pts:
            body_cx=int(np.mean([p[0] for p in body_pts])); body_cy=int(np.mean([p[1] for p in body_pts]))
        else:
            body_cx,body_cy=W//2,H//2
        s=gaara_effect_strength
        # spawn
        for _ in range(int(s*12)):
            if len(gaara_sand_particles)<220: gaara_sand_particles.append(SandParticle(W,H))
        for _ in range(int(s*6)):
            if len(gaara_spiral_particles)<90: gaara_spiral_particles.append(SpiralParticle(body_cx,body_cy,W,H,len(gaara_spiral_particles)))
        for _ in range(int(s*8)):
            if len(gaara_wave_particles)<160: gaara_wave_particles.append(SandWaveParticle(W,H))
        draw_ground_wave(frame,t,s,W,H)
        alive=[]; [p.update(dt,body_cx,body_cy,s) or (alive.append(p) if not p.is_dead() else None) for p in gaara_sand_particles]
        gaara_sand_particles=[p for p in alive]
        for p in gaara_sand_particles: p.draw(frame)
        draw_sand_trails(frame,gaara_spiral_particles,s)
        alive_sp=[]; [(p.update(dt,body_cx,body_cy,s), alive_sp.append(p) if not p.is_dead() else None) for p in gaara_spiral_particles]
        gaara_spiral_particles=alive_sp
        for p in gaara_spiral_particles: p.draw(frame)
        alive_w=[]; [(p.update(dt,s), alive_w.append(p) if not p.is_dead() else None) for p in gaara_wave_particles]
        gaara_wave_particles=alive_w
        for p in gaara_wave_particles: p.draw(frame)
        if face_results.multi_face_landmarks:
            draw_gaara_eyes(frame,face_results.multi_face_landmarks[0],W,H,s)
    else:
        gaara_effect_strength = max(0., gaara_effect_strength - 0.025)

    # ── CHIDORI ───────────────────────────────────────────────────
    if mode == 'chidori' and hand_results.multi_hand_landmarks:
        body_ids=[11,12,13,14,23,24]
        body_pts_chidori=[]
        if pose_results.pose_landmarks:
            for bid in body_ids:
                lm=pose_results.pose_landmarks.landmark[bid]
                if lm.visibility>0.5: body_pts_chidori.append((int(lm.x*W),int(lm.y*H)))
        for hand_lm in hand_results.multi_hand_landmarks:
            if is_fist(hand_lm):
                pcx=int(np.mean([hand_lm.landmark[i].x*W for i in [0,5,9,13,17]]))
                pcy=int(np.mean([hand_lm.landmark[i].y*H for i in [0,5,9,13,17]]))
                wrx=int(hand_lm.landmark[0].x*W); wry=int(hand_lm.landmark[0].y*H)
                hand_size=dist2d(lm_ptf(hand_lm,0,W,H), lm_ptf(hand_lm,9,W,H))
                draw_chidori(frame,pcx,pcy,wrx,wry,body_pts_chidori,t,hand_size)

    # ── CHAKRA ────────────────────────────────────────────────────
    elif mode == 'chakra':
        chakra_effect_strength = min(1., chakra_effect_strength + 0.06)
        s=chakra_effect_strength
        if len(body_pts)>=3:
            draw_chakra_lines(frame,body_pts,t)
            draw_chakra_aura(frame,body_pts,t,intensity=s)
            draw_orbiting_orbs(frame,body_pts,t,intensity=s)
            mid=extra.get('mid')
            if mid is not None: draw_rasengan_chakra(frame,mid,t,size=int(35+25*s))
    else:
        if mode not in ('gaara',):
            chakra_effect_strength = max(0., chakra_effect_strength - 0.04)

    # ── RASENGAN ──────────────────────────────────────────────────
    if mode == 'rasengan':
        cx_r,cy_r,r_r=extra.get('cx',W//2),extra.get('cy',H//2),extra.get('r',60)
        draw_rasengan_full(frame,cx_r,cy_r,t,radius=r_r)

    # ── UI ────────────────────────────────────────────────────────
    col = MODE_COLORS.get(mode, (120,120,120))
    label = MODE_NAMES.get(mode, '?')

    # Bandeau semi-transparent en haut
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (W,50), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, label, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)

    # Aide gestes en bas si aucun mode actif
    if mode is None:
        help_lines = [
            "Gestes : ✌ Sharingan  🤟 Byakugan  🖖 Gaara  ✊ Chidori  👆👆 Chakra/Rasengan",
        ]
        cv2.putText(frame, help_lines[0], (10, H-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160,160,200), 1)

    fps = 1. / max(now - (now - dt), 1e-6)

    cv2.imshow("Naruto All Effects", frame)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27): break

cap.release()
cv2.destroyAllWindows()

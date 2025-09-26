import time
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# ----------------- Config -----------------
MIN_GREEN = 5.0
MAX_GREEN = 20.0
QUEUE_THRESHOLD = 3
YELLOW_TIME = 2.0
MAIN_FREE_FLOW = 40   # km/h
CONGESTED_SPEED = 25  # km/h

st.set_page_config(page_title="SIH Smart Intersection (Basic Prototype)", layout="wide")
st.title("SIH Smart Intersection — Basic Prototype")

# ----------------- Model ------------------
@st.cache_resource
def load_model():
	model = YOLO("yolov8n.pt")  # small & fast
	return model

model = load_model()

# ----------------- Video ------------------
cap = cv2.VideoCapture("data/clip.mp4")
if not cap.isOpened():
	st.error("Could not open data/clip.mp4. Please add a short intersection video there.")
	st.stop()

# ----------------- State ------------------
if "phase" not in st.session_state:
	st.session_state.phase = "MAIN_GREEN"  # MAIN_GREEN, MAIN_YELLOW, CROSS_GREEN, CROSS_YELLOW
	st.session_state.phase_start = time.time()

def time_in_phase():
	return time.time() - st.session_state.phase_start

def set_phase(p):
	st.session_state.phase = p
	st.session_state.phase_start = time.time()

# ----------------- UI Layout --------------
col1, col2 = st.columns([2, 1])
video_placeholder = col1.empty()
with col2:
	phase_box = st.empty()
	counts_box = st.empty()
	risk_box = st.empty()
	speed_box = st.empty()

# ----------------- Main Loop --------------
VEH_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck (COCO IDs)

while True:
	ret, frame = cap.read()
	if not ret:
		# loop the video
		cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
		continue

	h, w, _ = frame.shape
	mid = w // 2
	stop_x = mid - 20
	stop_y = h // 2

	# Detection
	results = model.predict(frame, imgsz=640, conf=0.25, verbose=False)
	dets = []
	if len(results):
		for r in results:
			if r.boxes is None:
				continue
			for b in r.boxes:
				cls_id = int(b.cls[0].item())
				conf = float(b.conf[0].item())
				x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
				dets.append((cls_id, conf, x1, y1, x2, y2))

	# Counting by halves (simple)
	main_count, cross_count = 0, 0
	fast_near_stop = False
	for cls_id, conf, x1, y1, x2, y2 in dets:
		color = (0, 255, 0) if cls_id in VEH_CLASSES else (255, 0, 0)
		cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
		label = f"{cls_id}:{conf:.2f}"
		cv2.putText(frame, label, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

		cx = (x1 + x2) // 2
		cy = (y1 + y2) // 2
		if cx < mid:
			main_count += 1
		else:
			cross_count += 1

		# very rough red-run risk proxy near stop line
		if abs(cx - stop_x) < 50 or abs(cy - stop_y) < 50:
			if (x2 - x1) > 50 or (y2 - y1) > 50:
				fast_near_stop = True

	# Phase logic
	phase = st.session_state.phase
	tp = time_in_phase()
	next_phase = phase

	if phase == "MAIN_GREEN":
		if (cross_count - main_count) > QUEUE_THRESHOLD and tp > MIN_GREEN:
			next_phase = "MAIN_YELLOW"
		elif tp > MAX_GREEN:
			next_phase = "MAIN_YELLOW"

	elif phase == "MAIN_YELLOW":
		if tp > YELLOW_TIME:
			next_phase = "CROSS_GREEN"

	elif phase == "CROSS_GREEN":
		if (main_count - cross_count) > QUEUE_THRESHOLD and tp > MIN_GREEN:
			next_phase = "CROSS_YELLOW"
		elif tp > MAX_GREEN:
			next_phase = "CROSS_YELLOW"

	elif phase == "CROSS_YELLOW":
		if tp > YELLOW_TIME:
			next_phase = "MAIN_GREEN"

	# Risk during yellow
	redrun_risk = "Low"
	if "YELLOW" in phase:
		redrun_risk = "High" if fast_near_stop else "Medium"
	# simple mitigation: hold MAIN_YELLOW briefly if risk high
	if phase == "MAIN_YELLOW" and redrun_risk == "High" and tp < (YELLOW_TIME + 1.0):
		next_phase = "MAIN_YELLOW"

	if next_phase != phase:
		set_phase(next_phase)

	# Speed advisory
	queue_proxy = main_count if "MAIN" in st.session_state.phase else cross_count
	advisory = CONGESTED_SPEED if queue_proxy >= 6 else MAIN_FREE_FLOW

	# Overlay
	if "MAIN" in st.session_state.phase:
		cv2.putText(frame,
		            "MAIN: GREEN" if "GREEN" in st.session_state.phase else "MAIN: YELLOW",
		            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
		            (0, 255, 0) if "GREEN" in st.session_state.phase else (0, 255, 255), 2)
	else:
		cv2.putText(frame,
		            "CROSS: GREEN" if "GREEN" in st.session_state.phase else "CROSS: YELLOW",
		            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
		            (0, 255, 0) if "GREEN" in st.session_state.phase else (0, 255, 255), 2)

	# Update UI
	video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
	with col2:
		phase_box.markdown(f"### Signal phase\n- {st.session_state.phase}\n- Time in phase: {time_in_phase():.1f}s")
		counts_box.markdown(f"### Queue counts\n- Main: {main_count}\n- Cross: {cross_count}")
		risk_box.markdown(f"### Red-light risk\n- {redrun_risk}")
		speed_box.markdown(f"### Speed advisory\n- {advisory} km/h")

	# keep loop responsive
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

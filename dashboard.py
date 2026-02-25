import cv2
import torch
import numpy as np
import heapq
import time

# ===============================
# Load MiDaS (Pseudo LiDAR)
# ===============================
print("Loading MiDaS model...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
midas.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
transform = midas_transforms.small_transform

print("Model loaded on:", device)

# ===============================
# A* Algorithm
# ===============================
def astar(grid, start, goal):
    rows, cols = grid.shape
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            neighbor = (current[0]+dx, current[1]+dy)

            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor] == 1:
                    continue

                temp_g = g_score[current] + 1
                if neighbor not in g_score or temp_g < g_score[neighbor]:
                    g_score[neighbor] = temp_g
                    f = temp_g + heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f, neighbor))
                    came_from[neighbor] = current
    return []

# ===============================
# Webcam
# ===============================
cap = cv2.VideoCapture(0)

frame_count = 0
path = []
pseudo_lidar = np.zeros((240, 320, 3), dtype=np.uint8)

print("Starting camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Reduce resolution for speed
    frame = cv2.resize(frame, (320, 240))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    start_time = time.time()

    # --------------------------------
    # Run depth every 3 frames only
    # --------------------------------
    if frame_count % 3 == 0:

        input_batch = transform(rgb).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype(np.uint8)

        pseudo_lidar = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)

        # Occupancy grid (smaller for speed)
        obstacle_map = depth_norm > 150
        grid = obstacle_map.astype(int)

        small_grid = cv2.resize(grid.astype(np.uint8), (40, 30))
        small_grid = small_grid.astype(int)

        start_node = (29, 20)
        goal_node = (5, 20)

        path = astar(small_grid, start_node, goal_node)

    frame_count += 1

    # --------------------------------
    # Draw Optimal Path
    # --------------------------------
    optimal_view = frame.copy()

    scale_x = 320 // 40
    scale_y = 240 // 30

    for point in path:
        y, x = point
        cv2.circle(optimal_view,
                   (int(x * scale_x), int(y * scale_y)),
                   2, (0, 255, 0), -1)

    # Bird's Eye View (Improved)
    # -----------------------------
    h, w = frame.shape[:2]

    # Focus only on lower half (floor region)
    src = np.float32([
        [w*0.35, h*0.65],   # top-left of road
        [w*0.65, h*0.65],   # top-right of road
        [w*0.1,  h*0.95],   # bottom-left
        [w*0.9,  h*0.95]    # bottom-right
    ])

    dst = np.float32([
        [w*0.2, 0],
        [w*0.8, 0],
        [w*0.2, h],
        [w*0.8, h]
    ])

    matrix = cv2.getPerspectiveTransform(src, dst)
    bird_eye = cv2.warpPerspective(frame, matrix, (w, h))

    # --------------------------------
    # Add Labels
    # --------------------------------
    cv2.putText(optimal_view, "Optimal Path View", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.putText(pseudo_lidar, "Pseudo LiDAR", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.putText(bird_eye, "Bird's Eye View", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # --------------------------------
    # Dashboard Layout (dimension safe)
    # --------------------------------
    pseudo_resized = cv2.resize(pseudo_lidar, (320,240))
    bird_resized = cv2.resize(bird_eye, (320,240))

    bottom_row = np.hstack((pseudo_resized, bird_resized))
    optimal_resized = cv2.resize(optimal_view, (640,240))

    dashboard = np.vstack((optimal_resized, bottom_row))

    # --------------------------------
    # FPS Display
    # --------------------------------
    elapsed = time.time() - start_time
    if elapsed > 0:
        fps = 1 / elapsed
    else:
        fps = 0
    cv2.putText(dashboard, f"FPS: {int(fps)}", (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("Navigation Dashboard", dashboard)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
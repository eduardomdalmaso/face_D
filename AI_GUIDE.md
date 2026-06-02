# AI Orientation Guide: Facial Detection Benchmark & Web API Project

Hello, AI agent! This guide is written to help you orient yourself in this repository and continue developing features without breaking the existing architecture.

---

## 1. Project Context & Environment
*   **Purpose:** A hybrid face detection and recognition system. It has a command-line benchmark runner comparing local detectors (RetinaFace, YuNet, MediaPipe, Haar Cascades), and an interactive FastAPI web application that supports real-time facial recognition and enrollment stored in an SQLite database.
*   **Conda Environment:** The environment is named `Face` (Python 3.11).
*   **Execution Host:** Running on a Linux machine with an NVIDIA GPU. Keep GPU execution in mind.

---

## 2. Key Architecture Rules

### A. Keep Process Isolation in `main.py`
*   Do NOT run face detectors sequentially in the main thread of `main.py`.
*   Always use `multiprocessing.Process` with `spawn` start method to run face detectors in CLI.
*   Note: The FastAPI server runs YuNet inside the main thread because YuNet is native to OpenCV and highly thread-safe (non-tf-based), which runs in 16ms without segfaulting. DeepFace embedding extraction is also run on cropped face matrices with `detector_backend='skip'` for fast and stable operation.

### B. Database & Web API Architecture
*   **Database (`src/database.py`):** Managed via native `sqlite3`. Uses Pydantic for validation, serializes JSON arrays to save embeddings. Compares signatures using cosine similarity math. Stores a `quality_score` column for each user.
*   **FastAPI (`src/server.py`):** Serves `templates/index.html` as frontend. Connects to `/register` (HTTP POST form-data with image file upload to compute and save embedding) and `/recognize` (HTTP POST with base64 webcam frame payloads).
*   **Dynamic Threshold:** The `/recognize` request body accepts a float `threshold` representing the required similarity. Ensure this value is passed down to the comparison algorithm.
*   **Self-Improving Enrollment (Quality Upgrade):** When a user is successfully recognized with a similarity score higher than their stored `quality_score` (and >= 0.65), crop their face and overwrite their saved image and embedding. This automatically upgrades profile pictures with higher quality captures.

### C. Implement the `BaseDetector` Interface
If you need to add a new detector (e.g. YOLOv8, Dlib, SSD):
1.  Create a file under `src/detectors/new_detector.py`.
2.  Inherit from `BaseDetector` in [base.py](file:///home/hades/Documents/visao/face_D/src/detectors/base.py).
3.  Implement the `detect(self, image_path: str) -> list[DetectionResult]` method.
4.  Ensure it returns a list of Pydantic `DetectionResult` objects.
5.  Add it to `DETECTORS_CONFIG` in [main.py](file:///home/hades/Documents/visao/face_D/main.py).

---

## 3. GPU/CUDA Conventions
*   **OpenCV YuNet:** Check if OpenCV is compiled with CUDA before setting the backend:
    ```python
    backends = cv2.dnn.getAvailableBackends()
    if cv2.dnn.DNN_BACKEND_CUDA in backends:
        self.detector.setBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.detector.setTarget(cv2.dnn.DNN_TARGET_CUDA)
    ```
*   **MediaPipe:** The GPU delegate can segfault in headless runs or background terminal CLI runners. Use `os.getenv("USE_GPU", "true")` to determine whether to enable EGL GPU context.
*   **TensorFlow:** Will automatically try to use CUDA if driver paths are found in the environment.

---

## 4. Testing & Running
*   To test changes, activate the conda environment and run `main.py`:
    ```bash
    conda activate Face
    python main.py
    ```
*   Webcam capture in [capture.py](file:///home/hades/Documents/visao/face_D/src/capture.py) is headless. It opens the camera, grabs one frame, and writes it directly to disk. Do NOT try to use `cv2.imshow` or open windows, as it will crash in CLI execution contexts.

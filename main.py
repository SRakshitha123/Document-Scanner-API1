from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import cv2
import numpy as np
import uuid

# Create FastAPI application
app = FastAPI(title="Document Scanner API")

# Folder to save scanned images
OUTPUT_DIR = Path(__file__).parent / "scanned_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Maximum upload size: 10 MB
MAX_FILE_SIZE = 10 * 1024 * 1024

ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp"
}


def order_points(points):
    """Arrange document corners in the correct order."""
    points = np.array(points, dtype="float32")
    ordered = np.zeros((4, 2), dtype="float32")

    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).flatten()

    ordered[0] = points[np.argmin(sums)]        # Top-left
    ordered[1] = points[np.argmin(differences)]  # Top-right
    ordered[2] = points[np.argmax(sums)]        # Bottom-right
    ordered[3] = points[np.argmax(differences)] # Bottom-left

    return ordered


def scan_document(image):
    """Detect, straighten and enhance a document."""

    height, width = image.shape[:2]

    # Resize image for faster boundary detection
    scale = min(1.0, 1000 / max(height, width))

    if scale < 1.0:
        small = cv2.resize(
            image, None, fx=scale, fy=scale
        )
    else:
        small = image.copy()

    # Convert to grayscale and detect edges
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    # Connect nearby broken edges
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (5, 5)
    )
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, kernel
    )

    # Find contours
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    document = None
    image_area = small.shape[0] * small.shape[1]

    # Find a large contour with four corners
    for contour in contours[:50]:
        if cv2.contourArea(contour) < image_area * 0.05:
            continue

        perimeter = cv2.arcLength(contour, True)

        for factor in (0.02, 0.03, 0.04, 0.05):
            approx = cv2.approxPolyDP(
                contour,
                factor * perimeter,
                True
            )

            if (
                len(approx) == 4
                and cv2.isContourConvex(approx)
            ):
                document = approx.reshape(4, 2).astype(
                    "float32"
                ) / scale
                break

        if document is not None:
            break

    if document is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "Document boundary not detected. "
                "Use a clear photo showing all four corners "
                "against a contrasting background."
            )
        )

    # Order the corners
    corners = order_points(document)

    top_left, top_right, bottom_right, bottom_left = corners

    # Calculate output width and height
    width_a = np.linalg.norm(bottom_right - bottom_left)
    width_b = np.linalg.norm(top_right - top_left)
    output_width = max(1, int(max(width_a, width_b)))

    height_a = np.linalg.norm(top_right - bottom_right)
    height_b = np.linalg.norm(top_left - bottom_left)
    output_height = max(1, int(max(height_a, height_b)))

    destination = np.array([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1]
    ], dtype="float32")

    # Perspective transformation straightens the document
    matrix = cv2.getPerspectiveTransform(
        corners, destination
    )

    warped = cv2.warpPerspective(
        image,
        matrix,
        (output_width, output_height)
    )

    # Convert to grayscale and improve contrast
    gray_scanned = cv2.cvtColor(
        warped, cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray_scanned)

    return enhanced


@app.get("/")
def home():
    return {
        "message": "Document Scanner API is running",
        "endpoint": "POST /scan-document"
    }


@app.post("/scan-document")
async def upload_document(file: UploadFile = File(...)):

    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPG, PNG, or WEBP image."
        )

    # Read only up to the permitted size plus one byte
    content = await file.read(MAX_FILE_SIZE + 1)

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File size must not exceed 10 MB."
        )

    # Decode image using OpenCV
    image_array = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid or unsupported image file."
        )

    # Detect and scan the document
    scanned = scan_document(image)

    # Generate a unique output filename
    filename = f"{uuid.uuid4().hex}.jpg"
    output_path = OUTPUT_DIR / filename

    # Save the processed image
    if not cv2.imwrite(str(output_path), scanned):
        raise HTTPException(
            status_code=500,
            detail="Failed to save scanned image."
        )

    return {
        "message": "Document scanned successfully",
        "original_filename": file.filename,
        "original_width": int(image.shape[1]),
        "original_height": int(image.shape[0]),
        "scanned_width": int(scanned.shape[1]),
        "scanned_height": int(scanned.shape[0]),
        "output_file": filename,
        "download_url": f"/scanned/{filename}"
    }


@app.get("/scanned/{filename}")
def download_scanned_image(filename: str):

    # Validate generated filename
    if (
        len(filename) != 36
        or not filename.endswith(".jpg")
        or any(
            char not in "0123456789abcdef-"
            for char in filename[:-4]
        )
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid filename."
        )

    output_path = OUTPUT_DIR / filename

    if not output_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Scanned image not found."
        )

    return FileResponse(
        path=output_path,
        media_type="image/jpeg",
        filename="scanned_document.jpg"
    )

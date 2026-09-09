"""
Storage Abstraction Service
Provides a unified interface for saving citizen-uploaded photos and AI-annotated images.
Supports:
  1. Local Storage (Default for local development)
  2. Cloudinary Cloud Storage (For Render/Production deployment)
  3. Pluggable Cloud Storage Providers (Supabase/S3 compatible)
"""

import os
import io
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from backend.config import UPLOADS_DIR, RESULTS_DIR

logger = logging.getLogger("roadhealth.storage")


class StorageProvider(ABC):
    """Abstract base class for image storage backends."""

    @abstractmethod
    def save_bytes(self, image_bytes: bytes, filename: str, folder: str = "uploads") -> str:
        """Save raw image bytes and return the public URL or static path."""
        pass

    @abstractmethod
    def save_cv2_image(self, image_bgr: np.ndarray, filename: str, folder: str = "results", quality: int = 92) -> str:
        """Save OpenCV BGR image and return the public URL or static path."""
        pass


class LocalStorageProvider(StorageProvider):
    """Saves images to local filesystem (used for local development)."""

    def __init__(self, uploads_dir: Path = UPLOADS_DIR, results_dir: Path = RESULTS_DIR):
        self.uploads_dir = Path(uploads_dir)
        self.results_dir = Path(results_dir)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save_bytes(self, image_bytes: bytes, filename: str, folder: str = "uploads") -> str:
        target_dir = self.results_dir if folder == "results" else self.uploads_dir
        dest = target_dir / filename
        with open(dest, "wb") as f:
            f.write(image_bytes)
        logger.info(f"Saved local image ({len(image_bytes)} bytes) -> {dest}")
        return f"/{folder}/{filename}"

    def save_cv2_image(self, image_bgr: np.ndarray, filename: str, folder: str = "results", quality: int = 92) -> str:
        target_dir = self.results_dir if folder == "results" else self.uploads_dir
        dest = target_dir / filename
        cv2.imwrite(str(dest), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        logger.info(f"Saved local cv2 image ({image_bgr.shape}) -> {dest}")
        return f"/{folder}/{filename}"


def is_production_env() -> bool:
    """Check if running with explicit Cloudinary production backend."""
    return os.environ.get("STORAGE_BACKEND", "").lower() == "cloudinary"


class CloudinaryStorageProvider(StorageProvider):
    """
    Saves images directly to Cloudinary (used for Render deployment).
    Does NOT store image binaries in PostgreSQL. Returns secure HTTPS URLs.
    """

    def __init__(self):
        try:
            import cloudinary
            import cloudinary.uploader
            self.uploader = cloudinary.uploader
            
            # Configure from CLOUDINARY_URL or discrete keys
            cloudinary_url = os.environ.get("CLOUDINARY_URL")
            if cloudinary_url:
                cloudinary.config(cloudinary_url=cloudinary_url)
            else:
                cloudinary.config(
                    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
                    api_key=os.environ.get("CLOUDINARY_API_KEY"),
                    api_secret=os.environ.get("CLOUDINARY_API_SECRET"),
                    secure=True,
                )
            logger.info("✅ Cloudinary storage provider initialized successfully.")
        except ImportError:
            raise ImportError("cloudinary package is required for CloudinaryStorageProvider. Run: pip install cloudinary")

    def save_bytes(self, image_bytes: bytes, filename: str, folder: str = "uploads") -> str:
        try:
            public_id = Path(filename).stem
            res = self.uploader.upload(
                image_bytes,
                folder=f"roadcare/{folder}",
                public_id=public_id,
                resource_type="image",
            )
            secure_url = res.get("secure_url") or res.get("url")
            logger.info(f"Uploaded bytes to Cloudinary -> {secure_url}")
            return secure_url
        except Exception as e:
            if is_production_env():
                logger.error(f"Cloudinary upload failed: {e}")
                raise RuntimeError(
                    f"Cloudinary image upload failed: {e}. "
                    "Image submission halted to prevent data loss."
                )
            logger.error(f"Cloudinary upload failed: {e}. Falling back to local storage.")
            return LocalStorageProvider().save_bytes(image_bytes, filename, folder)

    def save_cv2_image(self, image_bgr: np.ndarray, filename: str, folder: str = "results", quality: int = 92) -> str:
        try:
            success, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not success:
                raise ValueError("Failed to encode cv2 image to JPEG.")
            return self.save_bytes(buffer.tobytes(), filename, folder)
        except Exception as e:
            if is_production_env():
                logger.error(f"Cloudinary cv2 upload failed: {e}")
                raise RuntimeError(
                    f"Cloudinary annotated image upload failed: {e}. "
                    "Image submission halted to prevent data loss."
                )
            logger.error(f"Cloudinary cv2 upload failed: {e}. Falling back to local storage.")
            return LocalStorageProvider().save_cv2_image(image_bgr, filename, folder, quality)


# ── Storage Singleton Factory ────────────────────────────────────────

_storage_provider: Optional[StorageProvider] = None


def get_storage_provider() -> StorageProvider:
    """
    Returns the configured StorageProvider.
    - If STORAGE_BACKEND=local: uses LocalStorageProvider (Render Phase 1 ephemeral local disk or local dev).
    - If STORAGE_BACKEND=cloudinary: uses CloudinaryStorageProvider (requires CLOUDINARY_URL).
    - If STORAGE_BACKEND=auto (or unset): uses Cloudinary if credentials exist, else LocalStorageProvider.
    """
    global _storage_provider
    if _storage_provider is not None:
        return _storage_provider

    backend = os.environ.get("STORAGE_BACKEND", "local").lower().strip()
    has_cloudinary_creds = bool(
        os.environ.get("CLOUDINARY_URL") or
        (os.environ.get("CLOUDINARY_CLOUD_NAME") and os.environ.get("CLOUDINARY_API_KEY"))
    )

    # 1. Explicit Local Storage (Phase 1 Render deployment or local development)
    if backend == "local":
        _storage_provider = LocalStorageProvider()
        logger.info("Active Storage: LOCAL FILESYSTEM (Local / Ephemeral Storage)")
        return _storage_provider

    # 2. Explicit Cloudinary Storage requested
    if backend == "cloudinary":
        if not has_cloudinary_creds:
            error_msg = (
                "STORAGE_BACKEND is set to 'cloudinary', but CLOUDINARY_URL is missing. "
                "Please configure the 'CLOUDINARY_URL' environment variable or set STORAGE_BACKEND=local."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        try:
            _storage_provider = CloudinaryStorageProvider()
            logger.info("Active Storage: CLOUDINARY (Cloud Storage)")
            return _storage_provider
        except Exception as e:
            logger.error(f"Failed to initialize Cloudinary storage: {e}")
            raise RuntimeError(f"Cloudinary initialization failed: {e}")

    # 3. Auto Mode: use Cloudinary if credentials are provided, otherwise local filesystem
    if has_cloudinary_creds:
        try:
            _storage_provider = CloudinaryStorageProvider()
            logger.info("Active Storage: CLOUDINARY (Auto-detected Cloud Storage)")
            return _storage_provider
        except Exception as e:
            logger.warning(f"Failed to initialize Cloudinary storage ({e}). Falling back to LocalStorageProvider.")

    _storage_provider = LocalStorageProvider()
    logger.info("Active Storage: LOCAL FILESYSTEM")
    return _storage_provider

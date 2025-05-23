import os
import glob
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import cv2
import requests
from dotenv import load_dotenv
import time
from dataclasses import dataclass
from enum import Enum


class TaskStatus(Enum):
    """Enum for FitRoom API task statuses."""
    CREATED = "CREATED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ClothingType(Enum):
    """Enum for clothing types supported by FitRoom API."""
    UPPER = "upper"
    LOWER = "lower"
    FULL = "full"
    COMBO = "combo"


@dataclass
class ValidationResult:
    """Data class for image validation results."""
    is_valid: bool
    error_message: Optional[str] = None
    good_clothes_types: Optional[list] = None
    clothes_type: Optional[str] = None


@dataclass
class TryOnResult:
    """Data class for try-on task results."""
    task_id: str
    status: TaskStatus
    progress: int = 0
    download_url: Optional[str] = None
    error_message: Optional[str] = None


class FitRoomAPIError(Exception):
    """Custom exception for FitRoom API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, error_code: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


class ClothesTransposer:
    """
    A professional virtual try-on service using FitRoom API.
    
    This class provides functionality to transpose clothing from one image onto a model
    in another image using the FitRoom virtual try-on API service.
    
    Reference: https://developer.fitroom.app/
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ClothesTransposer with API configuration.
        
        Args:
            config: Optional configuration dictionary. If None, loads from environment.
        """
        self._setup_logging()
        self._load_configuration(config)
        self._setup_directories()
        self._setup_session()

    def _setup_logging(self) -> None:
        """Configure logging for the application."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('fitroom_tryon.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _load_configuration(self, config: Optional[Dict[str, Any]]) -> None:
        """Load configuration from environment or provided config."""
        load_dotenv()
        
        if config:
            self.api_key = config.get('api_key')
            self.base_url = config.get('base_url', 'https://platform.fitroom.app')
        else:
            self.api_key = os.getenv('FITROOM_API_KEY')
            self.base_url = os.getenv('FITROOM_BASE_URL', 'https://platform.fitroom.app')
        
        if not self.api_key:
            raise ValueError(
                "FITROOM_API_KEY not found. Please set it in your environment variables "
                "or pass it in the config parameter."
            )
        
        # API configuration
        self.max_retries = 3
        self.timeout = 30
        self.poll_interval = 2  # seconds
        self.max_poll_time = 300  # 5 minutes max polling time

    def _setup_directories(self) -> None:
        """Setup required directories for input and output."""
        self.screenshot_dir = Path('screenshots')
        self.reference_dir = Path('references')
        self.output_dir = Path('output')
        
        # Create directories if they don't exist
        for directory in [self.screenshot_dir, self.reference_dir, self.output_dir]:
            directory.mkdir(exist_ok=True)
            self.logger.info(f"Directory setup: {directory}")

    def _setup_session(self) -> None:
        """Setup requests session with proper headers and configuration."""
        self.session = requests.Session()
        self.session.headers.update({
            'X-API-KEY': self.api_key,
            'User-Agent': 'ClothesTransposer/1.0'
        })
        
        # Setup retry strategy
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def get_latest_image(self, directory: Path) -> str:
        """
        Get the most recent image from a directory.
        
        Args:
            directory: Path to the directory to search
            
        Returns:
            Path to the most recent image file
            
        Raises:
            FileNotFoundError: If no images are found in the directory
        """
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        files = []
        
        for extension in image_extensions:
            files.extend(glob.glob(str(directory / extension)))
        
        if not files:
            raise FileNotFoundError(f"No images found in {directory}")
        
        latest_file = max(files, key=os.path.getctime)
        self.logger.info(f"Selected latest image: {latest_file}")
        return latest_file

    def validate_local_images(self, cloth_path: str, model_path: str) -> bool:
        """
        Validate input images locally before sending to API.
        
        Args:
            cloth_path: Path to the clothing image
            model_path: Path to the model image
            
        Returns:
            True if images are valid, False otherwise
        """
        try:
            # Check if files exist
            if not os.path.exists(cloth_path) or not os.path.exists(model_path):
                self.logger.error("One or both image files not found")
                return False

            # Read and validate images
            cloth_img = cv2.imread(cloth_path)
            model_img = cv2.imread(model_path)

            if cloth_img is None or model_img is None:
                self.logger.error("Failed to read one or both images")
                return False

            # Check minimum dimensions
            min_size = 256
            if (cloth_img.shape[0] < min_size or cloth_img.shape[1] < min_size or
                model_img.shape[0] < min_size or model_img.shape[1] < min_size):
                self.logger.error(f"Images must be at least {min_size}x{min_size} pixels")
                return False

            self.logger.info("Local image validation passed")
            return True

        except Exception as e:
            self.logger.error(f"Image validation error: {e}")
            return False

    def validate_model_image(self, model_path: str) -> ValidationResult:
        """
        Validate model image using FitRoom API.
        
        Args:
            model_path: Path to the model image
            
        Returns:
            ValidationResult object with validation details
        """
        url = f"{self.base_url}/api/tryon/input_check/v1/model"
        
        try:
            with open(model_path, 'rb') as f:
                files = {'input_image': f}
                response = self.session.post(url, files=files, timeout=self.timeout)
                response.raise_for_status()
                
            data = response.json()
            
            if data.get('is_good', False):
                self.logger.info("Model image validation successful")
                return ValidationResult(
                    is_valid=True,
                    good_clothes_types=data.get('good_clothes_types', [])
                )
            else:
                error_msg = f"Model validation failed: {data.get('error_code', 'Unknown error')}"
                self.logger.warning(error_msg)
                return ValidationResult(is_valid=False, error_message=error_msg)
                
        except requests.RequestException as e:
            error_msg = f"Model validation request failed: {e}"
            self.logger.error(error_msg)
            return ValidationResult(is_valid=False, error_message=error_msg)

    def validate_clothes_image(self, cloth_path: str) -> ValidationResult:
        """
        Validate clothing image using FitRoom API.
        
        Args:
            cloth_path: Path to the clothing image
            
        Returns:
            ValidationResult object with validation details
        """
        url = f"{self.base_url}/api/tryon/input_check/v1/clothes"
        
        try:
            with open(cloth_path, 'rb') as f:
                files = {'input_image': f}
                response = self.session.post(url, files=files, timeout=self.timeout)
                response.raise_for_status()
                
            data = response.json()
            
            if data.get('is_clothes', False):
                self.logger.info("Clothes image validation successful")
                return ValidationResult(
                    is_valid=True,
                    clothes_type=data.get('clothes_type')
                )
            else:
                error_msg = f"Clothes validation failed: Invalid clothing image"
                self.logger.warning(error_msg)
                return ValidationResult(is_valid=False, error_message=error_msg)
                
        except requests.RequestException as e:
            error_msg = f"Clothes validation request failed: {e}"
            self.logger.error(error_msg)
            return ValidationResult(is_valid=False, error_message=error_msg)

    def create_tryon_task(self, cloth_path: str, model_path: str, cloth_type: str) -> TryOnResult:
        """
        Create a virtual try-on task using FitRoom API.
        
        Args:
            cloth_path: Path to the clothing image
            model_path: Path to the model image
            cloth_type: Type of clothing ('upper', 'lower', 'full', 'combo')
            
        Returns:
            TryOnResult object with task details
        """
        url = f"{self.base_url}/api/tryon/v2/tasks"
        
        try:
            with open(cloth_path, 'rb') as cloth_file, open(model_path, 'rb') as model_file:
                files = {
                    'cloth_image': cloth_file,
                    'model_image': model_file
                }
                data = {'cloth_type': cloth_type}
                
                response = self.session.post(url, files=files, data=data, timeout=self.timeout)
                response.raise_for_status()
                
            response_data = response.json()
            
            task_id = response_data.get('task_id')
            status = TaskStatus(response_data.get('status', 'CREATED'))
            
            self.logger.info(f"Try-on task created successfully: {task_id}")
            
            return TryOnResult(
                task_id=task_id,
                status=status,
                progress=0
            )
            
        except requests.RequestException as e:
            error_msg = f"Failed to create try-on task: {e}"
            self.logger.error(error_msg)
            raise FitRoomAPIError(error_msg)

    def get_task_status(self, task_id: str) -> TryOnResult:
        """
        Get the status of a try-on task.
        
        Args:
            task_id: The task ID to check
            
        Returns:
            TryOnResult object with current task status
        """
        url = f"{self.base_url}/api/tryon/v2/tasks/{task_id}"
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            return TryOnResult(
                task_id=task_id,
                status=TaskStatus(data.get('status')),
                progress=data.get('progress', 0),
                download_url=data.get('download_signed_url'),
                error_message=data.get('error')
            )
            
        except requests.RequestException as e:
            error_msg = f"Failed to get task status: {e}"
            self.logger.error(error_msg)
            raise FitRoomAPIError(error_msg)

    def wait_for_completion(self, task_id: str) -> TryOnResult:
        """
        Poll task status until completion with exponential backoff.
        
        Args:
            task_id: The task ID to monitor
            
        Returns:
            TryOnResult object with final task status
        """
        start_time = time.time()
        poll_interval = self.poll_interval
        
        self.logger.info(f"Starting to poll task {task_id} for completion")
        
        while time.time() - start_time < self.max_poll_time:
            try:
                result = self.get_task_status(task_id)
                
                self.logger.info(f"Task {task_id} status: {result.status.value}, progress: {result.progress}%")
                
                if result.status == TaskStatus.COMPLETED:
                    self.logger.info(f"Task {task_id} completed successfully")
                    return result
                elif result.status == TaskStatus.FAILED:
                    error_msg = f"Task {task_id} failed: {result.error_message}"
                    self.logger.error(error_msg)
                    raise FitRoomAPIError(error_msg)
                
                # Wait before next poll with exponential backoff
                time.sleep(poll_interval)
                poll_interval = min(poll_interval * 1.2, 10)  # Cap at 10 seconds
                
            except FitRoomAPIError:
                raise
            except Exception as e:
                self.logger.warning(f"Error during polling: {e}, retrying...")
                time.sleep(poll_interval)
        
        raise FitRoomAPIError(f"Task {task_id} timed out after {self.max_poll_time} seconds")

    def download_result(self, download_url: str, output_path: str) -> bool:
        """
        Download the result image from the provided URL.
        
        Args:
            download_url: URL to download the image from
            output_path: Local path to save the image
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            response = requests.get(download_url, timeout=self.timeout, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.logger.info(f"Result image downloaded successfully: {output_path}")
            return True
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to download result image: {e}")
            return False

    def transpose_clothes(self, cloth_type: Optional[str] = None, validate_inputs: bool = False) -> Optional[str]:
        """
        Main function to perform virtual try-on using FitRoom API.
        
        Args:
            cloth_type: Clothing type ('upper', 'lower', 'full'). If None, defaults to 'upper'.
            validate_inputs: Whether to validate inputs using FitRoom API before processing.
                           Set to False (default) to minimize credit usage.
            
        Returns:
            Path to the generated image if successful, None otherwise
        """
        try:
            # Get latest images
            cloth_path = self.get_latest_image(self.screenshot_dir)
            model_path = self.get_latest_image(self.reference_dir)
            
            self.logger.info(f"Processing: cloth={cloth_path}, model={model_path}")
            
            # Local validation (free - no API calls)
            if not self.validate_local_images(cloth_path, model_path):
                return None
            
            # API validation (optional - uses credits)
            if validate_inputs:
                self.logger.info("Validating images using FitRoom API (uses credits)...")
                
                model_validation = self.validate_model_image(model_path)
                if not model_validation.is_valid:
                    self.logger.error(f"Model validation failed: {model_validation.error_message}")
                    return None
                
                clothes_validation = self.validate_clothes_image(cloth_path)
                if not clothes_validation.is_valid:
                    self.logger.error(f"Clothes validation failed: {clothes_validation.error_message}")
                    return None
                
                # Auto-detect cloth type if not provided
                if not cloth_type:
                    cloth_type = clothes_validation.clothes_type
                    self.logger.info(f"Auto-detected clothing type: {cloth_type}")
            else:
                self.logger.info("Skipping API validation to minimize credit usage")
            
            # Default to 'upper' if still not determined (most common use case)
            if not cloth_type:
                cloth_type = ClothingType.UPPER.value
                self.logger.info(f"Using default clothing type: {cloth_type}")
            
            # Create try-on task (main API call that uses credits)
            self.logger.info("Creating virtual try-on task...")
            start_time = time.time()
            
            task_result = self.create_tryon_task(cloth_path, model_path, cloth_type)
            
            # Wait for completion (polling - minimal credit usage)
            self.logger.info(f"Waiting for task {task_result.task_id} to complete...")
            final_result = self.wait_for_completion(task_result.task_id)
            
            # Download result (free)
            if final_result.download_url:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = self.output_dir / f"fitroom_tryon_{timestamp}.png"
                
                if self.download_result(final_result.download_url, str(output_path)):
                    total_time = time.time() - start_time
                    self.logger.info(f"Virtual try-on completed successfully in {total_time:.2f} seconds")
                    return str(output_path)
            
            return None
            
        except FitRoomAPIError as e:
            self.logger.error(f"FitRoom API error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error during try-on: {e}")
            return None


def main():
    """Main entry point for the application."""
    try:
        transposer = ClothesTransposer()
        output_path = transposer.transpose_clothes()
        
        if output_path:
            print(f"✅ Virtual try-on completed successfully!")
            print(f"📁 Result saved to: {output_path}")
        else:
            print("❌ Virtual try-on failed. Check logs for details.")
            
    except Exception as e:
        print(f"❌ Application error: {e}")


if __name__ == "__main__":
    main() 
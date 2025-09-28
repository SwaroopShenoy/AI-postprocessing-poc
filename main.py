import os
import cv2
import torch
import psutil
import GPUtil
import subprocess
import tempfile
import shutil
from pathlib import Path
from flask import Flask, request, jsonify
from threading import Thread
import time
import logging
from typing import Optional, Tuple, Dict
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class ResourceManager:
    """Automatically detects and manages system resources"""
    
    def __init__(self):
        self.update_resources()
    
    def update_resources(self):
        """Update current system resource information"""
        # CPU info
        self.cpu_cores = psutil.cpu_count(logical=False)
        self.cpu_threads = psutil.cpu_count(logical=True)
        
        # Memory info
        memory = psutil.virtual_memory()
        self.total_ram = memory.total // (1024**3)  # GB
        self.available_ram = memory.available // (1024**3)  # GB
        
        # GPU info
        self.gpus = GPUtil.getGPUs()
        self.gpu_available = len(self.gpus) > 0
        if self.gpu_available:
            self.gpu_memory = self.gpus[0].memoryTotal  # MB
            self.gpu_name = self.gpus[0].name
        else:
            self.gpu_memory = 0
            self.gpu_name = "None"
        
        logger.info(f"Resources: CPU({self.cpu_cores}c/{self.cpu_threads}t), RAM({self.total_ram}GB), GPU({self.gpu_name}, {self.gpu_memory}MB)")
    
    def get_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available GPU memory"""
        if not self.gpu_available:
            return 1
        
        if self.gpu_memory >= 8000:  # 8GB+
            return 4
        elif self.gpu_memory >= 6000:  # 6GB+
            return 2
        else:
            return 1
    
    def get_optimal_tile_size(self) -> int:
        """Get optimal tile size for processing based on GPU memory"""
        if not self.gpu_available:
            return 256
        
        if self.gpu_memory >= 8000:
            return 512
        elif self.gpu_memory >= 6000:
            return 400
        else:
            return 256

class VideoProcessor:
    """Main video processing pipeline"""
    
    def __init__(self):
        self.resource_manager = ResourceManager()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize models (lazy loading)
        self.gfpgan_model = None
        self.esrgan_model = None
        self.rife_model = None
    
    def load_gfpgan(self):
        """Load GFPGAN model for face enhancement"""
        if self.gfpgan_model is None:
            try:
                from gfpgan import GFPGANer
                
                # Download model if not exists
                model_path = 'weights/GFPGANv1.4.pth'
                if not os.path.exists(model_path):
                    os.makedirs('weights', exist_ok=True)
                    logger.info("Downloading GFPGAN model...")
                    subprocess.run([
                        'wget', 
                        'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
                        '-P', 'weights/'
                    ])
                
                self.gfpgan_model = GFPGANer(
                    model_path=model_path,
                    upscale=1,  # We'll handle upscaling separately
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=None,
                    device=self.device
                )
                logger.info("GFPGAN model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load GFPGAN: {e}")
                raise
    
    def load_esrgan(self):
        """Load Real-ESRGAN model for upscaling"""
        if self.esrgan_model is None:
            try:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                
                # Model configuration for 2x upscaling (360p to 720p)
                model_path = 'weights/RealESRGAN_x2plus.pth'
                if not os.path.exists(model_path):
                    os.makedirs('weights', exist_ok=True)
                    logger.info("Downloading Real-ESRGAN model...")
                    subprocess.run([
                        'wget',
                        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                        '-P', 'weights/'
                    ])
                
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                
                self.esrgan_model = RealESRGANer(
                    scale=2,
                    model_path=model_path,
                    model=model,
                    tile=self.resource_manager.get_optimal_tile_size(),
                    tile_pad=10,
                    pre_pad=0,
                    half=True if self.device.type == 'cuda' else False,
                    device=self.device
                )
                logger.info("Real-ESRGAN model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Real-ESRGAN: {e}")
                raise
    
    def load_rife(self):
        """Load RIFE model for frame interpolation"""
        if self.rife_model is None:
            try:
                # This is a simplified RIFE loader - you may need to adjust based on your RIFE installation
                import sys
                sys.path.append('RIFE')  # Adjust path as needed
                
                from model.RIFE_HDv3 import Model
                
                self.rife_model = Model()
                self.rife_model.load_model('RIFE/train_log', -1)  # Adjust path as needed
                self.rife_model.eval()
                self.rife_model.device()
                
                logger.info("RIFE model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load RIFE: {e}")
                raise
    
    def extract_frames(self, video_path: str, temp_dir: str) -> Tuple[int, float, Dict]:
        """Extract frames from video"""
        start_time = time.time()
        logger.info("Extracting frames...")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        frames_dir = os.path.join(temp_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_path = os.path.join(frames_dir, f'frame_{frame_idx:06d}.png')
            cv2.imwrite(frame_path, frame)
            frame_idx += 1
            
            if frame_idx % 50 == 0:
                elapsed = time.time() - start_time
                extraction_fps = frame_idx / elapsed if elapsed > 0 else 0
                logger.info(f"Extracting: {frame_idx}/{frame_count} frames ({extraction_fps:.1f} fps)")
        
        cap.release()
        processing_time = time.time() - start_time
        extraction_fps = frame_idx / processing_time if processing_time > 0 else 0
        
        logger.info(f"Extracted {frame_idx} frames at {fps} FPS in {processing_time:.2f}s ({extraction_fps:.1f} extraction fps)")
        
        metrics = {
            'processing_time': processing_time,
            'frames_extracted': frame_idx,
            'extraction_fps': extraction_fps,
            'video_duration': duration,
            'stage': 'extraction'
        }
        
        return frame_idx, fps, metrics
    
    def process_frame_gfpgan(self, frame_path: str, output_path: str):
        """Process single frame with GFPGAN"""
        img = cv2.imread(frame_path)
        _, _, enhanced_img = self.gfpgan_model.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        cv2.imwrite(output_path, enhanced_img)
    
    def process_frame_esrgan(self, frame_path: str, output_path: str):
        """Process single frame with Real-ESRGAN"""
        img = cv2.imread(frame_path)
        enhanced_img, _ = self.esrgan_model.enhance(img, outscale=2)
        cv2.imwrite(output_path, enhanced_img)
    
    def apply_gfpgan(self, temp_dir: str, frame_count: int) -> Dict:
        """Apply GFPGAN to all frames"""
        start_time = time.time()
        logger.info("Applying GFPGAN face enhancement...")
        
        self.load_gfpgan()
        
        frames_dir = os.path.join(temp_dir, 'frames')
        gfpgan_dir = os.path.join(temp_dir, 'gfpgan')
        os.makedirs(gfpgan_dir, exist_ok=True)
        
        processed_frames = 0
        for i in range(frame_count):
            frame_path = os.path.join(frames_dir, f'frame_{i:06d}.png')
            output_path = os.path.join(gfpgan_dir, f'frame_{i:06d}.png')
            
            if os.path.exists(frame_path):
                self.process_frame_gfpgan(frame_path, output_path)
                processed_frames += 1
            
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                fps = processed_frames / elapsed if elapsed > 0 else 0
                logger.info(f"GFPGAN: Processed {i + 1}/{frame_count} frames ({fps:.2f} fps)")
        
        processing_time = time.time() - start_time
        fps = processed_frames / processing_time if processing_time > 0 else 0
        
        logger.info(f"GFPGAN completed: {processing_time:.2f}s, {fps:.2f} fps")
        
        return {
            'processing_time': processing_time,
            'frames_processed': processed_frames,
            'fps': fps,
            'stage': 'gfpgan'
        }
    
    def apply_esrgan(self, temp_dir: str, frame_count: int) -> Dict:
        """Apply Real-ESRGAN to upscale frames"""
        start_time = time.time()
        logger.info("Applying Real-ESRGAN upscaling...")
        
        self.load_esrgan()
        
        gfpgan_dir = os.path.join(temp_dir, 'gfpgan')
        esrgan_dir = os.path.join(temp_dir, 'esrgan')
        os.makedirs(esrgan_dir, exist_ok=True)
        
        processed_frames = 0
        for i in range(frame_count):
            frame_path = os.path.join(gfpgan_dir, f'frame_{i:06d}.png')
            output_path = os.path.join(esrgan_dir, f'frame_{i:06d}.png')
            
            if os.path.exists(frame_path):
                self.process_frame_esrgan(frame_path, output_path)
                processed_frames += 1
            
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                fps = processed_frames / elapsed if elapsed > 0 else 0
                logger.info(f"Real-ESRGAN: Processed {i + 1}/{frame_count} frames ({fps:.2f} fps)")
        
        processing_time = time.time() - start_time
        fps = processed_frames / processing_time if processing_time > 0 else 0
        
        logger.info(f"Real-ESRGAN completed: {processing_time:.2f}s, {fps:.2f} fps")
        
        return {
            'processing_time': processing_time,
            'frames_processed': processed_frames,
            'fps': fps,
            'stage': 'esrgan'
        }
    
    def apply_rife_interpolation(self, temp_dir: str, frame_count: int, target_fps: float) -> Tuple[int, Dict]:
        """Apply RIFE for frame interpolation"""
        start_time = time.time()
        logger.info("Applying RIFE frame interpolation...")
        
        self.load_rife()
        
        esrgan_dir = os.path.join(temp_dir, 'esrgan')
        rife_dir = os.path.join(temp_dir, 'rife')
        os.makedirs(rife_dir, exist_ok=True)
        
        # This is a simplified RIFE implementation
        # You may need to adjust based on your specific RIFE setup
        
        interpolated_count = 0
        processed_pairs = 0
        
        for i in range(frame_count - 1):
            frame1_path = os.path.join(esrgan_dir, f'frame_{i:06d}.png')
            frame2_path = os.path.join(esrgan_dir, f'frame_{i+1:06d}.png')
            
            if os.path.exists(frame1_path) and os.path.exists(frame2_path):
                # Copy original frame
                output1_path = os.path.join(rife_dir, f'frame_{interpolated_count:06d}.png')
                shutil.copy2(frame1_path, output1_path)
                interpolated_count += 1
                
                # Generate interpolated frame (simplified)
                # In a full implementation, you'd use the RIFE model here
                interpolated_path = os.path.join(rife_dir, f'frame_{interpolated_count:06d}.png')
                
                # Placeholder: simple averaging (replace with actual RIFE inference)
                img1 = cv2.imread(frame1_path)
                img2 = cv2.imread(frame2_path)
                interpolated = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
                cv2.imwrite(interpolated_path, interpolated)
                interpolated_count += 1
                processed_pairs += 1
                
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                pairs_per_sec = processed_pairs / elapsed if elapsed > 0 else 0
                logger.info(f"RIFE: Processed {i + 1}/{frame_count-1} pairs ({pairs_per_sec:.2f} pairs/sec)")
        
        # Copy last frame
        last_frame_path = os.path.join(esrgan_dir, f'frame_{frame_count-1:06d}.png')
        if os.path.exists(last_frame_path):
            output_path = os.path.join(rife_dir, f'frame_{interpolated_count:06d}.png')
            shutil.copy2(last_frame_path, output_path)
            interpolated_count += 1
        
        processing_time = time.time() - start_time
        pairs_per_sec = processed_pairs / processing_time if processing_time > 0 else 0
        
        logger.info(f"RIFE completed: Generated {interpolated_count} frames in {processing_time:.2f}s ({pairs_per_sec:.2f} pairs/sec)")
        
        metrics = {
            'processing_time': processing_time,
            'pairs_processed': processed_pairs,
            'frames_generated': interpolated_count,
            'pairs_per_second': pairs_per_sec,
            'stage': 'rife'
        }
        
        return interpolated_count, metrics
    
    def extract_audio(self, video_path: str, temp_dir: str) -> Optional[str]:
        """Extract audio from original video"""
        try:
            audio_path = os.path.join(temp_dir, 'audio.aac')
            
            # Extract audio using FFmpeg
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'copy',  # Copy audio codec
                audio_path
            ]
            
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(audio_path):
                logger.info("Audio extracted successfully")
                return audio_path
            else:
                logger.warning("No audio track found or extraction failed")
                return None
                
        except Exception as e:
            logger.warning(f"Audio extraction failed: {e}")
            return None

    def create_video(self, temp_dir: str, output_path: str, fps: float, frame_count: int, original_video_path: str) -> Dict:
        """Create final video from processed frames with preserved audio"""
        start_time = time.time()
        logger.info("Creating final video...")
        
        rife_dir = os.path.join(temp_dir, 'rife')
        
        # Get video properties from first processed frame
        first_frame_path = os.path.join(rife_dir, 'frame_000000.png')
        first_frame = cv2.imread(first_frame_path)
        height, width = first_frame.shape[:2]
        
        # Enhanced FPS (doubled due to interpolation)
        output_fps = fps * 2
        
        # Extract audio from original video
        audio_start = time.time()
        audio_path = self.extract_audio(original_video_path, temp_dir)
        audio_time = time.time() - audio_start
        
        # Create video with FFmpeg
        video_start = time.time()
        if audio_path and os.path.exists(audio_path):
            # With audio track
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-framerate', str(output_fps),
                '-i', os.path.join(rife_dir, 'frame_%06d.png'),
                '-i', audio_path,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '18',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-pix_fmt', 'yuv420p',
                '-map', '0:v:0',  # Map video from first input
                '-map', '1:a:0',  # Map audio from second input
                '-shortest',      # Match shortest stream duration
                output_path
            ]
            logger.info("Creating video with preserved audio")
        else:
            # Video only (no audio)
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-framerate', str(output_fps),
                '-i', os.path.join(rife_dir, 'frame_%06d.png'),
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '18',
                '-pix_fmt', 'yuv420p',
                output_path
            ]
            logger.info("Creating video without audio (no audio track found)")
        
        subprocess.run(ffmpeg_cmd, check=True)
        video_creation_time = time.time() - video_start
        total_time = time.time() - start_time
        
        logger.info(f"Video created: {output_path} in {total_time:.2f}s (audio: {audio_time:.2f}s, encoding: {video_creation_time:.2f}s)")
        
        return {
            'processing_time': total_time,
            'audio_extraction_time': audio_time,
            'video_encoding_time': video_creation_time,
            'output_resolution': f"{width}x{height}",
            'output_fps': output_fps,
            'stage': 'video_creation'
        }
    
    def process_video(self, input_path: str, output_path: str) -> Dict:
        """Main video processing pipeline"""
        start_time = time.time()
        
        # Validate input
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Processing video: {input_path}")
            
            # Extract frames
            frame_count, fps = self.extract_frames(input_path, temp_dir)
            
            # Apply processing pipeline
            self.apply_gfpgan(temp_dir, frame_count)
            self.apply_esrgan(temp_dir, frame_count)
            final_frame_count = self.apply_rife_interpolation(temp_dir, frame_count, fps)
            
            # Create final video with preserved audio
            self.create_video(temp_dir, output_path, fps, final_frame_count, input_path)
        
    def process_video(self, input_path: str, output_path: str) -> Dict:
        """Main video processing pipeline"""
        pipeline_start = time.time()
        
        # Validate input
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        # Get input file size for efficiency calculations
        input_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Processing video: {input_path} ({input_size_mb:.1f} MB)")
            
            # Stage timings
            stage_metrics = {}
            
            # Extract frames
            frame_count, fps, extraction_metrics = self.extract_frames(input_path, temp_dir)
            stage_metrics['extraction'] = extraction_metrics
            
            # Apply processing pipeline
            gfpgan_metrics = self.apply_gfpgan(temp_dir, frame_count)
            stage_metrics['gfpgan'] = gfpgan_metrics
            
            esrgan_metrics = self.apply_esrgan(temp_dir, frame_count)
            stage_metrics['esrgan'] = esrgan_metrics
            
            final_frame_count, rife_metrics = self.apply_rife_interpolation(temp_dir, frame_count, fps)
            stage_metrics['rife'] = rife_metrics
            
            # Create final video with preserved audio
            video_metrics = self.create_video(temp_dir, output_path, fps, final_frame_count, input_path)
            stage_metrics['video_creation'] = video_metrics
        
        # Calculate total processing time and efficiency metrics
        total_processing_time = time.time() - pipeline_start
        
        # Get output file size
        output_size_mb = os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else 0
        
        # Calculate efficiency metrics
        video_duration = extraction_metrics['video_duration']
        processing_ratio = total_processing_time / video_duration if video_duration > 0 else 0
        mb_per_second = input_size_mb / total_processing_time if total_processing_time > 0 else 0
        
        # Check if audio was preserved
        audio_preserved = False
        audio_info = {}
        
        if os.path.exists(output_path):
            # Use FFprobe to check audio streams in output
            try:
                ffprobe_cmd = [
                    'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                    '-show_streams', '-select_streams', 'a', output_path
                ]
                result = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    import json
                    probe_data = json.loads(result.stdout)
                    if probe_data.get('streams'):
                        audio_preserved = True
                        audio_stream = probe_data['streams'][0]
                        audio_info = {
                            'codec': audio_stream.get('codec_name', 'unknown'),
                            'bitrate': audio_stream.get('bit_rate', 'unknown'),
                            'sample_rate': audio_stream.get('sample_rate', 'unknown')
                        }
            except:
                pass
        
        # Compile comprehensive results
        result = {
            'input_path': input_path,
            'output_path': output_path,
            'original_frames': frame_count,
            'final_frames': final_frame_count,
            'original_fps': fps,
            'output_fps': fps * 2,
            'processing_time': total_processing_time,
            'upscale_factor': 2,
            'audio_preserved': audio_preserved,
            'audio_info': audio_info,
            
            # File metrics
            'input_size_mb': input_size_mb,
            'output_size_mb': output_size_mb,
            'size_increase_ratio': output_size_mb / input_size_mb if input_size_mb > 0 else 0,
            
            # Efficiency metrics
            'video_duration_seconds': video_duration,
            'processing_ratio': processing_ratio,
            'mb_processed_per_second': mb_per_second,
            'frames_processed_per_second': frame_count / total_processing_time if total_processing_time > 0 else 0,
            
            # Stage-by-stage breakdown
            'stage_timings': {
                'extraction': {
                    'time_seconds': extraction_metrics['processing_time'],
                    'percentage': (extraction_metrics['processing_time'] / total_processing_time) * 100,
                    'fps': extraction_metrics['extraction_fps']
                },
                'gfpgan': {
                    'time_seconds': gfpgan_metrics['processing_time'],
                    'percentage': (gfpgan_metrics['processing_time'] / total_processing_time) * 100,
                    'fps': gfpgan_metrics['fps']
                },
                'esrgan': {
                    'time_seconds': esrgan_metrics['processing_time'],
                    'percentage': (esrgan_metrics['processing_time'] / total_processing_time) * 100,
                    'fps': esrgan_metrics['fps']
                },
                'rife': {
                    'time_seconds': rife_metrics['processing_time'],
                    'percentage': (rife_metrics['processing_time'] / total_processing_time) * 100,
                    'pairs_per_second': rife_metrics['pairs_per_second']
                },
                'video_creation': {
                    'time_seconds': video_metrics['processing_time'],
                    'percentage': (video_metrics['processing_time'] / total_processing_time) * 100,
                    'encoding_fps': final_frame_count / video_metrics['video_encoding_time'] if video_metrics['video_encoding_time'] > 0 else 0
                }
            },
            
            # Performance summary
            'performance_summary': {
                'total_time_formatted': f"{int(total_processing_time // 60)}m {int(total_processing_time % 60)}s",
                'slowest_stage': max(stage_metrics.keys(), key=lambda k: stage_metrics[k]['processing_time']),
                'efficiency_rating': 'Excellent' if processing_ratio < 60 else 'Good' if processing_ratio < 120 else 'Fair' if processing_ratio < 300 else 'Slow',
                'throughput_rating': 'High' if mb_per_second > 1 else 'Medium' if mb_per_second > 0.5 else 'Low'
            }
        }
        
        # Log performance summary
        logger.info(f"=== PROCESSING COMPLETE ===")
        logger.info(f"Total time: {result['performance_summary']['total_time_formatted']}")
        logger.info(f"Processing ratio: {processing_ratio:.1f}x (took {processing_ratio:.1f}x longer than video duration)")
        logger.info(f"Throughput: {mb_per_second:.2f} MB/s")
        logger.info(f"Efficiency: {result['performance_summary']['efficiency_rating']}")
        logger.info(f"Slowest stage: {result['performance_summary']['slowest_stage']}")
        logger.info(f"Output size: {output_size_mb:.1f} MB ({result['size_increase_ratio']:.1f}x larger)")
        
        return result

# Initialize processor
processor = VideoProcessor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with detailed system metrics"""
    # Get current resource usage
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # GPU information
    gpu_info = {}
    if processor.resource_manager.gpu_available:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_info = {
                    'name': gpu.name,
                    'memory_total': gpu.memoryTotal,
                    'memory_used': gpu.memoryUsed,
                    'memory_free': gpu.memoryFree,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'temperature': gpu.temperature,
                    'load': gpu.load * 100
                }
        except:
            gpu_info = {'error': 'Could not retrieve GPU stats'}
    
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'system_resources': {
            'cpu': {
                'cores': processor.resource_manager.cpu_cores,
                'threads': processor.resource_manager.cpu_threads,
                'usage_percent': cpu_percent
            },
            'memory': {
                'total_gb': processor.resource_manager.total_ram,
                'available_gb': memory.available // (1024**3),
                'used_gb': memory.used // (1024**3),
                'usage_percent': memory.percent
            },
            'disk': {
                'total_gb': disk.total // (1024**3),
                'free_gb': disk.free // (1024**3),
                'usage_percent': (disk.used / disk.total) * 100
            },
            'gpu': gpu_info if gpu_info else {'available': False}
        },
        'optimization_settings': {
            'optimal_batch_size': processor.resource_manager.get_optimal_batch_size(),
            'optimal_tile_size': processor.resource_manager.get_optimal_tile_size(),
            'device': str(processor.device)
        }
    })

@app.route('/process', methods=['POST'])
def process_video():
    """Process video endpoint"""
    try:
        data = request.json
        input_path = data.get('input_path')
        output_path = data.get('output_path')
        
        if not input_path:
            return jsonify({'error': 'input_path is required'}), 400
        
        if not output_path:
            # Generate output path
            input_dir = os.path.dirname(input_path)
            input_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(input_dir, f"{input_name}_processed.mp4")
        
        # Process video
        result = processor.process_video(input_path, output_path)
        
        return jsonify({
            'status': 'success',
            'result': result
        })
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/process_async', methods=['POST'])
def process_video_async():
    """Asynchronous video processing endpoint"""
    try:
        data = request.json
        input_path = data.get('input_path')
        output_path = data.get('output_path')
        
        if not input_path:
            return jsonify({'error': 'input_path is required'}), 400
        
        if not output_path:
            input_dir = os.path.dirname(input_path)
            input_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(input_dir, f"{input_name}_processed.mp4")
        
        # Start processing in background thread
        def background_process():
            try:
                result = processor.process_video(input_path, output_path)
                logger.info(f"Background processing completed: {result}")
            except Exception as e:
                logger.error(f"Background processing error: {e}")
        
        thread = Thread(target=background_process)
        thread.start()
        
        return jsonify({
            'status': 'accepted',
            'message': 'Processing started in background',
            'output_path': output_path
        })
        
    except Exception as e:
        logger.error(f"Async processing error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # Create weights directory
    os.makedirs('weights', exist_ok=True)
    
    # Start the API server
    app.run(host='0.0.0.0', port=5000, debug=False)
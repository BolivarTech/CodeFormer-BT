'''
The code is modified from the Real-ESRGAN:
https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan_video.py

'''
import cv2
import sys
import subprocess
import numpy as np

try:
    import ffmpeg
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ffmpeg-python'])
    import ffmpeg


# =============================================================================
# FFmpeg CUDA/NVENC Detection
# =============================================================================

def get_ffmpeg_cuda_info():
    """
    Check FFmpeg for CUDA/NVENC support and return detailed info.

    Returns:
        dict: {
            'nvenc_available': bool,
            'nvdec_available': bool,
            'cuda_hwaccel': bool,
            'encoders': list of available NVENC encoders,
            'decoders': list of available NVDEC decoders
        }
    """
    info = {
        'nvenc_available': False,
        'nvdec_available': False,
        'cuda_hwaccel': False,
        'encoders': [],
        'decoders': [],
        'ffmpeg_version': 'unknown'
    }

    try:
        # Check FFmpeg version
        version_result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True, text=True, timeout=10
        )
        if version_result.returncode == 0:
            first_line = version_result.stdout.split('\n')[0]
            info['ffmpeg_version'] = first_line.split(' ')[2] if len(first_line.split(' ')) > 2 else 'unknown'

        # Check hardware accelerators
        hwaccel_result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-hwaccels'],
            capture_output=True, text=True, timeout=10
        )
        if hwaccel_result.returncode == 0:
            info['cuda_hwaccel'] = 'cuda' in hwaccel_result.stdout.lower()

        # Check NVENC encoders
        encoder_result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True, text=True, timeout=10
        )
        if encoder_result.returncode == 0:
            nvenc_encoders = []
            for line in encoder_result.stdout.split('\n'):
                if 'nvenc' in line.lower():
                    parts = line.split()
                    if len(parts) >= 2:
                        nvenc_encoders.append(parts[1])
            info['encoders'] = nvenc_encoders
            info['nvenc_available'] = 'h264_nvenc' in nvenc_encoders

        # Check NVDEC decoders
        decoder_result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-decoders'],
            capture_output=True, text=True, timeout=10
        )
        if decoder_result.returncode == 0:
            nvdec_decoders = []
            for line in decoder_result.stdout.split('\n'):
                if 'cuvid' in line.lower():
                    parts = line.split()
                    if len(parts) >= 2:
                        nvdec_decoders.append(parts[1])
            info['decoders'] = nvdec_decoders
            info['nvdec_available'] = 'h264_cuvid' in nvdec_decoders

    except FileNotFoundError:
        print('FFmpeg not found. Please install FFmpeg.')
    except subprocess.TimeoutExpired:
        print('FFmpeg check timed out.')
    except Exception as e:
        print(f'Error checking FFmpeg capabilities: {e}')

    return info


def print_decoding_banner(use_gpu: bool, info: dict):
    """Print a clear and noticeable banner showing decoding mode."""

    width = 60
    border = '=' * width

    if use_gpu:
        mode = 'GPU (NVIDIA NVDEC/CUVID)'
        color_start = '\033[92m'  # Green
    else:
        mode = 'CPU (Software Decoder)'
        color_start = '\033[93m'  # Yellow

    color_end = '\033[0m'

    print(f'\n{color_start}{border}')
    print(f'{"VIDEO DECODING":^{width}}')
    print(border)
    print(f'  Mode:      {mode}')
    print(f'  FFmpeg:    {info.get("ffmpeg_version", "unknown")}')
    if use_gpu and info.get('decoders'):
        print(f'  NVDEC:     {", ".join(info["decoders"][:4])}')
    print(border)
    print(f'{color_end}')


def print_encoding_banner(use_gpu: bool, codec: str, info: dict):
    """Print a clear and noticeable banner showing encoding mode."""

    width = 60
    border = '=' * width

    if use_gpu:
        mode = 'GPU (NVIDIA NVENC)'
        color_start = '\033[92m'  # Green
    else:
        mode = 'CPU (libx264)'
        color_start = '\033[93m'  # Yellow

    color_end = '\033[0m'

    print(f'\n{color_start}{border}')
    print(f'{"VIDEO ENCODING":^{width}}')
    print(border)
    print(f'  Mode:      {mode}')
    print(f'  Codec:     {codec}')
    print(f'  FFmpeg:    {info.get("ffmpeg_version", "unknown")}')
    if use_gpu and info.get('encoders'):
        print(f'  NVENC:     {", ".join(info["encoders"][:3])}')
    print(border)
    print(f'{color_end}')


def print_cuda_not_available_warning(info: dict, operation: str = 'encoding'):
    """Print a warning when CUDA is requested but not available."""

    width = 60
    border = '!' * width
    color_start = '\033[91m'  # Red
    color_end = '\033[0m'

    print(f'\n{color_start}{border}')
    print(f'{f"WARNING: CUDA {operation.upper()} NOT AVAILABLE":^{width}}')
    print(border)
    if operation == 'encoding':
        print(f'  NVENC encoders: {"Not found" if not info["encoders"] else "Found"}')
    else:
        print(f'  NVDEC decoders: {"Not found" if not info["decoders"] else "Found"}')
    print(f'  CUDA hwaccel:   {"No" if not info["cuda_hwaccel"] else "Yes"}')
    print(f'  ')
    print(f'  Falling back to CPU {operation} (slower)')
    print(f'  ')
    print(f'  To enable CUDA {operation}:')
    print(f'  1. Install NVIDIA GPU drivers')
    print(f'  2. Use FFmpeg with NVENC/NVDEC support')
    print(border)
    print(f'{color_end}')


# Initialize CUDA info at module load
FFMPEG_CUDA_INFO = get_ffmpeg_cuda_info()


# =============================================================================
# Video Reader/Writer Classes
# =============================================================================

def get_video_meta_info(video_path):
    ret = {}
    probe = ffmpeg.probe(video_path)
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
    ret['width'] = video_streams[0]['width']
    ret['height'] = video_streams[0]['height']
    ret['fps'] = eval(video_streams[0]['avg_frame_rate'])
    ret['audio'] = ffmpeg.input(video_path).audio if has_audio else None
    ret['nb_frames'] = int(video_streams[0]['nb_frames'])
    ret['codec_name'] = video_streams[0].get('codec_name', 'unknown')
    return ret


def get_cuvid_decoder(codec_name: str) -> str:
    """Get the appropriate CUVID decoder for a given codec."""
    codec_map = {
        'h264': 'h264_cuvid',
        'hevc': 'hevc_cuvid',
        'h265': 'hevc_cuvid',
        'vp8': 'vp8_cuvid',
        'vp9': 'vp9_cuvid',
        'mpeg1video': 'mpeg1_cuvid',
        'mpeg2video': 'mpeg2_cuvid',
        'mpeg4': 'mpeg4_cuvid',
        'mjpeg': 'mjpeg_cuvid',
        'vc1': 'vc1_cuvid',
        'av1': 'av1_cuvid',
    }
    return codec_map.get(codec_name.lower(), None)


class VideoReader:
    def __init__(self, video_path, use_cuda=True):
        self.paths = []  # for image&folder type
        self.audio = None
        self.use_gpu = False

        # Get video metadata first
        try:
            meta = get_video_meta_info(video_path)
        except Exception as e:
            print(f'Error reading video metadata: {e}')
            sys.exit(1)

        self.width = meta['width']
        self.height = meta['height']
        self.input_fps = meta['fps']
        self.audio = meta['audio']
        self.nb_frames = meta['nb_frames']
        codec_name = meta['codec_name']

        # Determine if we can use CUDA decoding
        cuda_available = FFMPEG_CUDA_INFO['nvdec_available'] and FFMPEG_CUDA_INFO['cuda_hwaccel']
        cuvid_decoder = get_cuvid_decoder(codec_name)
        can_use_cuda = cuda_available and cuvid_decoder and cuvid_decoder in FFMPEG_CUDA_INFO['decoders']

        self.use_gpu = use_cuda and can_use_cuda

        # Show warning if CUDA requested but not available
        if use_cuda and not can_use_cuda:
            if not cuda_available:
                print_cuda_not_available_warning(FFMPEG_CUDA_INFO, 'decoding')
            elif not cuvid_decoder:
                print(f'\n\033[93m[WARNING] No CUDA decoder available for codec: {codec_name}\033[0m')
                print(f'\033[93m          Falling back to CPU decoding\033[0m\n')

        # Print decoding banner
        print_decoding_banner(self.use_gpu, FFMPEG_CUDA_INFO)

        try:
            if self.use_gpu:
                # Use CUDA hardware acceleration for decoding
                self.stream_reader = (
                    ffmpeg.input(video_path, hwaccel='cuda', vcodec=cuvid_decoder)
                    .output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel='error')
                    .run_async(pipe_stdin=True, pipe_stdout=True, cmd='ffmpeg'))
            else:
                # Use CPU software decoding
                self.stream_reader = (
                    ffmpeg.input(video_path)
                    .output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel='error')
                    .run_async(pipe_stdin=True, pipe_stdout=True, cmd='ffmpeg'))
        except FileNotFoundError:
            print('Please install ffmpeg (not ffmpeg-python).\n',
                  'Download from: https://ffmpeg.org/download.html\n',
                  'Or install via: pip install ffmpeg-python')
            sys.exit(0)
        except Exception as e:
            # If CUDA decoding fails, fall back to CPU
            if self.use_gpu:
                print(f'\n\033[93m[WARNING] CUDA decoding failed: {e}\033[0m')
                print(f'\033[93m          Falling back to CPU decoding\033[0m\n')
                self.use_gpu = False
                print_decoding_banner(self.use_gpu, FFMPEG_CUDA_INFO)
                self.stream_reader = (
                    ffmpeg.input(video_path)
                    .output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel='error')
                    .run_async(pipe_stdin=True, pipe_stdout=True, cmd='ffmpeg'))
            else:
                raise

        self.idx = 0

    def get_resolution(self):
        return self.height, self.width

    def get_fps(self):
        if self.input_fps is not None:
            return self.input_fps
        return 24

    def get_audio(self):
        return self.audio

    def __len__(self):
        return self.nb_frames

    def get_frame_from_stream(self):
        img_bytes = self.stream_reader.stdout.read(self.width * self.height * 3)  # 3 bytes for one pixel
        if not img_bytes:
            return None
        img = np.frombuffer(img_bytes, np.uint8).reshape([self.height, self.width, 3])
        return img

    def get_frame_from_list(self):
        if self.idx >= self.nb_frames:
            return None
        img = cv2.imread(self.paths[self.idx])
        self.idx += 1
        return img

    def get_frame(self):
        return self.get_frame_from_stream()

    def close(self):
        self.stream_reader.stdin.close()
        self.stream_reader.wait()


class VideoWriter:
    def __init__(self, video_save_path, height, width, fps, audio, use_cuda=True):
        self.use_gpu = False

        if height > 2160:
            print('You are generating video that is larger than 4K, which will be very slow due to IO speed.',
                  'We highly recommend to decrease the outscale(aka, -s).')

        # Determine encoding mode based on CUDA availability
        cuda_available = FFMPEG_CUDA_INFO['nvenc_available']
        self.use_gpu = use_cuda and cuda_available

        if self.use_gpu:
            vcodec = 'h264_nvenc'
            codec_params = {
                'preset': 'p4',      # Balance between speed and quality (p1=fastest, p7=best quality)
                'tune': 'hq',        # High quality tuning
                'rc': 'vbr',         # Variable bitrate
                'cq': '19',          # Constant quality (lower = better, 19 is visually lossless)
            }
        else:
            vcodec = 'libx264'
            codec_params = {
                'preset': 'medium',
                'crf': '17',
            }
            # Show warning if CUDA was requested but not available
            if use_cuda and not cuda_available:
                print_cuda_not_available_warning(FFMPEG_CUDA_INFO, 'encoding')

        # Print encoding banner
        print_encoding_banner(self.use_gpu, vcodec, FFMPEG_CUDA_INFO)

        if audio is not None:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}',
                            framerate=fps).output(
                                audio,
                                video_save_path,
                                pix_fmt='yuv420p',
                                vcodec=vcodec,
                                loglevel='error',
                                acodec='copy',
                                **codec_params).overwrite_output().run_async(
                                    pipe_stdin=True, pipe_stdout=True, cmd='ffmpeg'))
        else:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}',
                            framerate=fps).output(
                                video_save_path, pix_fmt='yuv420p', vcodec=vcodec,
                                loglevel='error',
                                **codec_params).overwrite_output().run_async(
                                    pipe_stdin=True, pipe_stdout=True, cmd='ffmpeg'))

    def write_frame(self, frame):
        try:
            frame = frame.astype(np.uint8).tobytes()
            self.stream_writer.stdin.write(frame)
        except BrokenPipeError:
            print('FFmpeg pipe error. Please reinstall ffmpeg.\n',
                  'Download from: https://ffmpeg.org/download.html')
            sys.exit(0)

    def close(self):
        self.stream_writer.stdin.close()
        self.stream_writer.wait()

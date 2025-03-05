# Internal helpers

try: 
    import numpy as np
except ImportError:
    np = None

def is_numpy_audio_signal(data):
    """
    Returns True if 'data' is a numpy array that likely represents an audio signal.
    
    An audio signal is typically:
      - A numpy array with a numeric dtype.
      - Either 1-dimensional (mono) or 2-dimensional (multi-channel).
      - In the 2-D case, the second dimension usually represents channels (commonly 1 or 2).
    """

    if np is None:
        return False

    # Check if it's a numpy array
    if not isinstance(data, np.ndarray):
        return False
    
    # Check if the dtype is numeric (e.g., int16, float32, etc.)
    if not np.issubdtype(data.dtype, np.number):
        return False
    
    # Check the number of dimensions: 1-D (mono) or 2-D (multi-channel)
    if data.ndim == 1:
        return True
    elif data.ndim == 2:
        # Optionally, you might enforce a reasonable limit on the number of channels.
        # Common audio signals have 1 (mono) or 2 (stereo) channels.
        if data.shape[1] in (1, 2):
            return True
        # You could relax this if you expect multi-channel audio.
        return True
    return False
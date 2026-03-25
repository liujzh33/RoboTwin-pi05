"""
WebSocket-based model client for connecting to websocket_policy_server.
This adapter makes the WebSocket server compatible with our eval_policy_client.py
"""
import websockets.sync.client
import websockets.exceptions
from openpi_client import msgpack_numpy
import numpy as np
from typing import Optional, Dict, Any


class WebsocketModelClient:
    """Adapter that makes WebSocket policy server compatible with our eval interface."""
    
    def __init__(self, host='localhost', port=8000, timeout=30):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._ws = None
        self._packer = msgpack_numpy.Packer()
        
        # Local state management (mimicking PI0 class behavior)
        self.instruction = None
        self.observation_window = None
        self.pi0_step = 50  # Default action horizon
        
        self._connect()
    
    def _connect(self):
        """Connect to WebSocket server"""
        uri = f"ws://{self.host}:{self.port}"
        print(f"🔗 Connecting to WebSocket server at {uri}...")
        
        try:
            # Connect to WebSocket server
            # Note: websockets.sync.client.connect() doesn't support ping_interval/ping_timeout
            # These are handled automatically by the library with default values
            # For long inference times, the server should respond before default timeout
            self._ws = websockets.sync.client.connect(
                uri, 
                compression=None, 
                max_size=None,
                open_timeout=self.timeout
            )
            # Receive metadata from server
            metadata = msgpack_numpy.unpackb(self._ws.recv())
            print(f"✅ Connected to WebSocket server, metadata: {metadata}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to WebSocket server: {str(e)}")
    
    def set_language(self, instruction: str):
        """Set language instruction (stored locally)"""
        self.instruction = instruction
        print(f"✅ Set instruction: {instruction}")
        return None
    
    def update_observation_window(self, input_rgb_arr, input_state):
        """Update observation window (stored locally)"""
        # Format observation window similar to PI0 class
        # input_rgb_arr order: [head_camera, right_camera, left_camera]
        img_front, img_right, img_left = input_rgb_arr[0], input_rgb_arr[1], input_rgb_arr[2]
        
        # Transpose images from HWC to CHW format
        img_front = np.transpose(img_front, (2, 0, 1)) if len(img_front.shape) == 3 else img_front
        img_right = np.transpose(img_right, (2, 0, 1)) if len(img_right.shape) == 3 else img_right
        img_left = np.transpose(img_left, (2, 0, 1)) if len(img_left.shape) == 3 else img_left
        
        # Server expects: head_camera, left_camera, right_camera
        # Note: The server expects 'left_camera' and 'right_camera' (not 'cam_left_wrist' and 'cam_right_wrist')
        # Server also expects 'reward' field (set to 0.0)
        self.observation_window = {
            "state": input_state,
            "images": {
                "head_camera": img_front,      # head_camera (was cam_high)
                "left_camera": img_left,        # left_camera (was cam_left_wrist)
                "right_camera": img_right,       # right_camera (was cam_right_wrist)
            },
            "prompt": self.instruction,
            "reward": np.asarray([0.0], dtype=np.float32),  # Add reward field set to 0.0
        }
        return None
    
    def get_action(self):
        """Get action by calling server's infer method with current observation_window"""
        if self.observation_window is None:
            raise RuntimeError("observation_window is None! Call update_observation_window first.")
        
        # Send observation_window to server via WebSocket
        data = self._packer.pack(self.observation_window)
        try:
            self._ws.send(data)
        except websockets.exceptions.ConnectionClosed:
            # Connection closed, try to reconnect
            print("⚠️ WebSocket connection closed, attempting to reconnect...")
            self._connect()
            self._ws.send(data)
        
        # Receive action from server (with longer timeout for inference)
        try:
            # Use recv() without timeout parameter - websockets library handles keepalive automatically
            # The ping_interval and ping_timeout set in _connect() will handle keepalive
            response = self._ws.recv()
        except websockets.exceptions.ConnectionClosed as e:
            raise RuntimeError(f"WebSocket connection closed while waiting for response: {e}")
        
        if isinstance(response, str):
            # Server sent an error message
            raise RuntimeError(f"Error from inference server:\n{response}")
        
        result = msgpack_numpy.unpackb(response)
        return result.get("actions", result)  # Return actions array
    
    def reset_obsrvationwindows(self):
        """Reset observation window and instruction"""
        self.instruction = None
        self.observation_window = None
        print("✅ Reset observation window and instruction")
        return None
    
    def call(self, func_name=None, obs=None):
        """Compatibility method to match ModelClient interface"""
        if func_name == 'reset_obsrvationwindows' or func_name == 'reset_model':
            return self.reset_obsrvationwindows()
        elif func_name == 'set_language':
            return self.set_language(obs)
        elif func_name == 'update_observation_window':
            if obs and 'rgb' in obs and 'state' in obs:
                return self.update_observation_window(obs['rgb'], obs['state'])
            else:
                raise ValueError("update_observation_window requires obs with 'rgb' and 'state' keys")
        elif func_name == 'get_action':
            return self.get_action()
        else:
            raise AttributeError(f"Unknown method: {func_name}")
    
    def close(self):
        """Close WebSocket connection"""
        if self._ws:
            try:
                self._ws.close()
            except:
                pass
            finally:
                self._ws = None
                print("🔌 WebSocket connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


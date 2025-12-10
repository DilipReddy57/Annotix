"""
Comprehensive Mock Triton Module for Windows Compatibility.

Triton is a Linux-only GPU kernel compiler used by SAM3 for performance optimization.
This mock allows SAM3 to import and run on Windows, falling back to standard PyTorch ops.

The mock handles:
- triton.jit decorator
- triton.autotune decorator
- triton.language module with all common functions
- triton.Config for autotuning
"""

import sys
import functools
from typing import Any, Callable, List, Optional


class MockTLConstexpr:
    """Mock for triton.language.constexpr."""
    def __init__(self, value):
        self.value = value
    
    def __repr__(self):
        return f"constexpr({self.value})"
    
    def __eq__(self, other):
        if isinstance(other, MockTLConstexpr):
            return self.value == other.value
        return self.value == other


class MockTritonLanguage:
    """
    Mock triton.language module.
    Provides all commonly used triton.language functions as no-ops or passthrough.
    """
    
    # Type aliases
    int1 = bool
    int8 = int
    int16 = int
    int32 = int
    int64 = int
    uint8 = int
    uint16 = int
    uint32 = int
    uint64 = int
    float16 = float
    float32 = float
    float64 = float
    bfloat16 = float
    
    # Block types
    pointer_type = type
    block_type = type
    
    @staticmethod
    def constexpr(value):
        """Create a compile-time constant."""
        return MockTLConstexpr(value)
    
    @staticmethod
    def program_id(axis: int) -> int:
        """Get program/block ID along axis."""
        return 0
    
    @staticmethod
    def num_programs(axis: int) -> int:
        """Get number of programs along axis."""
        return 1
    
    @staticmethod
    def arange(start: int, end: int = None):
        """Create a range tensor."""
        if end is None:
            return list(range(start))
        return list(range(start, end))
    
    @staticmethod
    def zeros(shape, dtype=None):
        """Create zero tensor."""
        return 0
    
    @staticmethod
    def full(shape, value, dtype=None):
        """Create tensor filled with value."""
        return value
    
    @staticmethod
    def load(pointer, mask=None, other=None, cache_modifier="", eviction_policy="", volatile=False):
        """Load from memory."""
        return 0
    
    @staticmethod
    def store(pointer, value, mask=None, cache_modifier="", eviction_policy=""):
        """Store to memory."""
        pass
    
    @staticmethod
    def atomic_add(pointer, value, mask=None):
        """Atomic add operation."""
        return value
    
    @staticmethod
    def atomic_max(pointer, value, mask=None):
        """Atomic max operation."""
        return value
    
    @staticmethod
    def atomic_min(pointer, value, mask=None):
        """Atomic min operation."""
        return value
    
    @staticmethod
    def atomic_cas(pointer, compare, value):
        """Atomic compare-and-swap."""
        return value
    
    @staticmethod
    def where(condition, x, y):
        """Conditional selection."""
        return x if condition else y
    
    @staticmethod
    def maximum(x, y):
        """Element-wise maximum."""
        return max(x, y) if not hasattr(x, '__iter__') else x
    
    @staticmethod
    def minimum(x, y):
        """Element-wise minimum."""
        return min(x, y) if not hasattr(x, '__iter__') else x
    
    @staticmethod
    def exp(x):
        """Exponential."""
        import math
        return math.exp(x) if isinstance(x, (int, float)) else x
    
    @staticmethod
    def log(x):
        """Natural logarithm."""
        import math
        return math.log(x) if isinstance(x, (int, float)) and x > 0 else 0
    
    @staticmethod
    def log2(x):
        """Base-2 logarithm."""
        import math
        return math.log2(x) if isinstance(x, (int, float)) and x > 0 else 0
    
    @staticmethod
    def sqrt(x):
        """Square root."""
        import math
        return math.sqrt(x) if isinstance(x, (int, float)) and x >= 0 else x
    
    @staticmethod
    def rsqrt(x):
        """Reciprocal square root."""
        import math
        return 1.0 / math.sqrt(x) if isinstance(x, (int, float)) and x > 0 else x
    
    @staticmethod
    def sigmoid(x):
        """Sigmoid activation."""
        import math
        return 1.0 / (1.0 + math.exp(-x)) if isinstance(x, (int, float)) else x
    
    @staticmethod
    def tanh(x):
        """Hyperbolic tangent."""
        import math
        return math.tanh(x) if isinstance(x, (int, float)) else x
    
    @staticmethod
    def sin(x):
        """Sine."""
        import math
        return math.sin(x) if isinstance(x, (int, float)) else x
    
    @staticmethod
    def cos(x):
        """Cosine."""
        import math
        return math.cos(x) if isinstance(x, (int, float)) else x
    
    @staticmethod
    def abs(x):
        """Absolute value."""
        return abs(x) if isinstance(x, (int, float)) else x
    
    @staticmethod
    def cdiv(a, b):
        """Ceiling division."""
        return (a + b - 1) // b
    
    @staticmethod
    def sum(x, axis=None):
        """Sum reduction."""
        return x
    
    @staticmethod
    def max(x, axis=None):
        """Max reduction."""
        return x
    
    @staticmethod
    def min(x, axis=None):
        """Min reduction."""
        return x
    
    @staticmethod
    def argmax(x, axis=None):
        """Argmax."""
        return 0
    
    @staticmethod
    def argmin(x, axis=None):
        """Argmin."""
        return 0
    
    @staticmethod
    def broadcast_to(x, shape):
        """Broadcast tensor to shape."""
        return x
    
    @staticmethod
    def reshape(x, shape):
        """Reshape tensor."""
        return x
    
    @staticmethod
    def trans(x):
        """Transpose."""
        return x
    
    @staticmethod
    def dot(a, b, allow_tf32=True):
        """Matrix multiplication."""
        return 0
    
    @staticmethod
    def debug_barrier():
        """Debug barrier (no-op)."""
        pass
    
    @staticmethod
    def static_print(*args):
        """Static print (no-op in mock)."""
        pass
    
    @staticmethod
    def static_assert(cond, msg=""):
        """Static assert (no-op in mock)."""
        pass
    
    @staticmethod
    def device_print(prefix, *args):
        """Device print."""
        pass
    
    @staticmethod
    def multiple_of(x, value):
        """Assert x is multiple of value."""
        return x


class MockConfig:
    """Mock for triton.Config used in autotuning."""
    def __init__(self, kwargs=None, num_warps=4, num_stages=2, num_ctas=1, 
                 maxnreg=None, pre_hook=None):
        self.kwargs = kwargs or {}
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.num_ctas = num_ctas
        self.maxnreg = maxnreg
        self.pre_hook = pre_hook


class MockTriton:
    """
    Mock triton module for Windows/CPU compatibility.
    Allows SAM3 to import without errors on non-Linux systems.
    """
    
    language = MockTritonLanguage()
    Config = MockConfig
    
    # Make language module accessible as tl
    @property
    def tl(self):
        return self.language
    
    @staticmethod
    def jit(fn=None, *, version=None, do_not_specialize=None, debug=None, 
            noinline=None, repr=None, launch_metadata=None):
        """
        Mock JIT decorator - returns function unchanged.
        The decorated function won't run on GPU but allows code to import.
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # In mock mode, triton kernels are no-ops
                # The calling code should have fallback PyTorch implementations
                pass
            return wrapper
        
        if fn is not None:
            return decorator(fn)
        return decorator
    
    @staticmethod
    def autotune(configs: List[MockConfig] = None, key: List[str] = None, 
                 prune_configs_by: dict = None, reset_to_zero: List[str] = None,
                 restore_value: List[str] = None, warmup: int = 25, rep: int = 100):
        """Mock autotune decorator."""
        def decorator(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                pass
            return wrapper
        return decorator
    
    @staticmethod
    def heuristics(values: dict):
        """Mock heuristics decorator."""
        def decorator(fn):
            return fn
        return decorator
    
    @staticmethod
    def cdiv(a: int, b: int) -> int:
        """Ceiling division helper."""
        return (a + b - 1) // b
    
    @staticmethod
    def next_power_of_2(x: int) -> int:
        """Return next power of 2 >= x."""
        return 1 << (x - 1).bit_length() if x > 0 else 1


def install_mock_triton():
    """
    Install mock triton modules into sys.modules.
    Call this before importing any SAM3 modules on Windows.
    """
    mock = MockTriton()
    
    # Install main module
    sys.modules['triton'] = mock
    
    # Install language submodule
    sys.modules['triton.language'] = MockTritonLanguage()
    
    # Install other common submodules
    sys.modules['triton.runtime'] = type('MockRuntime', (), {'driver': None})()
    sys.modules['triton.runtime.jit'] = type('MockJit', (), {})()
    
    return mock


# Auto-install on import
if 'triton' not in sys.modules:
    install_mock_triton()
    print("[INFO] Mock triton module installed for Windows/CPU compatibility.")

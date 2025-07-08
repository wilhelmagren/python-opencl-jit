import pyopencl as cl
import hashlib
import argparse
import time
import numpy as np
import warnings

from functools import wraps
from collections import OrderedDict

warnings.simplefilter("ignore", cl.CompilerWarning)


class LRUCache:
    def __init__(self, capacity):
        self._capacity = capacity
        self._cache = OrderedDict()

    def get(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key, value):
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self._capacity:
            self._cache.popitem(last=False)


KERNEL_CACHE = LRUCache(capacity=8)


def get_cl_context():
    platforms = cl.get_platforms()
    gpu_devices = platforms[0].get_devices(device_type=cl.device_type.GPU)
    return cl.Context(devices=gpu_devices)


CONTEXT = get_cl_context()
QUEUE = cl.CommandQueue(CONTEXT, properties=cl.command_queue_properties.PROFILING_ENABLE)


def get_cached_program(ctx, kernel_src):
    key = hashlib.md5(kernel_src.encode("utf-8")).hexdigest()
    cached = KERNEL_CACHE.get(key)
    if cached:
        return cached

    program = cl.Program(ctx, kernel_src).build()
    KERNEL_CACHE.put(key, program)
    return program


def profile_kernel(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        event, memory_bytes = fn(*args, **kwargs)
        event.wait()
        elapsed_ns = event.profile.end - event.profile.start
        elapsed_ms = elapsed_ns / 1e6
        print(f"[OpenCL] Kernel time: {elapsed_ms:.3f} ms | GPU mem: {memory_bytes / 1024:.1f} KB")
        return kwargs.get("result")
    return wrapper


def jit_kernel(kernel_src, kernel_name):
    def decorator(method):
        @wraps(method)
        def wrapper(self, other):
            if self.mode == "numpy":
                print(f"[NumPy] running method '{method.__name__}'")
                return Tensor(method(self, other), mode="numpy")

            if not isinstance(other, Tensor):
                other = Tensor(other)

            assert self.shape == other.shape

            a_np = self.data.ravel()
            b_np = other.data.ravel()
            out_np = np.empty_like(a_np)

            if self.mode == "jit":
                kernel_src_actual = kernel_src + f"\n// {time.perf_counter()}"
                program = cl.Program(CONTEXT, kernel_src_actual).build()
            else:
                program = get_cached_program(CONTEXT, kernel_src)

            if not hasattr(program, "_kernel_cache"):
                program._kernel_cache = {}

            if kernel_name not in program._kernel_cache:
                program._kernel_cache[kernel_name] = getattr(program, kernel_name)

            kernel_fn = program._kernel_cache[kernel_name]

            mf = cl.mem_flags
            a_buf = cl.Buffer(CONTEXT, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
            b_buf = cl.Buffer(CONTEXT, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
            out_buf = cl.Buffer(CONTEXT, mf.WRITE_ONLY, out_np.nbytes)

            @profile_kernel
            def run_kernel():
                event = kernel_fn(QUEUE, (a_np.size, ), None, a_buf, b_buf, out_buf)
                return event, a_np.nbytes + b_np.nbytes + out_np.nbytes
            
            run_kernel()
            cl.enqueue_copy(QUEUE, out_np, out_buf)
            return Tensor(out_np.reshape(self.shape), mode=self.mode)
        return wrapper
    return decorator


class Tensor:
    def __init__(self, data, mode="jit-cache"):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)

        self.data = data.astype(np.float32)
        self.shape = self.data.shape
        self.mode = mode

    @jit_kernel("""
    __kernel void add(__global const float *a,
                      __global const float *b,
                      __global float *res) {
        int gid = get_global_id(0);
        res[gid] = a[gid] + b[gid];
    }
    """, "add")
    def add(self, other):
        return self.data + (other.data if isinstance(other, Tensor) else other)

    @jit_kernel("""
    __kernel void sub(__global const float *a,
                      __global const float *b,
                      __global float *res) {
        int gid = get_global_id(0);
        res[gid] = a[gid] - b[gid];
    }
    """, "sub")
    def sub(self, other):
        return self.data - (other.data if isinstance(other, Tensor) else other)

    @jit_kernel("""
    __kernel void mul(__global const float *a,
                      __global const float *b,
                      __global float *res) {
        int gid = get_global_id(0);
        res[gid] = a[gid] * b[gid];
    }
    """, "mul")
    def mul(self, other):
        return self.data * (other.data if isinstance(other, Tensor) else other)

    def matmul(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        assert self.data.ndim == 2 and other.data.ndim == 2, "matmul only supports 2d"
        M, K = self.data.shape
        K2, N = other.data.shape

        assert K == K2, f"incompatible shapes {self.data.shape} @ {other.data.shape}"

        if self.mode == "numpy":
            result = self.data @ other.data
            return self.__class__(result, mode="numpy")

        A = self.data.astype(np.float32).ravel()
        B = other.data.astype(np.float32).ravel()
        C = np.empty((M * N), dtype=np.float32)

        kernel_src = """
        __kernel void matmul(
            __global const float* A,
            __global const float* B,
            __global float* C,
            int M, int N, int K)
        {
            int row = get_global_id(0);
            int col = get_global_id(1);
            if (row < M && col < N) {
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) {
                    acc += A[row * K + k] * B[k * N + col];
                }
                C[row * N + col] = acc;
            }
        }
        """

        if self.mode == "jit":
            kernel_src += f"\n// {time.perf_counter()}"
            program = cl.Program(CONTEXT, kernel_src).build()
        else:
            program = get_cached_program(CONTEXT, kernel_src)

        if not hasattr(program, "_kernel_cache"):
            program._kernel_cache = {}

        if "matmul" not in program._kernel_cache:
            program._kernel_cache["matmul"] = getattr(program, "matmul")

        kernel_fn = program._kernel_cache["matmul"]

        mf = cl.mem_flags
        a_buf = cl.Buffer(CONTEXT, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        b_buf = cl.Buffer(CONTEXT, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        c_buf = cl.Buffer(CONTEXT, mf.WRITE_ONLY, C.nbytes)

        @profile_kernel
        def run_kernel():
            event = kernel_fn(
                QUEUE, (M, N), None,
                a_buf, b_buf, c_buf,
                np.int32(M), np.int32(N), np.int32(K),
            )

            return event, A.nbytes + B.nbytes + C.nbytes

        run_kernel()
        cl.enqueue_copy(QUEUE, C, c_buf)

        result = C.reshape((M, N))
        return self.__class__(result, mode=self.mode)

    def numpy(self):
        return self.data

    def reshape(self, shape):
        self.data = self.data.reshape(shape)
        self.shape = self.data.shape
        return self

    def __repr__(self):
        return f"Tensor(shape={self.shape}, mode={self.mode})"


def benchmark(mode, runs=50):
    print(f"\n--- Benchmark: {mode.upper()} ---")
    a = Tensor(np.random.rand(1024, 1024).astype(np.float32), mode=mode)
    b = Tensor(np.random.rand(1024, 512).astype(np.float32), mode=mode)
    c = Tensor(np.random.rand(1024 * 512).astype(np.float32), mode=mode)

    # warmup
    c.sub(c)

    times = []
    for i in range(runs):
        start = time.perf_counter()
        d = a.matmul(b)
        d = d.reshape((1024 * 512, ))
        e = d.add(c)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg = sum(times) / runs
    print(f"[{mode}] Avg wall time over {runs} runs: {avg:.3f} ms")
    return avg

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("-n", "--n-runs", help="Number of runs to do")

    args = argparse.parse_args()
    runs = int(args.n_runs)

    navg = benchmark("numpy", runs=runs)
    javg = benchmark("jit", runs=runs)
    jcavg = benchmark("jit-cached", runs=runs)

    print("\n---------------------- SUMMARY -------------------")
    print(f"[NumPy]\t\taverage: {navg:.3f} ms\t({runs=})")
    print(f"[JIT]\t\taverage: {javg:.3f} ms\t({runs=})")
    print(f"[JIT-CACHED]\taverage: {jcavg:.3f} ms\t({runs=})")
    print("--------------------------------------------------")

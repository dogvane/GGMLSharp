using System;

using static GGMLSharp.InternalStructs;

namespace GGMLSharp
{
    /// <summary>
    /// Managed handle for a ggml backend instance.
    /// A backend represents the execution target for graphs and the source of backend-owned buffers,
    /// such as CPU, CUDA, or Vulkan.
    /// </summary>
    public unsafe class SafeGGmlBackend : SafeGGmlHandleBase
    {
        /// <summary>
        /// Exposes the raw native handle for advanced interop scenarios.
        /// </summary>
        public IntPtr NativeHandle => handle;

        /// <summary>
        /// Creates an empty backend handle.
        /// </summary>
        public SafeGGmlBackend()
        {
            this.handle = IntPtr.Zero;
        }

        /// <summary>
        /// Returns the default buffer type used by this backend for tensor allocations.
        /// </summary>
        public SafeGGmlBackendBufferType GetDefaultBufferType()
        {
            return Native.ggml_backend_get_default_buffer_type(this);
        }

        /// <summary>
        /// Creates a CPU backend.
        /// </summary>
        public static SafeGGmlBackend CpuInit()
        {
            return Native.ggml_backend_cpu_init();
        }

        /// <summary>
        /// Creates a CUDA backend for the specified device index.
        /// </summary>
        public static SafeGGmlBackend CudaInit(int index = 0)
        {
            if (!HasCuda)
            {
                throw new NotSupportedException("Cuda Not Support");
            }
            return Native.ggml_backend_cuda_init(index);
        }

        /// <summary>
        /// Creates a Vulkan backend for the specified device index.
        /// </summary>
        public static SafeGGmlBackend VulkanInit(int index = 0)
        {
            if (!HasVulkan)
            {
                throw new NotSupportedException("Vulkan Not Support");
            }
            return Native.ggml_backend_vk_init(index);
        }

        /// <summary>
        /// Returns true when the current native build exposes a CUDA backend registry entry.
        /// </summary>
        public static bool HasCuda
        {
            get
            {
                try
                {
                    return Native.ggml_backend_reg_by_name("CUDA") != IntPtr.Zero;
                }
                catch (DllNotFoundException)
                {
                    return false;
                }
                catch (EntryPointNotFoundException)
                {
                    return false;
                }
            }
        }

        /// <summary>
        /// Returns true when the current native build exposes a Vulkan backend registry entry.
        /// </summary>
        public static bool HasVulkan
        {
            get
            {
                try
                {
                    return Native.ggml_backend_reg_by_name("Vulkan") != IntPtr.Zero;
                }
                catch (DllNotFoundException)
                {
                    return false;
                }
                catch (EntryPointNotFoundException)
                {
                    return false;
                }
            }
        }

        /// <summary>
        /// Frees the native backend instance.
        /// This is an explicit native release; the base SafeHandle only clears the managed handle value.
        /// </summary>
        public void Free()
        {
            Native.ggml_backend_free(this);
        }

        /// <summary>
        /// Allocates a backend-owned buffer with the requested size in bytes.
        /// </summary>
        public SafeGGmlBackendBuffer AllocBuffer(ulong size)
        {
            return Native.ggml_backend_alloc_buffer(this, size);
        }

        /// <summary>
        /// Returns true if this backend is the CPU backend.
        /// </summary>
        public bool IsCpu()
        {
            return Native.ggml_backend_is_cpu(this);
        }

        /// <summary>
        /// Sets the CPU thread count for this backend.
        /// Only meaningful for CPU backends.
        /// </summary>
        public void SetCpuThreads(int n_threads)
        {
            Native.ggml_backend_cpu_set_n_threads(this, n_threads);
        }

        /// <summary>
        /// Computes the provided graph on this backend.
        /// </summary>
        public Structs.GGmlStatus Compute(SafeGGmlGraph graph)
        {
            return (Structs.GGmlStatus)Native.ggml_backend_graph_compute(this, graph);
        }

    }
}

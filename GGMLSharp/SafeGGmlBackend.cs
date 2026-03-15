using System;

using static GGMLSharp.InternalStructs;

namespace GGMLSharp
{
    public unsafe class SafeGGmlBackend : SafeGGmlHandleBase
    {
        public SafeGGmlBackend()
        {
            this.handle = IntPtr.Zero;
        }

        public SafeGGmlBackendBufferType GetDefaultBufferType()
        {
            return Native.ggml_backend_get_default_buffer_type(this);
        }

        public static SafeGGmlBackend CpuInit()
        {
            return Native.ggml_backend_cpu_init();
        }

        public static SafeGGmlBackend CudaInit(int index = 0)
        {
            if (!HasCuda)
            {
                throw new NotSupportedException("Cuda Not Support");
            }
            return Native.ggml_backend_cuda_init(index);
        }

        public static SafeGGmlBackend VulkanInit(int index = 0)
        {
            if (!HasVulkan)
            {
                throw new NotSupportedException("Vulkan Not Support");
            }
            return Native.ggml_backend_vk_init(index);
        }

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

        public void Free()
        {
            Native.ggml_backend_free(this);
        }

    }
}

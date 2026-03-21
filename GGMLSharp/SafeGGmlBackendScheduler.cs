using System;

namespace GGMLSharp
{
    public unsafe class SafeGGmlBackendScheduler : SafeGGmlHandleBase
    {
        public IntPtr NativeHandle => handle;

        public SafeGGmlBackendScheduler()
        {
            handle = IntPtr.Zero;
        }

        private SafeGGmlBackendScheduler(IntPtr handle) : base(handle, true)
        {
        }

        public static SafeGGmlBackendScheduler Create(SafeGGmlBackend[] backends, ulong graphSize = Structs.GGML_DEFAULT_GRAPH_SIZE, bool parallel = false, bool opOffload = true)
        {
            if (backends == null || backends.Length == 0)
            {
                throw new ArgumentException("At least one backend is required.", nameof(backends));
            }

            IntPtr[] backendHandles = new IntPtr[backends.Length];
            for (int i = 0; i < backends.Length; i++)
            {
                if (backends[i] == null || backends[i].IsInvalid)
                {
                    throw new ArgumentException($"Backend at index {i} is null or invalid.", nameof(backends));
                }

                backendHandles[i] = backends[i].NativeHandle;
            }

            fixed (IntPtr* backendsPtr = backendHandles)
            {
                IntPtr scheduler = Native.ggml_backend_sched_new(backendsPtr, null, backendHandles.Length, graphSize, parallel, opOffload);
                if (scheduler == IntPtr.Zero)
                {
                    throw new InvalidOperationException("Failed to create ggml backend scheduler.");
                }

                return new SafeGGmlBackendScheduler(scheduler);
            }
        }

        public void Free()
        {
            if (!IsInvalid)
            {
                Native.ggml_backend_sched_free(handle);
                handle = IntPtr.Zero;
            }
        }

        public void Reset()
        {
            Native.ggml_backend_sched_reset(handle);
        }

        public bool AllocGraph(SafeGGmlGraph graph)
        {
            return Native.ggml_backend_sched_alloc_graph(handle, graph);
        }

        public Structs.GGmlStatus GraphCompute(SafeGGmlGraph graph)
        {
            return (Structs.GGmlStatus)Native.ggml_backend_sched_graph_compute(handle, graph);
        }

        public SafeGGmlBackend GetBackend(int index)
        {
            return Native.ggml_backend_sched_get_backend(handle, index);
        }
    }
}

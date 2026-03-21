using System;
using static GGMLSharp.InternalStructs;

namespace GGMLSharp
{
	/// <summary>
	/// Managed handle for a ggml backend buffer.
	/// A backend buffer is a raw memory allocation owned by a specific ggml backend and can host
	/// tensor storage directly or be used together with allocators such as GGmlTallocr / gallocr.
	/// </summary>
	public unsafe class SafeGGmlBackendBuffer : SafeGGmlHandleBase
	{
		/// <summary>
		/// Creates an empty handle.
		/// </summary>
		public SafeGGmlBackendBuffer()
		{
			this.handle = IntPtr.Zero;
		}

		/// <summary>
		/// Wraps an existing native ggml_backend_buffer pointer.
		/// </summary>
		internal SafeGGmlBackendBuffer(ggml_backend_buffer* ggml_backend_buffer)
		{
			handle = (IntPtr)ggml_backend_buffer;
		}

		/// <summary>
		/// Returns the backend buffer type that created this buffer.
		/// </summary>
		public SafeGGmlBackendBufferType BufferType => Native.ggml_backend_buffer_get_type(this);

		/// <summary>
		/// Frees the native backend buffer.
		/// This is an explicit native release; the base SafeHandle only clears the managed handle value.
		/// </summary>
		public void Free()
		{
			Native.ggml_backend_buffer_free(this);
		}

	}
}

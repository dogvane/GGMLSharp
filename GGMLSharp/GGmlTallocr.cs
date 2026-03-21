using static GGMLSharp.InternalStructs;

namespace GGMLSharp
{
    /// <summary>
    /// Thin wrapper over ggml_tallocr.
    /// Used when a caller already owns a backend buffer and wants to place tensors into it
    /// sequentially with ggml's native alignment rules.
    /// </summary>
    public unsafe sealed class GGmlTallocr
    {
        // Native allocator state. The offset advances each time Alloc is called.
        private ggml_tallocr _tallocr;

        /// <summary>
        /// Creates a linear tensor allocator backed by an existing backend buffer.
        /// </summary>
        public GGmlTallocr(SafeGGmlBackendBuffer buffer)
        {
            _tallocr = Native.ggml_tallocr_new(buffer);
        }

        /// <summary>
        /// Allocates storage for <paramref name="tensor"/> from the wrapped backend buffer.
        /// Allocation is sequential and consumes the current offset inside the tallocr.
        /// </summary>
        public void Alloc(SafeGGmlTensor tensor)
        {
            fixed (ggml_tallocr* tallocr = &_tallocr)
            {
                Native.ggml_tallocr_alloc(tallocr, tensor);
            }
        }
    }
}

using System;
using System.Runtime.InteropServices;

namespace GGMLSharp
{
    /// <summary>
    /// Thin managed wrappers around ggml's quantization traits and CPU quantization helpers.
    /// </summary>
    public static unsafe class GGmlQuantization
    {
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void GgmlToFloatDelegate(void* src, float* dst, long count);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void GgmlFromFloatDelegate(float* src, void* dst, long count);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void GgmlVecDotDelegate(int n, float* sum, nuint bs, void* x, nuint bx, void* y, nuint by, int nrc);

        /// <summary>
        /// Managed view of ggml_type_traits. Function pointers are exposed as IntPtr and resolved lazily.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        private struct NativeTypeTraits
        {
            public IntPtr type_name;
            public long blck_size;
            public long blck_size_interleave;
            public nuint type_size;

            [MarshalAs(UnmanagedType.I1)]
            public bool is_quantized;

            public IntPtr to_float;
            public IntPtr from_float_ref;
        }

        /// <summary>
        /// Managed view of CPU-specific type traits used by the optimized quantizer and vec_dot path.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        private struct NativeTypeTraitsCpu
        {
            public IntPtr from_float;
            public IntPtr vec_dot;
            public Structs.GGmlType vec_dot_type;
            public long nrows;
        }

        /// <summary>
        /// Public metadata for a ggml type. This only describes capability and layout, not actual data.
        /// </summary>
        public readonly record struct TypeTraits(
            string? TypeName,
            long BlockSize,
            long BlockSizeInterleave,
            ulong TypeSize,
            bool IsQuantized,
            bool HasToFloat,
            bool HasFromFloatReference);

        /// <summary>
        /// CPU-side metadata for fast quantization and dot-product kernels.
        /// </summary>
        public readonly record struct CpuTypeTraits(
            bool HasFromFloat,
            bool HasVecDot,
            Structs.GGmlType VecDotType,
            long NRows);

        /// <summary>
        /// Initializes ggml CPU quantization tables. Safe to call before using CPU trait-based helpers.
        /// </summary>
        public static void InitializeCpu()
        {
            Native.ggml_cpu_init();
        }

        /// <summary>
        /// Initializes quantization state for a specific ggml type.
        /// </summary>
        public static void Initialize(Structs.GGmlType type)
        {
            Native.ggml_quantize_init(type);
        }

        /// <summary>
        /// Releases global quantization state allocated by ggml_quantize_init.
        /// </summary>
        public static void Free()
        {
            Native.ggml_quantize_free();
        }

        /// <summary>
        /// Returns generic trait metadata for a ggml type.
        /// </summary>
        public static TypeTraits GetTypeTraits(Structs.GGmlType type)
        {
            IntPtr ptr = Native.ggml_get_type_traits(type);
            if (ptr == IntPtr.Zero)
            {
                throw new InvalidOperationException($"ggml_get_type_traits returned null for {type}.");
            }

            NativeTypeTraits traits = Marshal.PtrToStructure<NativeTypeTraits>(ptr);
            return new TypeTraits(
                Marshal.PtrToStringAnsi(traits.type_name),
                traits.blck_size,
                traits.blck_size_interleave,
                traits.type_size,
                traits.is_quantized,
                traits.to_float != IntPtr.Zero,
                traits.from_float_ref != IntPtr.Zero);
        }

        /// <summary>
        /// Returns CPU-specific trait metadata, including optimized quantizer and vec_dot availability.
        /// </summary>
        public static CpuTypeTraits GetCpuTypeTraits(Structs.GGmlType type)
        {
            IntPtr ptr = Native.ggml_get_type_traits_cpu(type);
            if (ptr == IntPtr.Zero)
            {
                throw new InvalidOperationException($"ggml_get_type_traits_cpu returned null for {type}.");
            }

            NativeTypeTraitsCpu traits = Marshal.PtrToStructure<NativeTypeTraitsCpu>(ptr);
            return new CpuTypeTraits(
                traits.from_float != IntPtr.Zero,
                traits.vec_dot != IntPtr.Zero,
                traits.vec_dot_type,
                traits.nrows);
        }

        /// <summary>
        /// Quantizes <paramref name="input"/> using the CPU-optimized quantizer for <paramref name="type"/>.
        /// </summary>
        public static byte[] Quantize(Structs.GGmlType type, float[] input)
        {
            NativeTypeTraitsCpu cpuTraits = GetNativeCpuTypeTraits(type);
            if (cpuTraits.from_float == IntPtr.Zero)
            {
                throw new InvalidOperationException($"CPU quantizer is not available for {type}.");
            }

            return QuantizeWith(cpuTraits.from_float, type, input);
        }

        /// <summary>
        /// Quantizes <paramref name="input"/> using ggml's reference implementation.
        /// Useful when comparing optimized and reference output.
        /// </summary>
        public static byte[] QuantizeReference(Structs.GGmlType type, float[] input)
        {
            NativeTypeTraits traits = GetNativeTypeTraits(type);
            if (traits.from_float_ref == IntPtr.Zero)
            {
                throw new InvalidOperationException($"Reference quantizer is not available for {type}.");
            }

            return QuantizeWith(traits.from_float_ref, type, input);
        }

        /// <summary>
        /// Converts quantized bytes back to F32 values.
        /// The caller must provide the logical element count to decode.
        /// </summary>
        public static float[] Dequantize(Structs.GGmlType type, byte[] quantized, int count)
        {
            NativeTypeTraits traits = GetNativeTypeTraits(type);
            if (traits.to_float == IntPtr.Zero)
            {
                throw new InvalidOperationException($"Dequantizer is not available for {type}.");
            }

            var output = new float[count];
            var toFloat = Marshal.GetDelegateForFunctionPointer<GgmlToFloatDelegate>(traits.to_float);

            fixed (byte* src = quantized)
            fixed (float* dst = output)
            {
                toFloat(src, dst, count);
            }

            return output;
        }

        /// <summary>
        /// Quantizes input into the companion type required by the CPU vec_dot kernel for <paramref name="type"/>.
        /// </summary>
        public static byte[] QuantizeForVecDot(Structs.GGmlType type, float[] input)
        {
            CpuTypeTraits cpuTraits = GetCpuTypeTraits(type);
            return Quantize(cpuTraits.VecDotType, input);
        }

        /// <summary>
        /// Executes ggml's CPU vec_dot kernel for a quantized type.
        /// <paramref name="count"/> is the number of logical float elements represented by the buffers.
        /// </summary>
        public static float VecDot(Structs.GGmlType type, int count, byte[] lhs, byte[] rhs)
        {
            NativeTypeTraitsCpu cpuTraits = GetNativeCpuTypeTraits(type);
            if (cpuTraits.vec_dot == IntPtr.Zero)
            {
                throw new InvalidOperationException($"vec_dot is not available for {type}.");
            }

            var vecDot = Marshal.GetDelegateForFunctionPointer<GgmlVecDotDelegate>(cpuTraits.vec_dot);
            float sum = float.PositiveInfinity;

            fixed (byte* x = lhs)
            fixed (byte* y = rhs)
            {
                vecDot(count, &sum, 0, x, 0, y, 0, 1);
            }

            return sum;
        }

        /// <summary>
        /// Returns the storage size in bytes for a row with <paramref name="count"/> logical elements.
        /// </summary>
        public static int RowSize(Structs.GGmlType type, int count)
        {
            return checked((int)Native.ggml_row_size(type, count));
        }

        /// <summary>
        /// Shared helper that dispatches either the CPU fast quantizer or the reference quantizer.
        /// </summary>
        private static byte[] QuantizeWith(IntPtr fnPtr, Structs.GGmlType type, float[] input)
        {
            int rowSize = RowSize(type, input.Length);
            var output = new byte[rowSize];
            var fromFloat = Marshal.GetDelegateForFunctionPointer<GgmlFromFloatDelegate>(fnPtr);

            fixed (float* src = input)
            fixed (byte* dst = output)
            {
                fromFloat(src, dst, input.Length);
            }

            return output;
        }

        private static NativeTypeTraits GetNativeTypeTraits(Structs.GGmlType type)
        {
            IntPtr ptr = Native.ggml_get_type_traits(type);
            if (ptr == IntPtr.Zero)
            {
                throw new InvalidOperationException($"ggml_get_type_traits returned null for {type}.");
            }

            return Marshal.PtrToStructure<NativeTypeTraits>(ptr);
        }

        private static NativeTypeTraitsCpu GetNativeCpuTypeTraits(Structs.GGmlType type)
        {
            IntPtr ptr = Native.ggml_get_type_traits_cpu(type);
            if (ptr == IntPtr.Zero)
            {
                throw new InvalidOperationException($"ggml_get_type_traits_cpu returned null for {type}.");
            }

            return Marshal.PtrToStructure<NativeTypeTraitsCpu>(ptr);
        }
    }
}

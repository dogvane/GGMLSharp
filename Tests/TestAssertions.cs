using GGMLSharp;
using System;

namespace GGMLSharp.Tests
{
    internal static class TestAssertions
    {
        public static void Equal(long expected, long actual, string message)
        {
            if (expected != actual)
            {
                throw new Exception($"{message}: expected {expected}, got {actual}");
            }
        }

        public static void Equal(float expected, float actual, string message)
        {
            if (expected != actual)
            {
                throw new Exception($"{message}: expected {expected}, got {actual}");
            }
        }

        public static void Near(float expected, float actual, float tolerance, string message)
        {
            if (Math.Abs(expected - actual) > tolerance)
            {
                throw new Exception($"{message}: expected {expected}, got {actual}, tolerance {tolerance}");
            }
        }

        public static void RelativeNear(float expected, float actual, float tolerance, string message)
        {
            float scale = Math.Abs(expected);
            if (scale <= float.Epsilon)
            {
                if (Math.Abs(actual) > tolerance)
                {
                    throw new Exception($"{message}: expected {expected}, got {actual}, tolerance {tolerance}");
                }

                return;
            }

            if (Math.Abs(expected - actual) / scale > tolerance)
            {
                throw new Exception($"{message}: expected {expected}, got {actual}, relative tolerance {tolerance}");
            }
        }

        public static void Shape(SafeGGmlTensor tensor, params long[] expected)
        {
            var actual = tensor.Shape;
            for (int i = 0; i < expected.Length; i++)
            {
                Equal(expected[i], actual[i], $"shape[{i}]");
            }
        }

        public static void FloatDataExact(SafeGGmlTensor tensor, params float[] expected)
        {
            var actual = ReadFloatData(tensor);
            Equal(expected.Length, actual.Length, "float data length");

            for (int i = 0; i < expected.Length; i++)
            {
                Equal(expected[i], actual[i], $"float data[{i}]");
            }
        }

        public static void FloatDataNear(SafeGGmlTensor tensor, float tolerance, params float[] expected)
        {
            var actual = ReadFloatData(tensor);
            Equal(expected.Length, actual.Length, "float data length");

            for (int i = 0; i < expected.Length; i++)
            {
                Near(expected[i], actual[i], tolerance, $"float data[{i}]");
            }
        }

        public static void FloatDataRelativeNear(SafeGGmlTensor tensor, float tolerance, params float[] expected)
        {
            var actual = ReadFloatData(tensor);
            Equal(expected.Length, actual.Length, "float data length");

            for (int i = 0; i < expected.Length; i++)
            {
                RelativeNear(expected[i], actual[i], tolerance, $"float data[{i}]");
            }
        }

        public static float[] ReadFloatData(SafeGGmlTensor tensor)
        {
            var bytes = tensor.GetBackend();

            return tensor.Type switch
            {
                Structs.GGmlType.GGML_TYPE_F32 => DataConverter.ConvertToFloats(bytes),
                Structs.GGmlType.GGML_TYPE_F16 => ReadFp16(bytes),
                _ => throw new NotSupportedException($"Unsupported tensor type: {tensor.Type}"),
            };
        }

        public static byte[] ToFp16Bytes(float[] values)
        {
            var halfs = DataConverter.F32ToF16(values);
            var bytes = new byte[halfs.Length * sizeof(ushort)];
            Buffer.BlockCopy(halfs, 0, bytes, 0, bytes.Length);
            return bytes;
        }

        private static float[] ReadFp16(byte[] bytes)
        {
            var fp32Bytes = (byte[])bytes.Clone();
            DataConverter.Fp16ToFp32Bytes(ref fp32Bytes);
            return DataConverter.ConvertToFloats(fp32Bytes);
        }
    }
}

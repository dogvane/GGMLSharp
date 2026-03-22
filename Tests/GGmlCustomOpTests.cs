using GGMLSharp;
using System;
using System.Runtime.InteropServices;
using System.Threading;

namespace GGMLSharp.Tests
{
    public class GGmlCustomOpTests
    {
        public void Test_CustomOps_Match_C()
        {
            var userData = Marshal.StringToHGlobalAnsi("ggml");
            try
            {
                TestMapCustom1();
                TestMapCustom2(userData);
                TestMapCustom3(userData);
                TestCustom4d(userData);
            }
            finally
            {
                Marshal.FreeHGlobal(userData);
            }
        }

        private static void TestMapCustom1()
        {
            using var ctx = CreateContext();

            var input = ctx.NewTensor2d(Structs.GGmlType.GGML_TYPE_F32, 10, 2);
            SetTensorData(input, CreateScaledRange(1.0f));

            var count = 0;
            Structs.Custom1OpDelegate custom1 = (dst, a, ith, nth, userdata) =>
            {
                if (userdata != IntPtr.Zero)
                {
                    throw new Exception("custom1 userdata should be null");
                }

                TestAssertions.Equal(dst.ElementsCount, a.ElementsCount, "custom1 shape");
                Interlocked.Increment(ref count);

                var ne = (int)dst.ElementsCount;
                var dr = (ne + nth - 1) / nth;
                var start = dr * ith;
                var end = Math.Min(start + dr, ne);

                for (int i = start; i < end; i++)
                {
                    dst.SetData(a.GetFloat(i) * 2.0f, i);
                }
            };

            var output = ctx.MapCustom1(input, custom1, 2, IntPtr.Zero);
            ExecuteGraph(ctx, output, 4);

            var actual = output.GetDataInFloats();
            var expected = CreateScaledRange(2.0f);
            AssertFloatArray(actual, expected, "custom1");
            TestAssertions.Equal(2, count, "custom1 count");
        }

        private static void TestMapCustom2(IntPtr userData)
        {
            using var ctx = CreateContext();

            var a = ctx.NewTensor2d(Structs.GGmlType.GGML_TYPE_F32, 10, 2);
            var b = ctx.NewTensor2d(Structs.GGmlType.GGML_TYPE_F32, 10, 2);
            SetTensorData(a, CreateScaledRange(1.0f));
            SetTensorData(b, CreateScaledRange(2.0f));

            var count = 0;
            Structs.Custom2OpDelegate custom2 = (dst, src0, src1, ith, nth, userdata) =>
            {
                AssertUserData(userData, userdata, "custom2");
                TestAssertions.Equal(dst.ElementsCount, src0.ElementsCount, "custom2 shape a");
                TestAssertions.Equal(dst.ElementsCount, src1.ElementsCount, "custom2 shape b");
                Interlocked.Increment(ref count);

                var nr = (int)dst.Shape[1];
                var dr = (nr + nth - 1) / nth;
                var rowStart = dr * ith;
                var rowEnd = Math.Min(rowStart + dr, nr);
                var nc = (int)dst.Shape[0];

                for (int row = rowStart; row < rowEnd; row++)
                {
                    for (int col = 0; col < nc; col++)
                    {
                        dst.SetData(src0.GetFloat(col, row) + src1.GetFloat(col, row), col, row);
                    }
                }
            };

            var output = ctx.MapCustom2(a, b, custom2, Structs.GGML_N_TASKS_MAX, userData);
            ExecuteGraph(ctx, output, 4);

            var actual = output.GetDataInFloats();
            var expected = new float[actual.Length];
            for (int i = 0; i < expected.Length; i++)
            {
                expected[i] = (i + 1) + (i + 1) * 2;
            }

            AssertFloatArray(actual, expected, "custom2");
            TestAssertions.Equal(4, count, "custom2 count");
        }

        private static void TestMapCustom3(IntPtr userData)
        {
            using var ctx = CreateContext();

            var a = ctx.NewTensor2d(Structs.GGmlType.GGML_TYPE_F32, 10, 2);
            var b = ctx.NewTensor2d(Structs.GGmlType.GGML_TYPE_F32, 10, 2);
            var c = ctx.NewTensor2d(Structs.GGmlType.GGML_TYPE_F32, 10, 2);
            SetTensorData(a, CreateScaledRange(1.0f));
            SetTensorData(b, CreateScaledRange(2.0f));
            SetTensorData(c, CreateScaledRange(3.0f));

            var count = 0;
            Structs.Custom3OpDelegate custom3 = (dst, src0, src1, src2, ith, nth, userdata) =>
            {
                AssertUserData(userData, userdata, "custom3");
                if (ith != 0)
                {
                    throw new Exception("custom3 should run on ith == 0");
                }

                TestAssertions.Equal(dst.ElementsCount, src0.ElementsCount, "custom3 shape a");
                TestAssertions.Equal(dst.ElementsCount, src1.ElementsCount, "custom3 shape b");
                TestAssertions.Equal(dst.ElementsCount, src2.ElementsCount, "custom3 shape c");
                Interlocked.Increment(ref count);

                for (int i = 0; i < dst.ElementsCount; i++)
                {
                    dst.SetData(src0.GetFloat(i) + src1.GetFloat(i) + src2.GetFloat(i), i);
                }
            };

            var output = ctx.MapCustom3(a, b, c, custom3, 1, userData);
            ExecuteGraph(ctx, output, 4);

            var actual = output.GetDataInFloats();
            var expected = new float[actual.Length];
            for (int i = 0; i < expected.Length; i++)
            {
                var x = i + 1;
                expected[i] = x + x * 2 + x * 3;
            }

            AssertFloatArray(actual, expected, "custom3");
            TestAssertions.Equal(1, count, "custom3 count");
        }

        private static void TestCustom4d(IntPtr userData)
        {
            using var ctx = CreateContext();

            var t1 = ctx.NewTensor2d(Structs.GGmlType.GGML_TYPE_F32, 10, 2);
            var t2 = ctx.NewTensor2d(Structs.GGmlType.GGML_TYPE_F32, 10, 2);
            var t3 = ctx.NewTensor2d(Structs.GGmlType.GGML_TYPE_F32, 10, 2);
            var t4 = ctx.NewTensor2d(Structs.GGmlType.GGML_TYPE_F32, 10, 2);
            var t5 = ctx.NewTensor2d(Structs.GGmlType.GGML_TYPE_F32, 10, 2);

            SetTensorData(t1, CreateScaledRange(1.0f));
            SetTensorData(t2, CreateScaledRange(2.0f));
            SetTensorData(t3, CreateScaledRange(3.0f));
            SetTensorData(t4, CreateScaledRange(1.0f));
            SetTensorData(t5, CreateScaledRange(2.0f));

            Structs.CustomOpDelegate custom = (dst, ith, nth, userdata) =>
            {
                AssertUserData(userData, userdata, "custom4d");

                var src = dst.Sources;
                for (int i = 0; i < 5; i++)
                {
                    if (src[i].NativeHandle == IntPtr.Zero)
                    {
                        throw new Exception($"custom4d src[{i}] is null");
                    }
                }

                for (int i = ith; i < dst.ElementsCount; i += nth)
                {
                    var value =
                        src[0].GetFloat(i) +
                        src[1].GetFloat(i) * src[2].GetFloat(i) -
                        src[3].GetFloat(i) * src[4].GetFloat(i);

                    dst.SetInt1D(i, (int)value);
                }
            };

            var output = ctx.Custom4d(
                Structs.GGmlType.GGML_TYPE_I32,
                10,
                2,
                1,
                1,
                new[] { t1, t2, t3, t4, t5 },
                custom,
                Structs.GGML_N_TASKS_MAX,
                userData);

            ExecuteGraph(ctx, output, 4);

            for (int i = 0; i < output.ElementsCount; i++)
            {
                var x = i + 1;
                var expected = x + 4 * x * x;
                TestAssertions.Equal(expected, output.GetInt1D(i), $"custom4d[{i}]");
            }
        }

        private static SafeGGmlContext CreateContext()
        {
            return new SafeGGmlContext(IntPtr.Zero, 1UL * 1024 * 1024, NoAllocateMemory: false);
        }

        private static void ExecuteGraph(SafeGGmlContext ctx, SafeGGmlTensor output, int threads)
        {
            var graph = ctx.NewGraph();
            graph.BuildForwardExpend(output);
            graph.ComputeWithGGmlContext(ctx, threads);
        }

        private static void SetTensorData(SafeGGmlTensor tensor, float[] values)
        {
            for (int i = 0; i < values.Length; i++)
            {
                tensor.SetData(values[i], i);
            }
        }

        private static float[] CreateScaledRange(float scale)
        {
            var values = new float[20];
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = (i + 1) * scale;
            }

            return values;
        }

        private static void AssertFloatArray(float[] actual, float[] expected, string name)
        {
            TestAssertions.Equal(expected.Length, actual.Length, $"{name} length");
            for (int i = 0; i < expected.Length; i++)
            {
                TestAssertions.Equal(expected[i], actual[i], $"{name}[{i}]");
            }
        }

        private static void AssertUserData(IntPtr expected, IntPtr actual, string name)
        {
            if (actual != expected)
            {
                throw new Exception($"{name} userdata pointer mismatch");
            }

            var text = Marshal.PtrToStringAnsi(actual);
            if (!string.Equals(text, "ggml", StringComparison.Ordinal))
            {
                throw new Exception($"{name} userdata text mismatch: {text}");
            }
        }
    }
}

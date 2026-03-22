using GGMLSharp;
using System;

namespace GGMLSharp.Tests
{
    public class GGmlConvTransposeTests
    {
        public void Test_ConvTranspose1D_Matches_C()
        {
            RunConvTranspose1DCase(1, new long[] { 4, 3, 1, 1 }, new float[]
            {
                18.0f, 45.0f, 59.0f, 37.0f,
                24.0f, 61.0f, 83.0f, 51.0f,
                30.0f, 77.0f, 107.0f, 65.0f,
            });

            RunConvTranspose1DCase(2, new long[] { 6, 3, 1, 1 }, new float[]
            {
                18.0f, 21.0f, 24.0f, 29.0f, 30.0f, 37.0f,
                24.0f, 27.0f, 34.0f, 39.0f, 44.0f, 51.0f,
                30.0f, 33.0f, 44.0f, 49.0f, 58.0f, 65.0f,
            });

            RunConvTranspose1DCase(3, new long[] { 8, 3, 1, 1 }, new float[]
            {
                18.0f, 21.0f, 0.0f, 24.0f, 29.0f, 0.0f, 30.0f, 37.0f,
                24.0f, 27.0f, 0.0f, 34.0f, 39.0f, 0.0f, 44.0f, 51.0f,
                30.0f, 33.0f, 0.0f, 44.0f, 49.0f, 0.0f, 58.0f, 65.0f,
            });
        }

        public void Test_ConvTranspose2D_Matches_C()
        {
            RunConvTranspose2DCase(1, new long[] { 4, 3, 3, 1 }, new float[]
            {
                72.0f, 162.0f, 188.0f, 106.0f,
                192.0f, 430.0f, 490.0f, 274.0f,
                132.0f, 292.0f, 326.0f, 180.0f,

                96.0f, 218.0f, 260.0f, 146.0f,
                264.0f, 590.0f, 682.0f, 378.0f,
                180.0f, 396.0f, 446.0f, 244.0f,

                120.0f, 274.0f, 332.0f, 186.0f,
                336.0f, 750.0f, 874.0f, 482.0f,
                228.0f, 500.0f, 566.0f, 308.0f,
            });

            RunConvTranspose2DCase(2, new long[] { 6, 4, 3, 1 }, new float[]
            {
                72.0f, 78.0f, 84.0f, 92.0f, 96.0f, 106.0f,
                84.0f, 90.0f, 100.0f, 108.0f, 116.0f, 126.0f,
                108.0f, 120.0f, 120.0f, 134.0f, 132.0f, 148.0f,
                132.0f, 144.0f, 148.0f, 162.0f, 164.0f, 180.0f,

                96.0f, 102.0f, 116.0f, 124.0f, 136.0f, 146.0f,
                108.0f, 114.0f, 132.0f, 140.0f, 156.0f, 166.0f,
                156.0f, 168.0f, 176.0f, 190.0f, 196.0f, 212.0f,
                180.0f, 192.0f, 204.0f, 218.0f, 228.0f, 244.0f,

                120.0f, 126.0f, 148.0f, 156.0f, 176.0f, 186.0f,
                132.0f, 138.0f, 164.0f, 172.0f, 196.0f, 206.0f,
                204.0f, 216.0f, 232.0f, 246.0f, 260.0f, 276.0f,
                228.0f, 240.0f, 260.0f, 274.0f, 292.0f, 308.0f,
            });

            RunConvTranspose2DCase(3, new long[] { 8, 5, 3, 1 }, new float[]
            {
                72.0f, 78.0f, 0.0f, 84.0f, 92.0f, 0.0f, 96.0f, 106.0f,
                84.0f, 90.0f, 0.0f, 100.0f, 108.0f, 0.0f, 116.0f, 126.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                108.0f, 120.0f, 0.0f, 120.0f, 134.0f, 0.0f, 132.0f, 148.0f,
                132.0f, 144.0f, 0.0f, 148.0f, 162.0f, 0.0f, 164.0f, 180.0f,

                96.0f, 102.0f, 0.0f, 116.0f, 124.0f, 0.0f, 136.0f, 146.0f,
                108.0f, 114.0f, 0.0f, 132.0f, 140.0f, 0.0f, 156.0f, 166.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                156.0f, 168.0f, 0.0f, 176.0f, 190.0f, 0.0f, 196.0f, 212.0f,
                180.0f, 192.0f, 0.0f, 204.0f, 218.0f, 0.0f, 228.0f, 244.0f,

                120.0f, 126.0f, 0.0f, 148.0f, 156.0f, 0.0f, 176.0f, 186.0f,
                132.0f, 138.0f, 0.0f, 164.0f, 172.0f, 0.0f, 196.0f, 206.0f,
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                204.0f, 216.0f, 0.0f, 232.0f, 246.0f, 0.0f, 260.0f, 276.0f,
                228.0f, 240.0f, 0.0f, 260.0f, 274.0f, 0.0f, 292.0f, 308.0f,
            });
        }

        private static void RunConvTranspose1DCase(int stride, long[] expectedShape, float[] expected)
        {
            using var ctx = CreateContext();
            using var backend = SafeGGmlBackend.CpuInit();

            var input = ctx.NewTensor2d(Structs.GGmlType.GGML_TYPE_F32, 3, 2);
            var kernel = ctx.NewTensor3d(Structs.GGmlType.GGML_TYPE_F16, 2, 3, 2);
            var output = ctx.ConvTranspose1D(kernel, input, stride, 0, 1);

            var graph = PrepareGraph(ctx, backend, output);
            input.SetData(CreateRange(6));
            kernel.SetData(TestAssertions.ToFp16Bytes(CreateRange(12)));
            graph.BackendCompute(backend);
            AssertTensor(output, expectedShape, expected);
        }

        private static void RunConvTranspose2DCase(int stride, long[] expectedShape, float[] expected)
        {
            using var ctx = CreateContext();
            using var backend = SafeGGmlBackend.CpuInit();

            var input = ctx.NewTensor4d(Structs.GGmlType.GGML_TYPE_F32, 3, 2, 2, 1);
            var kernel = ctx.NewTensor4d(Structs.GGmlType.GGML_TYPE_F16, 2, 2, 3, 2);
            var output = ctx.ConvTranspose2D(kernel, input, stride);

            var graph = PrepareGraph(ctx, backend, output);
            input.SetData(CreateRange(12));
            kernel.SetData(TestAssertions.ToFp16Bytes(CreateRange(24)));
            graph.BackendCompute(backend);
            AssertTensor(output, expectedShape, expected);
        }

        private static SafeGGmlContext CreateContext()
        {
            return new SafeGGmlContext(IntPtr.Zero, 2UL * 1024 * 1024, NoAllocateMemory: true);
        }

        private static SafeGGmlGraph PrepareGraph(SafeGGmlContext ctx, SafeGGmlBackend backend, SafeGGmlTensor tensor)
        {
            var graph = ctx.NewGraph();
            graph.BuildForwardExpend(tensor);
            ctx.BackendAllocContextTensors(backend);
            backend.SetCpuThreads(1);
            return graph;
        }

        private static float[] CreateRange(int count)
        {
            var values = new float[count];
            for (int i = 0; i < count; i++)
            {
                values[i] = i;
            }

            return values;
        }

        private static void AssertTensor(SafeGGmlTensor tensor, long[] shape, float[] expected)
        {
            TestAssertions.Shape(tensor, shape);
            TestAssertions.FloatDataExact(tensor, expected);
        }

    }
}

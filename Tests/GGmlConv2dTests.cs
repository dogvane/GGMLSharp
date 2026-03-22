using GGMLSharp;
using System;

namespace GGMLSharp.Tests
{
    public sealed class GGmlConv2dTests
    {
        public void Test_Conv2d_BackendGraph_Matches_UpstreamCpp()
        {
            SafeGGmlBackend? backend = null;
            SafeGGmlContext? modelContext = null;
            SafeGGmlBackendBuffer? modelBuffer = null;
            SafeGGmlGraphAllocr? allocr = null;
            SafeGGmlContext? reserveGraphContext = null;
            SafeGGmlContext? computeGraphContext = null;

            try
            {
                backend = SafeGGmlBackend.CpuInit();
                modelBuffer = backend.AllocBuffer((ulong)(900 * sizeof(ushort) + 480 * sizeof(float) + 1024));
                modelContext = new SafeGGmlContext(IntPtr.Zero, Common.TensorOverheadLength * 2, NoAllocateMemory: true);

                var weights = modelContext.NewTensor4d(Structs.GGmlType.GGML_TYPE_F16, 3, 3, 10, 10);
                var input = modelContext.NewTensor4d(Structs.GGmlType.GGML_TYPE_F32, 8, 6, 10, 1);

                var tallocr = new GGmlTallocr(modelBuffer);
                tallocr.Alloc(weights);
                tallocr.Alloc(input);

                weights.SetData(TestAssertions.ToFp16Bytes(Fill(900, 2.5f)));
                input.SetData(Fill(480, 1.5f));

                allocr = new SafeGGmlGraphAllocr(backend.GetDefaultBufferType());

                reserveGraphContext = CreateGraphContext();
                var reserveGraph = BuildGraph(reserveGraphContext, weights, input);
                if (!reserveGraph.Reserve(allocr))
                {
                    throw new Exception("ggml_gallocr_reserve failed for conv2d graph.");
                }

                computeGraphContext = CreateGraphContext();
                var computeGraph = BuildGraph(computeGraphContext, weights, input);
                if (!computeGraph.GraphAllocate(allocr))
                {
                    throw new Exception("ggml_gallocr_alloc_graph failed for conv2d graph.");
                }

                backend.SetCpuThreads(1);
                computeGraph.BackendCompute(backend);

                var im2col = computeGraph.GetTensor("im2col_res");
                var conv2d = computeGraph.GetTensor("conv2d_res");

                TestAssertions.Equal(4320, im2col.ElementsCount, "im2col element count");
                TestAssertions.Shape(conv2d, 8, 6, 10, 1);

                var im2colData = ReadU16Data(im2col);
                ushort[] expectedIm2colPrefix =
                {
                    0, 0, 0, 0, 15872, 15872, 0, 15872,
                    15872, 0, 0, 0, 0, 15872, 15872, 0,
                    15872, 15872, 0, 0, 0, 0, 15872, 15872,
                    0, 15872, 15872, 0, 0, 0, 0, 15872,
                    15872, 0, 15872, 15872, 0, 0, 0, 0,
                    15872, 15872, 0, 15872, 15872, 0, 0, 0,
                    0, 15872, 15872, 0, 15872, 15872, 0, 0,
                    0, 0, 15872, 15872, 0, 15872, 15872, 0,
                    0, 0, 0, 15872, 15872, 0, 15872, 15872,
                    0, 0, 0, 0, 15872, 15872, 0, 15872,
                    15872, 0, 0, 0, 0, 15872, 15872, 0,
                    15872, 15872, 0, 0, 0, 15872, 15872, 15872,
                    15872, 15872, 15872, 0, 0, 0, 15872, 15872,
                    15872, 15872, 15872, 15872, 0, 0, 0, 15872,
                    15872, 15872, 15872, 15872, 15872, 0, 0, 0,
                    15872, 15872, 15872, 15872, 15872, 15872, 0, 0,
                    0, 15872, 15872, 15872, 15872, 15872, 15872, 0,
                    0, 0, 15872, 15872, 15872, 15872, 15872, 15872,
                    0, 0, 0, 15872, 15872, 15872, 15872, 15872,
                    15872, 0, 0, 0, 15872, 15872, 15872, 15872,
                    15872, 15872, 0, 0, 0, 15872, 15872, 15872,
                    15872, 15872, 15872, 0, 0, 0, 15872, 15872,
                    15872, 15872, 15872, 15872, 0, 0, 0, 15872,
                    15872, 15872, 15872, 15872, 15872, 0, 0, 0,
                    15872, 15872, 15872, 15872, 15872, 15872, 0, 0,
                    0, 15872, 15872, 15872, 15872, 15872, 15872, 0,
                    0, 0, 15872, 15872, 15872, 15872, 15872, 15872,
                    0, 0, 0, 15872, 15872, 15872, 15872, 15872,
                    15872, 0, 0, 0, 15872, 15872, 15872, 15872,
                    15872, 15872, 0, 0, 0, 15872, 15872, 15872,
                    15872, 15872, 15872, 0, 0, 0, 15872, 15872,
                    15872, 15872, 15872, 15872, 0, 0, 0, 15872,
                    15872, 15872, 15872, 15872, 15872, 0, 0, 0,
                    15872, 15872, 15872, 15872, 15872, 15872, 0, 0,
                    0, 15872, 15872, 15872, 15872, 15872, 15872, 0,
                    0, 0, 15872, 15872, 15872, 15872, 15872, 15872,
                    0, 0, 0, 15872, 15872, 15872, 15872, 15872,
                    15872, 0, 0, 0, 15872, 15872, 15872, 15872,
                    15872, 15872, 0, 0, 0, 15872, 15872, 15872,
                    15872, 15872, 15872, 0, 0, 0, 15872, 15872,
                    15872, 15872, 15872, 15872, 0, 0, 0, 15872,
                    15872, 15872, 15872, 15872, 15872, 0, 0, 0,
                    15872, 15872, 15872, 15872, 15872, 15872, 0, 0,
                    0, 15872, 15872, 15872, 15872, 15872, 15872, 0,
                    0, 0, 15872, 15872, 15872, 15872, 15872, 15872,
                    0, 0, 0, 15872, 15872, 15872, 15872, 15872,
                    15872, 0, 0, 0, 15872, 15872, 15872, 15872,
                    15872, 15872, 0, 0, 0, 15872, 15872, 15872,
                    15872, 15872, 15872, 0, 0, 0, 15872, 15872,
                    15872, 15872, 15872, 15872, 0, 0, 0, 15872,
                    15872, 15872, 15872, 15872, 15872, 0, 0, 0,
                    15872, 15872, 15872, 15872, 15872, 15872, 0, 0,
                    0, 15872, 15872, 15872, 15872, 15872, 15872, 0,
                    0, 0, 15872, 15872, 15872, 15872, 15872, 15872,
                    0, 0, 0, 15872, 15872, 15872, 15872, 15872,
                    15872, 0, 0, 0, 15872, 15872, 15872, 15872,
                    15872, 15872, 0, 0, 0, 15872, 15872, 15872,
                    15872, 15872, 15872, 0, 0, 0, 15872, 15872,
                    15872, 15872, 15872, 15872, 0, 0, 0, 15872,
                    15872, 15872, 15872, 15872, 15872, 0, 0, 0
                };

                for (int i = 0; i < expectedIm2colPrefix.Length; i++)
                {
                    TestAssertions.Equal(expectedIm2colPrefix[i], im2colData[i], $"im2col data[{i}]");
                }

                TestAssertions.FloatDataExact(conv2d, BuildExpectedConv2d());
            }
            finally
            {
                computeGraphContext?.Free();
                reserveGraphContext?.Free();
                allocr?.Free();
                modelBuffer?.Free();
                modelContext?.Free();
                backend?.Free();
            }
        }

        private static SafeGGmlContext CreateGraphContext()
        {
            ulong graphBytes = Common.TensorOverheadLength * (ulong)Structs.GGML_DEFAULT_GRAPH_SIZE + Common.GraphOverheadLength;
            return new SafeGGmlContext(IntPtr.Zero, graphBytes, NoAllocateMemory: true);
        }

        private static SafeGGmlGraph BuildGraph(SafeGGmlContext context, SafeGGmlTensor weights, SafeGGmlTensor input)
        {
            var graph = context.NewGraph();

            var im2col = context.Im2Col(weights, input, 1, 1, 1, 1, 1, 1, true, Structs.GGmlType.GGML_TYPE_F16);
            im2col.Name = "im2col_res";
            graph.BuildForwardExpend(im2col);

            var conv2d = context.Conv2d(weights, input, 1, 1, 1, 1, 1, 1);
            conv2d.Name = "conv2d_res";
            graph.BuildForwardExpend(conv2d);

            return graph;
        }

        private static float[] Fill(int count, float value)
        {
            var data = new float[count];
            Array.Fill(data, value);
            return data;
        }

        private static ushort[] ReadU16Data(SafeGGmlTensor tensor)
        {
            byte[] bytes = tensor.GetBackend();
            ushort[] values = new ushort[bytes.Length / sizeof(ushort)];
            Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
            return values;
        }

        private static float[] BuildExpectedConv2d()
        {
            const int width = 8;
            const int height = 6;
            const int channels = 10;

            var expected = new float[width * height * channels];
            for (int c = 0; c < channels; c++)
            {
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int validX = (x == 0 || x == width - 1) ? 2 : 3;
                        int validY = (y == 0 || y == height - 1) ? 2 : 3;
                        int validKernelPoints = validX * validY;
                        expected[x + width * (y + height * c)] = validKernelPoints * 10 * 2.5f * 1.5f;
                    }
                }
            }

            return expected;
        }
    }
}

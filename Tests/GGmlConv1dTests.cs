using GGMLSharp;
using System;

namespace GGMLSharp.Tests
{
    public sealed class GGmlConv1dTests
    {
        public void Test_Conv1d_BackendGraph_Matches_UpstreamCpp()
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
                modelBuffer = backend.AllocBuffer((ulong)(300 * sizeof(ushort) + 80 * sizeof(float) + 1024));
                modelContext = new SafeGGmlContext(IntPtr.Zero, Common.TensorOverheadLength * 2, NoAllocateMemory: true);

                var weights = modelContext.NewTensor3d(Structs.GGmlType.GGML_TYPE_F16, 3, 10, 10);
                var input = modelContext.NewTensor3d(Structs.GGmlType.GGML_TYPE_F32, 8, 10, 1);

                var tallocr = new GGmlTallocr(modelBuffer);
                tallocr.Alloc(weights);
                tallocr.Alloc(input);

                weights.SetData(TestAssertions.ToFp16Bytes(Fill(300, 4.5f)));
                input.SetData(Fill(80, 2.5f));

                allocr = new SafeGGmlGraphAllocr(backend.GetDefaultBufferType());

                reserveGraphContext = CreateGraphContext();
                var reserveGraph = BuildGraph(reserveGraphContext, weights, input);
                if (!reserveGraph.Reserve(allocr))
                {
                    throw new Exception("ggml_gallocr_reserve failed for conv1d graph.");
                }

                computeGraphContext = CreateGraphContext();
                var computeGraph = BuildGraph(computeGraphContext, weights, input);
                if (!computeGraph.GraphAllocate(allocr))
                {
                    throw new Exception("ggml_gallocr_alloc_graph failed for conv1d graph.");
                }

                backend.SetCpuThreads(1);
                computeGraph.BackendCompute(backend);

                var im2col = computeGraph.GetTensor("im2col_res");
                var conv1d = computeGraph.GetTensor("conv1d_res");

                TestAssertions.Equal(240, im2col.ElementsCount, "im2col element count");
                TestAssertions.Shape(conv1d, 8, 10, 1, 1);

                var im2colData = ReadU16Data(im2col);
                ushort[] expectedIm2col =
                {
                    0, 16640, 16640, 0, 16640, 16640, 0, 16640,
                    16640, 0, 16640, 16640, 0, 16640, 16640, 0,
                    16640, 16640, 0, 16640, 16640, 0, 16640, 16640,
                    0, 16640, 16640, 0, 16640, 16640, 16640, 16640,
                    16640, 16640, 16640, 16640, 16640, 16640, 16640, 16640,
                    16640, 16640, 16640, 16640, 16640, 16640, 16640, 16640,
                    16640, 16640, 16640, 16640, 16640, 16640, 16640, 16640,
                    16640, 16640, 16640, 16640, 16640, 16640, 16640, 16640,
                    16640, 16640, 16640, 16640, 16640, 16640, 16640, 16640,
                    16640, 16640, 16640, 16640, 16640, 16640, 16640, 16640,
                };

                for (int i = 0; i < expectedIm2col.Length; i++)
                {
                    TestAssertions.Equal(expectedIm2col[i], im2colData[i], $"im2col data[{i}]");
                }

                TestAssertions.FloatDataExact(conv1d,
                    225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
                    225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
                    225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
                    225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
                    225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
                    225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
                    225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
                    225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
                    225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
                    225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f);
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

            var im2col = context.Im2Col(weights, input, 1, 0, 1, 0, 1, 0, false, Structs.GGmlType.GGML_TYPE_F16);
            im2col.Name = "im2col_res";
            graph.BuildForwardExpend(im2col);

            var conv1d = context.Conv1D(weights, input, 1, 1, 1);
            conv1d.Name = "conv1d_res";
            graph.BuildForwardExpend(conv1d);

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
    }
}

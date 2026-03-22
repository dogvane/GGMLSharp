using GGMLSharp;
using System;

namespace GGMLSharp.Tests
{
    public sealed class GGmlConv1dDepthwiseC2Tests
    {
        public void Test_Conv1dDepthwise_C2_BackendGraph_Matches_UpstreamCpp()
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
                modelBuffer = backend.AllocBuffer((ulong)(6 * sizeof(ushort) + 12 * sizeof(float) + 1024));
                modelContext = new SafeGGmlContext(IntPtr.Zero, Common.TensorOverheadLength * 2, NoAllocateMemory: true);

                var weight = modelContext.NewTensor3d(Structs.GGmlType.GGML_TYPE_F16, 3, 1, 2);
                var input = modelContext.NewTensor3d(Structs.GGmlType.GGML_TYPE_F32, 6, 2, 1);

                var tallocr = new GGmlTallocr(modelBuffer);
                tallocr.Alloc(weight);
                tallocr.Alloc(input);

                weight.SetData(TestAssertions.ToFp16Bytes(new float[] { 10.0f, 20.0f, 30.0f, 0.1f, 0.2f, 0.3f }));
                input.SetData(new float[]
                {
                    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                });

                allocr = new SafeGGmlGraphAllocr(backend.GetDefaultBufferType());

                reserveGraphContext = CreateGraphContext();
                var reserveGraph = BuildGraph(reserveGraphContext, weight, input);
                if (!reserveGraph.Reserve(allocr))
                {
                    throw new Exception("ggml_gallocr_reserve failed for conv1d-dw-c2 graph.");
                }

                computeGraphContext = CreateGraphContext();
                var computeGraph = BuildGraph(computeGraphContext, weight, input);
                if (!computeGraph.GraphAllocate(allocr))
                {
                    throw new Exception("ggml_gallocr_alloc_graph failed for conv1d-dw-c2 graph.");
                }

                backend.SetCpuThreads(1);
                computeGraph.BackendCompute(backend);

                var output = computeGraph.GetTensor("conv1d_dw_res");
                TestAssertions.Shape(output, 2, 2, 1, 1);
                TestAssertions.FloatDataNear(output, 1e-4f, 60.0f, 60.0f, 0.6f, 0.6f);
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

        private static SafeGGmlGraph BuildGraph(SafeGGmlContext context, SafeGGmlTensor weight, SafeGGmlTensor input)
        {
            var graph = context.NewGraph();
            var output = context.ConvDepthwise1D(weight, input, 3, 0, 1);
            output.Name = "conv1d_dw_res";
            graph.BuildForwardExpend(output);
            return graph;
        }
    }
}

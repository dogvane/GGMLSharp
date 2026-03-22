using GGMLSharp;
using System;

namespace GGMLSharp.Tests
{
    public sealed class GGmlConvTranspose1DTests
    {
        public void Test_ConvTranspose1D_BackendGraph_Matches_UpstreamCpp()
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
                modelBuffer = backend.AllocBuffer(CalculateModelBufferSizeBytes());
                modelContext = new SafeGGmlContext(IntPtr.Zero, Common.TensorOverheadLength * 10, NoAllocateMemory: true);

                var model = LoadModel(modelContext, modelBuffer);

                allocr = new SafeGGmlGraphAllocr(backend.GetDefaultBufferType());

                reserveGraphContext = CreateGraphContext();
                var reserveGraph = BuildGraph(reserveGraphContext, model);
                if (!reserveGraph.Reserve(allocr))
                {
                    throw new Exception("ggml_gallocr_reserve failed for conv-transpose-1d graph.");
                }

                computeGraphContext = CreateGraphContext();
                var computeGraph = BuildGraph(computeGraphContext, model);
                if (!computeGraph.GraphAllocate(allocr))
                {
                    throw new Exception("ggml_gallocr_alloc_graph failed for conv-transpose-1d graph.");
                }

                backend.SetCpuThreads(1);
                computeGraph.BackendCompute(backend);

                AssertTensor(computeGraph, "conv1d_transpose_res_0", new long[] { 4, 1, 1, 1 }, new float[] { 1, 4, 7, 6 });
                AssertTensor(computeGraph, "conv1d_transpose_res_1", new long[] { 5, 1, 1, 1 }, new float[] { 5, 18, 26, 18, 5 });
                AssertTensor(computeGraph, "conv1d_transpose_res_2", new long[] { 5, 2, 1, 1 }, new float[]
                {
                    7, 18, 22, 18, 7,
                    5, 18, 26, 18, 5,
                });
                AssertTensor(computeGraph, "conv1d_transpose_res_3", new long[] { 7, 2, 1, 1 }, new float[]
                {
                    7, 6, 17, 12, 17, 6, 7,
                    5, 6, 19, 12, 19, 6, 5,
                });
                AssertTensor(computeGraph, "conv1d_transpose_res_4", new long[] { 4, 3, 1, 1 }, new float[]
                {
                    18, 45, 59, 37,
                    24, 61, 83, 51,
                    30, 77, 107, 65,
                });
                AssertTensor(computeGraph, "conv1d_transpose_res_5", new long[] { 6, 3, 1, 1 }, new float[]
                {
                    18, 21, 24, 29, 30, 37,
                    24, 27, 34, 39, 44, 51,
                    30, 33, 44, 49, 58, 65,
                });
                AssertTensor(computeGraph, "conv1d_transpose_res_6", new long[] { 8, 3, 1, 1 }, new float[]
                {
                    18, 21, 0, 24, 29, 0, 30, 37,
                    24, 27, 0, 34, 39, 0, 44, 51,
                    30, 33, 0, 44, 49, 0, 58, 65,
                });

                var largeOutput = computeGraph.GetTensor("conv1d_transpose_res_7");
                TestAssertions.Shape(largeOutput, 1584, 32, 1, 1);

                var expectedLarge = ReferenceConvTranspose1D(
                    model.A4Data,
                    kernelWidth: 16,
                    outputChannels: 32,
                    inputChannels: 32,
                    input: model.B4Data,
                    inputWidth: 197,
                    stride: 8);

                TestAssertions.FloatDataRelativeNear(largeOutput, 1e-6f, expectedLarge);
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

        private static ModelState LoadModel(SafeGGmlContext context, SafeGGmlBackendBuffer buffer)
        {
            var model = new ModelState();
            var tallocr = new GGmlTallocr(buffer);

            model.A0Data = new float[] { 1, 2, 3 };
            model.B0Data = new float[] { 1, 2 };

            model.A1Data = new float[] { 1, 2, 3, 3, 2, 1 };
            model.B1Data = new float[] { 2, 3, 1, 1, 3, 2 };

            model.A2Data = new float[] { 3, 2, 1, 1, 2, 3, 1, 2, 3, 3, 2, 1 };
            model.B2Data = new float[] { 2, 3, 1, 1, 3, 2 };

            var sharedData = CreateModuloSequence(16 * 32 * 32, 1024);
            model.A3Data = Slice(sharedData, 2 * 3 * 2);
            model.B3Data = Slice(sharedData, 3 * 2);
            model.A4Data = sharedData;
            model.B4Data = Slice(sharedData, 197 * 32);

            model.A0 = context.NewTensor1d(Structs.GGmlType.GGML_TYPE_F32, 3);
            model.B0 = context.NewTensor1d(Structs.GGmlType.GGML_TYPE_F32, 2);
            model.A1 = context.NewTensor3d(Structs.GGmlType.GGML_TYPE_F32, 3, 1, 2);
            model.B1 = context.NewTensor2d(Structs.GGmlType.GGML_TYPE_F32, 3, 2);
            model.A2 = context.NewTensor3d(Structs.GGmlType.GGML_TYPE_F32, 3, 2, 2);
            model.B2 = context.NewTensor2d(Structs.GGmlType.GGML_TYPE_F32, 3, 2);
            model.A3 = context.NewTensor3d(Structs.GGmlType.GGML_TYPE_F32, 2, 3, 2);
            model.B3 = context.NewTensor2d(Structs.GGmlType.GGML_TYPE_F32, 3, 2);
            model.A4 = context.NewTensor3d(Structs.GGmlType.GGML_TYPE_F32, 16, 32, 32);
            model.B4 = context.NewTensor2d(Structs.GGmlType.GGML_TYPE_F32, 197, 32);

            tallocr.Alloc(model.A0);
            tallocr.Alloc(model.B0);
            tallocr.Alloc(model.A1);
            tallocr.Alloc(model.B1);
            tallocr.Alloc(model.A2);
            tallocr.Alloc(model.B2);
            tallocr.Alloc(model.A3);
            tallocr.Alloc(model.B3);
            tallocr.Alloc(model.A4);
            tallocr.Alloc(model.B4);

            model.A0.SetData(model.A0Data);
            model.B0.SetData(model.B0Data);
            model.A1.SetData(model.A1Data);
            model.B1.SetData(model.B1Data);
            model.A2.SetData(model.A2Data);
            model.B2.SetData(model.B2Data);
            model.A3.SetData(model.A3Data);
            model.B3.SetData(model.B3Data);
            model.A4.SetData(model.A4Data);
            model.B4.SetData(model.B4Data);

            return model;
        }

        private static SafeGGmlContext CreateGraphContext()
        {
            ulong graphBytes = Common.TensorOverheadLength * (ulong)Structs.GGML_DEFAULT_GRAPH_SIZE + Common.GraphOverheadLength;
            return new SafeGGmlContext(IntPtr.Zero, graphBytes, NoAllocateMemory: true);
        }

        private static SafeGGmlGraph BuildGraph(SafeGGmlContext context, ModelState model)
        {
            var graph = context.NewGraph();

            AddOutput(graph, context.ConvTranspose1D(model.A0, model.B0, 1, 0, 1), "conv1d_transpose_res_0");
            AddOutput(graph, context.ConvTranspose1D(model.A1, model.B1, 1, 0, 1), "conv1d_transpose_res_1");
            AddOutput(graph, context.ConvTranspose1D(model.A2, model.B2, 1, 0, 1), "conv1d_transpose_res_2");
            AddOutput(graph, context.ConvTranspose1D(model.A2, model.B2, 2, 0, 1), "conv1d_transpose_res_3");
            AddOutput(graph, context.ConvTranspose1D(model.A3, model.B3, 1, 0, 1), "conv1d_transpose_res_4");
            AddOutput(graph, context.ConvTranspose1D(model.A3, model.B3, 2, 0, 1), "conv1d_transpose_res_5");
            AddOutput(graph, context.ConvTranspose1D(model.A3, model.B3, 3, 0, 1), "conv1d_transpose_res_6");
            AddOutput(graph, context.ConvTranspose1D(model.A4, model.B4, 8, 0, 1), "conv1d_transpose_res_7");

            return graph;
        }

        private static void AddOutput(SafeGGmlGraph graph, SafeGGmlTensor tensor, string name)
        {
            tensor.Name = name;
            graph.BuildForwardExpend(tensor);
        }

        private static void AssertTensor(SafeGGmlGraph graph, string tensorName, long[] shape, float[] expected)
        {
            var tensor = graph.GetTensor(tensorName);
            TestAssertions.Shape(tensor, shape);
            TestAssertions.FloatDataExact(tensor, expected);
        }

        private static ulong CalculateModelBufferSizeBytes()
        {
            const ulong elementSize = sizeof(float);

            return elementSize * (ulong)(
                3 +
                2 +
                6 +
                6 +
                12 +
                6 +
                (2 * 3 * 2) +
                (3 * 2) +
                (16 * 32 * 32) +
                (197 * 32)) + 1024;
        }

        private static float[] CreateModuloSequence(int count, int modulo)
        {
            var data = new float[count];
            for (int i = 0; i < count; i++)
            {
                data[i] = i % modulo;
            }

            return data;
        }

        private static float[] Slice(float[] data, int length)
        {
            var slice = new float[length];
            Array.Copy(data, slice, length);
            return slice;
        }

        private static float[] ReferenceConvTranspose1D(
            float[] kernel,
            int kernelWidth,
            int outputChannels,
            int inputChannels,
            float[] input,
            int inputWidth,
            int stride)
        {
            int outputWidth = (inputWidth - 1) * stride + kernelWidth;
            var output = new float[outputWidth * outputChannels];

            for (int inputChannel = 0; inputChannel < inputChannels; inputChannel++)
            {
                for (int inputIndex = 0; inputIndex < inputWidth; inputIndex++)
                {
                    float inputValue = input[inputIndex + inputWidth * inputChannel];
                    int outputStart = inputIndex * stride;

                    for (int outputChannel = 0; outputChannel < outputChannels; outputChannel++)
                    {
                        int kernelBase = kernelWidth * (outputChannel + outputChannels * inputChannel);
                        int outputBase = outputWidth * outputChannel + outputStart;

                        for (int kernelIndex = 0; kernelIndex < kernelWidth; kernelIndex++)
                        {
                            output[outputBase + kernelIndex] += kernel[kernelBase + kernelIndex] * inputValue;
                        }
                    }
                }
            }

            return output;
        }

        private sealed class ModelState
        {
            public SafeGGmlTensor A0 = null!;
            public SafeGGmlTensor B0 = null!;
            public SafeGGmlTensor A1 = null!;
            public SafeGGmlTensor B1 = null!;
            public SafeGGmlTensor A2 = null!;
            public SafeGGmlTensor B2 = null!;
            public SafeGGmlTensor A3 = null!;
            public SafeGGmlTensor B3 = null!;
            public SafeGGmlTensor A4 = null!;
            public SafeGGmlTensor B4 = null!;

            public float[] A0Data = Array.Empty<float>();
            public float[] B0Data = Array.Empty<float>();
            public float[] A1Data = Array.Empty<float>();
            public float[] B1Data = Array.Empty<float>();
            public float[] A2Data = Array.Empty<float>();
            public float[] B2Data = Array.Empty<float>();
            public float[] A3Data = Array.Empty<float>();
            public float[] B3Data = Array.Empty<float>();
            public float[] A4Data = Array.Empty<float>();
            public float[] B4Data = Array.Empty<float>();
        }
    }
}

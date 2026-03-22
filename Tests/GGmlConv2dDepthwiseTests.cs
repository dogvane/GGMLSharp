using GGMLSharp;
using System;

namespace GGMLSharp.Tests
{
    public sealed class GGmlConv2dDepthwiseTests
    {
        public void Test_Conv2dDepthwise_BackendGraph_Matches_UpstreamCpp()
        {
            RunCase(channels: 3, kernelSize: 1, stride: 1, pad: 0, dilation: 1, contiguousChannels: false);
            RunCase(channels: 3, kernelSize: 1, stride: 1, pad: 0, dilation: 1, contiguousChannels: true);
            RunCase(channels: 42, kernelSize: 3, stride: 2, pad: 1, dilation: 1, contiguousChannels: false);
            RunCase(channels: 42, kernelSize: 3, stride: 2, pad: 1, dilation: 1, contiguousChannels: true);
            RunCase(channels: 8, kernelSize: 5, stride: 1, pad: 2, dilation: 2, contiguousChannels: false);
            RunCase(channels: 8, kernelSize: 5, stride: 1, pad: 2, dilation: 2, contiguousChannels: true);
        }

        private static void RunCase(int channels, int kernelSize, int stride, int pad, int dilation, bool contiguousChannels)
        {
            const int batch = 2;
            const int srcW = 8;
            const int srcH = 6;

            using var context = new SafeGGmlContext(
                IntPtr.Zero,
                64UL * Common.TensorOverheadLength + Common.GraphOverheadLength,
                NoAllocateMemory: true);
            using var backend = SafeGGmlBackend.CpuInit();

            var graph = context.NewGraph();

            var srcInput = context.NewTensor4d(Structs.GGmlType.GGML_TYPE_F32, srcW, srcH, channels, batch);
            var knlInput = context.NewTensor4d(Structs.GGmlType.GGML_TYPE_F32, kernelSize, kernelSize, 1, channels);

            SafeGGmlTensor src = srcInput;
            SafeGGmlTensor knl = knlInput;

            if (contiguousChannels)
            {
                src = context.Cont(context.Permute(src, 1, 2, 0, 3));
                src = context.Permute(src, 2, 0, 1, 3);

                knl = context.Cont(context.Permute(knl, 2, 3, 1, 0));
                knl = context.Permute(knl, 3, 2, 0, 1);
            }

            var result = context.ConvDepthwise2D(knl, src, stride, stride, pad, pad, dilation, dilation);
            if (contiguousChannels)
            {
                result = context.Cont(result);
            }

            result.Name = "conv2d_dw_res";
            graph.BuildForwardExpend(result);

            using var buffer = context.BackendAllocContextTensors(backend);
            backend.SetCpuThreads(2);

            var srcValues = CreateRange(srcInput.ElementsCount, -1.0f, 1.0f);
            var knlValues = CreateRange(knlInput.ElementsCount, -1.0f, 1.0f);

            srcInput.SetData(srcValues);
            knlInput.SetData(knlValues);

            graph.BackendCompute(backend);

            var expected = Conv2dDepthwiseReference(
                srcW,
                srcH,
                srcValues,
                kernelSize,
                kernelSize,
                knlValues,
                channels,
                batch,
                stride,
                pad,
                dilation);

            TestAssertions.Shape(
                result,
                OutputSize(srcW, kernelSize, stride, pad, dilation),
                OutputSize(srcH, kernelSize, stride, pad, dilation),
                channels,
                batch);
            TestAssertions.FloatDataNear(result, 1e-5f, expected);
        }

        private static float[] CreateRange(long count, float start, float end)
        {
            var values = new float[count];
            float step = (end - start) / count;
            for (int i = 0; i < count; i++)
            {
                values[i] = start + i * step;
            }

            return values;
        }

        private static int OutputSize(int input, int kernel, int stride, int pad, int dilation)
        {
            return (input + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
        }

        private static float[] Conv2dDepthwiseReference(
            int srcW,
            int srcH,
            float[] srcData,
            int knlW,
            int knlH,
            float[] knlData,
            int channels,
            int batch,
            int stride,
            int pad,
            int dilation)
        {
            int dstW = OutputSize(srcW, knlW, stride, pad, dilation);
            int dstH = OutputSize(srcH, knlH, stride, pad, dilation);
            var dstData = new float[dstW * dstH * channels * batch];

            for (int b = 0; b < batch; b++)
            {
                int srcBatchBase = b * srcW * srcH * channels;
                int dstBatchBase = b * dstW * dstH * channels;

                for (int c = 0; c < channels; c++)
                {
                    int srcChannelBase = srcBatchBase + c * srcW * srcH;
                    int dstChannelBase = dstBatchBase + c * dstW * dstH;
                    int knlChannelBase = c * knlW * knlH;

                    for (int y = 0; y < dstH; y++)
                    {
                        for (int x = 0; x < dstW; x++)
                        {
                            float sum = 0;
                            for (int ky = 0; ky < knlH; ky++)
                            {
                                for (int kx = 0; kx < knlW; kx++)
                                {
                                    int srcX = x * stride + kx * dilation - pad;
                                    int srcY = y * stride + ky * dilation - pad;
                                    if (srcX >= 0 && srcX < srcW && srcY >= 0 && srcY < srcH)
                                    {
                                        sum += srcData[srcChannelBase + srcY * srcW + srcX] *
                                               knlData[knlChannelBase + ky * knlW + kx];
                                    }
                                }
                            }

                            dstData[dstChannelBase + y * dstW + x] = sum;
                        }
                    }
                }
            }

            return dstData;
        }
    }
}

using GGMLSharp;
using System;

namespace GGMLSharp.Tests
{
    public class GGmlTimestepEmbeddingTests : TestBase
    {
        public void Test_TimestepEmbedding_Matches_C()
        {
            const int dim = 15;
            const int maxPeriod = 10000;

            var timesteps = Context.NewTensor1d(Structs.GGmlType.GGML_TYPE_F32, 2);
            var output = Context.TimeStepEmbedding(timesteps, dim, maxPeriod);

            var graph = Context.NewGraph();
            graph.BuildForwardExpend(output);

            Context.BackendAllocContextTensors(Backend);
            timesteps.SetData(new float[] { 12.0f, 24.0f });
            graph.BackendCompute(Backend);

            TestAssertions.Shape(output, 15, 2, 1, 1);
            TestAssertions.FloatDataNear(output, 1e-5f, ComputeExpected(new float[] { 12.0f, 24.0f }, dim, maxPeriod));
        }

        private static float[] ComputeExpected(float[] timesteps, int dim, int maxPeriod)
        {
            var expected = new float[dim * timesteps.Length];
            var half = dim / 2;
            var freqs = new float[half];

            for (int i = 0; i < half; i++)
            {
                freqs[i] = (float)Math.Exp(-Math.Log(maxPeriod) * i / half);
            }

            for (int i = 0; i < timesteps.Length; i++)
            {
                for (int j = 0; j < half; j++)
                {
                    var arg = timesteps[i] * freqs[j];
                    expected[i * dim + j] = (float)Math.Cos(arg);
                    expected[i * dim + j + half] = (float)Math.Sin(arg);
                }

                if ((dim & 1) != 0)
                {
                    expected[i * dim + dim - 1] = 0.0f;
                }
            }

            return expected;
        }
    }
}

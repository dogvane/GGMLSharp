using GGMLSharp;

namespace GGMLSharp.Tests
{
    public class GGmlInterpolateTests : TestBase
    {
        private const uint AlignCorners = 1u << 8;

        public void Test_Interpolate_Matches_C()
        {
            AssertInterpolateCase(
                new long[] { 2, 2, 1, 1 },
                new float[]
                {
                    0.0f, 1.0f,
                    2.0f, 4.0f,
                },
                4, 4, 1, 1,
                (uint)Structs.GGmlScaleMode.GGML_SCALE_MODE_NEAREST,
                new float[]
                {
                    0.0f, 0.0f, 1.0f, 1.0f,
                    0.0f, 0.0f, 1.0f, 1.0f,
                    2.0f, 2.0f, 4.0f, 4.0f,
                    2.0f, 2.0f, 4.0f, 4.0f,
                });

            AssertInterpolateCase(
                new long[] { 2, 2, 1, 1 },
                new float[]
                {
                    0.0f, 1.0f,
                    2.0f, 4.0f,
                },
                4, 4, 1, 1,
                (uint)Structs.GGmlScaleMode.GGML_SCALE_MODE_BILINEAR,
                new float[]
                {
                    0.0f, 0.2500f, 0.7500f, 1.00f,
                    0.5f, 0.8125f, 1.4375f, 1.75f,
                    1.5f, 1.9375f, 2.8125f, 3.25f,
                    2.0f, 2.5000f, 3.5000f, 4.00f,
                });

            AssertInterpolateCase(
                new long[] { 2, 2, 1, 1 },
                new float[]
                {
                    0.0f, 1.0f,
                    2.0f, 4.0f,
                },
                4, 4, 1, 1,
                (uint)Structs.GGmlScaleMode.GGML_SCALE_MODE_BILINEAR | AlignCorners,
                new float[]
                {
                    0.0000f, 0.3333f, 0.6667f, 1.0000f,
                    0.6667f, 1.1111f, 1.5556f, 2.0000f,
                    1.3333f, 1.8889f, 2.4444f, 3.0000f,
                    2.0000f, 2.6667f, 3.3333f, 4.0000f,
                });

            AssertInterpolateCase(
                new long[] { 2, 2, 1, 1 },
                new float[]
                {
                    0.0f, 1.0f,
                    2.0f, 4.0f,
                },
                2, 3, 1, 1,
                (uint)Structs.GGmlScaleMode.GGML_SCALE_MODE_BILINEAR | AlignCorners,
                new float[]
                {
                    0.0f, 1.0f,
                    1.0f, 2.5f,
                    2.0f, 4.0f,
                });

            AssertInterpolateCase(
                new long[] { 4, 3, 2, 1 },
                new float[]
                {
                    0.0f, -1.0f, -2.0f, 0.0f,
                    1.0f, 2.0f , 4.0f , 4.0f,
                    2.0f, 2.0f , 1.0f , 1.0f,

                    1.0f, 2.0f , 3.0f , 4.0f,
                    2.0f, 2.0f , 2.0f , 2.0f,
                    -2.0f, 2.0f, -4.0f, 4.0f,
                },
                2, 1, 2, 1,
                (uint)Structs.GGmlScaleMode.GGML_SCALE_MODE_NEAREST,
                new float[]
                {
                    0.0f, -2.0f,
                    1.0f, 3.0f,
                });

            AssertInterpolateCase(
                new long[] { 4, 3, 2, 1 },
                new float[]
                {
                    0.0f, -1.0f, -2.0f, 0.0f,
                    1.0f, 2.0f , 4.0f , 4.0f,
                    2.0f, 2.0f , 1.0f , 1.0f,

                    1.0f, 2.0f , 3.0f , 4.0f,
                    2.0f, 2.0f , 2.0f , 2.0f,
                    -2.0f, 2.0f, -4.0f, 4.0f,
                },
                3, 2, 2, 1,
                (uint)Structs.GGmlScaleMode.GGML_SCALE_MODE_BILINEAR,
                new float[]
                {
                    0.1667f, -0.3750f,  0.7500f,
                    1.7917f,  1.8750f,  1.7500f,

                    1.3750f,  2.3750f,  3.3750f,
                   -0.5000f, -0.2500f,  2.5000f,
                });

            AssertInterpolateCase(
                new long[] { 4, 3, 2, 1 },
                new float[]
                {
                    0.0f, -1.0f, -2.0f, 0.0f,
                    1.0f, 2.0f , 4.0f , 4.0f,
                    2.0f, 2.0f , 1.0f , 1.0f,

                    1.0f, 2.0f , 3.0f , 4.0f,
                    2.0f, 2.0f , 2.0f , 2.0f,
                    -2.0f, 2.0f, -4.0f, 4.0f,
                },
                3, 2, 2, 1,
                (uint)Structs.GGmlScaleMode.GGML_SCALE_MODE_BILINEAR | AlignCorners,
                new float[]
                {
                    0.0f , -1.5f, 0.0f,
                    2.0f ,  1.5f, 1.0f,

                    1.0f ,  2.5f, 4.0f,
                    -2.0f, -1.0f, 4.0f,
                });
        }

        private void AssertInterpolateCase(long[] srcShape, float[] srcData, long ne0, long ne1, long ne2, long ne3, uint mode, float[] expected)
        {
            var input = Context.NewTensor(Structs.GGmlType.GGML_TYPE_F32, srcShape);
            var output = Context.Interpolate(input, ne0, ne1, ne2, ne3, mode);

            var graph = Context.NewGraph();
            graph.BuildForwardExpend(output);

            Context.BackendAllocContextTensors(Backend);
            input.SetData(srcData);
            graph.BackendCompute(Backend);

            TestAssertions.Shape(output, ne0, ne1, ne2, ne3);
            TestAssertions.FloatDataNear(output, 1e-4f, expected);
        }
    }
}

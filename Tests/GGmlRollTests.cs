using GGMLSharp;

namespace GGMLSharp.Tests
{
    public class GGmlRollTests : TestBase
    {
        public void Test_Roll_Matches_C()
        {
            AssertRollCase(new long[] { 3, 7, 4, 2 }, new int[] { 1, 0, -1, 0 }, permute: false);
            AssertRollCase(new long[] { 37, 42, 59, 2 }, new int[] { -4, 3, -7, 1 }, permute: false);
            AssertRollCase(new long[] { 37, 42, 59, 2 }, new int[] { -4, 3, -7, 1 }, permute: true);
        }

        private void AssertRollCase(long[] shape, int[] shift, bool permute)
        {
            var source = Context.NewTensor(Structs.GGmlType.GGML_TYPE_F32, shape);

            SafeGGmlTensor result;
            if (!permute)
            {
                result = Context.Roll(source, shift[0], shift[1], shift[2], shift[3]);
            }
            else
            {
                var permuted = Context.Permute(source, 0, 3, 1, 2);
                var rolled = Context.Roll(permuted, shift[0], shift[2], shift[3], shift[1]);
                result = Context.Cont(Context.Permute(rolled, 0, 2, 3, 1));
            }

            var graph = Context.NewGraph();
            graph.BuildForwardExpend(result);

            Context.BackendAllocContextTensors(Backend);

            var sourceValues = CreateRange((int)source.ElementsCount);
            source.SetData(sourceValues);

            graph.BackendCompute(Backend);

            var actual = TestAssertions.ReadFloatData(result);
            var expected = RollReference(sourceValues, shape, shift);

            TestAssertions.Equal(expected.Length, actual.Length, "roll length");
            for (int i = 0; i < expected.Length; i++)
            {
                TestAssertions.Equal(expected[i], actual[i], $"roll[{i}]");
            }
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

        private static float[] RollReference(float[] source, long[] shape, int[] shift)
        {
            var ne0 = shape[0];
            var ne1 = shape[1];
            var ne2 = shape[2];
            var ne3 = shape[3];

            var result = new float[source.Length];

            for (long i3 = 0; i3 < ne3; i3++)
            {
                for (long i2 = 0; i2 < ne2; i2++)
                {
                    for (long i1 = 0; i1 < ne1; i1++)
                    {
                        for (long i0 = 0; i0 < ne0; i0++)
                        {
                            var srcI3 = Wrap(i3 - shift[3], ne3);
                            var srcI2 = Wrap(i2 - shift[2], ne2);
                            var srcI1 = Wrap(i1 - shift[1], ne1);
                            var srcI0 = Wrap(i0 - shift[0], ne0);

                            result[FlatIndex(i0, i1, i2, i3, ne0, ne1, ne2)] =
                                source[FlatIndex(srcI0, srcI1, srcI2, srcI3, ne0, ne1, ne2)];
                        }
                    }
                }
            }

            return result;
        }

        private static int FlatIndex(long i0, long i1, long i2, long i3, long ne0, long ne1, long ne2)
        {
            return checked((int)(i3 * (ne2 * ne1 * ne0) + i2 * (ne1 * ne0) + i1 * ne0 + i0));
        }

        private static long Wrap(long index, long length)
        {
            if (index < 0)
            {
                return index + length;
            }

            if (index >= length)
            {
                return index - length;
            }

            return index;
        }
    }
}

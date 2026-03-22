using GGMLSharp;

namespace GGMLSharp.Tests
{
    public class GGmlPadReflect1dTests : TestBase
    {
        public void Test_PadReflect1D_Vector_Cases_Match_C()
        {
            var input = Context.NewTensor1d(Structs.GGmlType.GGML_TYPE_F32, 4);
            var out1 = Context.PadReflect1D(input, 1, 1);
            var out2 = Context.PadReflect1D(input, 2, 1);
            var out3 = Context.PadReflect1D(input, 1, 2);

            ExecuteGraph(new[] { out1, out2, out3 }, () => input.SetData(new[] { 1.0f, 2.0f, 3.0f, 4.0f }));

            TestAssertions.Shape(out1, 6, 1, 1, 1);
            TestAssertions.Shape(out2, 7, 1, 1, 1);
            TestAssertions.Shape(out3, 7, 1, 1, 1);

            TestAssertions.FloatDataExact(out1, 2, 1, 2, 3, 4, 3);
            TestAssertions.FloatDataExact(out2, 3, 2, 1, 2, 3, 4, 3);
            TestAssertions.FloatDataExact(out3, 2, 1, 2, 3, 4, 3, 2);
        }

        public void Test_PadReflect1D_Matrix_Case_Matches_C()
        {
            var input = Context.NewTensor2d(Structs.GGmlType.GGML_TYPE_F32, 5, 4);
            var output = Context.PadReflect1D(input, 3, 2);

            ExecuteGraph(new[] { output }, () => input.SetData(new[]
            {
                1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
                16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
            }));

            TestAssertions.Shape(output, 10, 4, 1, 1);
            TestAssertions.FloatDataExact(output,
                4, 3, 2, 1, 2, 3, 4, 5, 4, 3,
                9, 8, 7, 6, 7, 8, 9, 10, 9, 8,
                14, 13, 12, 11, 12, 13, 14, 15, 14, 13,
                19, 18, 17, 16, 17, 18, 19, 20, 19, 18);
        }

        private void ExecuteGraph(SafeGGmlTensor[] results, System.Action initializeInputs)
        {
            var graph = Context.NewGraph();
            foreach (var result in results)
            {
                graph.BuildForwardExpend(result);
            }

            Context.BackendAllocContextTensors(Backend);
            initializeInputs();
            graph.BackendCompute(Backend);
        }
    }
}

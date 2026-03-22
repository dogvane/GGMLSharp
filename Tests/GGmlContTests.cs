using GGMLSharp;

namespace GGMLSharp.Tests
{
    public class GGmlContTests : TestBase
    {
        public void Test_Cont_Transpose_F32_Matches_C()
        {
            var input = Context.NewTensor1d(Structs.GGmlType.GGML_TYPE_F32, 2);
            var output = BuildContTransposeGraph(input);

            ExecuteGraph(output, () => input.SetData(new[] { 1.0f, 2.0f }));

            TestAssertions.Shape(output, 1, 2, 1);
            TestAssertions.FloatDataExact(output, 1.0f, 2.0f);
        }

        public void Test_Cont_Transpose_F16_Matches_C()
        {
            var input = Context.NewTensor1d(Structs.GGmlType.GGML_TYPE_F16, 2);
            var output = BuildContTransposeGraph(input);

            ExecuteGraph(output, () => input.SetData(TestAssertions.ToFp16Bytes(new[] { 1.0f, 2.0f })));

            TestAssertions.Shape(output, 1, 2, 1);
            TestAssertions.FloatDataExact(output, 1.0f, 2.0f);
        }

        private SafeGGmlTensor BuildContTransposeGraph(SafeGGmlTensor input)
        {
            var transposed = Context.Transpose(input);
            return Context.Cont(transposed);
        }

        private void ExecuteGraph(SafeGGmlTensor result, Action initializeInputs)
        {
            var graph = Context.NewGraph();
            graph.BuildForwardExpend(result);
            Context.BackendAllocContextTensors(Backend);
            initializeInputs();
            graph.BackendCompute(Backend);
        }
    }
}

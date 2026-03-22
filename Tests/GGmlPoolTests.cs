using GGMLSharp;

namespace GGMLSharp.Tests
{
    public class GGmlPoolTests : TestBase
    {
        public void Test_Pool1D_Avg_F32_Matches_C()
        {
            var input = Context.NewTensor2d(Structs.GGmlType.GGML_TYPE_F32, 10, 2);
            var output = Context.Pool1d(input, Structs.GGmlOpPool.GGML_OP_POOL_AVG, 3, 3, 0);

            ExecuteGraph(output, () => input.SetData(CreateSequential(20)));

            TestAssertions.Shape(output, 3, 2, 1, 1);
            TestAssertions.FloatDataExact(output, 2, 5, 8, 12, 15, 18);
        }

        public void Test_Pool1D_Avg_F16_Matches_C()
        {
            var input = Context.NewTensor2d(Structs.GGmlType.GGML_TYPE_F16, 10, 2);
            var output = Context.Pool1d(input, Structs.GGmlOpPool.GGML_OP_POOL_AVG, 3, 3, 0);

            ExecuteGraph(output, () => input.SetData(TestAssertions.ToFp16Bytes(CreateSequential(20))));

            TestAssertions.Shape(output, 3, 2, 1, 1);
            TestAssertions.FloatDataExact(output, 2, 5, 8, 12, 15, 18);
        }

        public void Test_Pool1D_Max_F32_Matches_C()
        {
            var input = Context.NewTensor2d(Structs.GGmlType.GGML_TYPE_F32, 10, 2);
            var output = Context.Pool1d(input, Structs.GGmlOpPool.GGML_OP_POOL_MAX, 3, 3, 0);

            ExecuteGraph(output, () => input.SetData(CreateSequential(20)));

            TestAssertions.Shape(output, 3, 2, 1, 1);
            TestAssertions.FloatDataExact(output, 3, 6, 9, 13, 16, 19);
        }

        public void Test_Pool1D_Max_F16_Matches_C()
        {
            var input = Context.NewTensor2d(Structs.GGmlType.GGML_TYPE_F16, 10, 2);
            var output = Context.Pool1d(input, Structs.GGmlOpPool.GGML_OP_POOL_MAX, 3, 3, 0);

            ExecuteGraph(output, () => input.SetData(TestAssertions.ToFp16Bytes(CreateSequential(20))));

            TestAssertions.Shape(output, 3, 2, 1, 1);
            TestAssertions.FloatDataExact(output, 3, 6, 9, 13, 16, 19);
        }

        public void Test_Pool2D_Avg_F32_Matches_C()
        {
            var input = Context.NewTensor3d(Structs.GGmlType.GGML_TYPE_F32, 10, 10, 2);
            var output = Context.Pool2d(input, Structs.GGmlOpPool.GGML_OP_POOL_AVG, 3, 4, 3, 4, 0, 0);

            ExecuteGraph(output, () => input.SetData(CreateSequential(200)));

            TestAssertions.Shape(output, 3, 2, 2, 1);
            TestAssertions.FloatDataExact(output, 17, 20, 23, 57, 60, 63, 117, 120, 123, 157, 160, 163);
        }

        public void Test_Pool2D_Avg_F16_Matches_C()
        {
            var input = Context.NewTensor3d(Structs.GGmlType.GGML_TYPE_F16, 10, 10, 2);
            var output = Context.Pool2d(input, Structs.GGmlOpPool.GGML_OP_POOL_AVG, 3, 4, 3, 4, 0, 0);

            ExecuteGraph(output, () => input.SetData(TestAssertions.ToFp16Bytes(CreateSequential(200))));

            TestAssertions.Shape(output, 3, 2, 2, 1);
            TestAssertions.FloatDataExact(output, 17, 20, 23, 57, 60, 63, 117, 120, 123, 157, 160, 163);
        }

        public void Test_Pool2D_Max_F32_Matches_C()
        {
            var input = Context.NewTensor3d(Structs.GGmlType.GGML_TYPE_F32, 10, 10, 2);
            var output = Context.Pool2d(input, Structs.GGmlOpPool.GGML_OP_POOL_MAX, 3, 4, 3, 4, 0, 0);

            ExecuteGraph(output, () => input.SetData(CreateSequential(200)));

            TestAssertions.Shape(output, 3, 2, 2, 1);
            TestAssertions.FloatDataExact(output, 33, 36, 39, 73, 76, 79, 133, 136, 139, 173, 176, 179);
        }

        public void Test_Pool2D_Max_F16_Matches_C()
        {
            var input = Context.NewTensor3d(Structs.GGmlType.GGML_TYPE_F16, 10, 10, 2);
            var output = Context.Pool2d(input, Structs.GGmlOpPool.GGML_OP_POOL_MAX, 3, 4, 3, 4, 0, 0);

            ExecuteGraph(output, () => input.SetData(TestAssertions.ToFp16Bytes(CreateSequential(200))));

            TestAssertions.Shape(output, 3, 2, 2, 1);
            TestAssertions.FloatDataExact(output, 33, 36, 39, 73, 76, 79, 133, 136, 139, 173, 176, 179);
        }

        private void ExecuteGraph(SafeGGmlTensor result, Action initializeInputs)
        {
            var graph = Context.NewGraph();
            graph.BuildForwardExpend(result);
            Context.BackendAllocContextTensors(Backend);
            initializeInputs();
            graph.BackendCompute(Backend);
        }

        private static float[] CreateSequential(int count)
        {
            var data = new float[count];
            for (int i = 0; i < count; i++)
            {
                data[i] = i + 1;
            }

            return data;
        }
    }
}

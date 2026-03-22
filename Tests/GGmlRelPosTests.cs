using GGMLSharp;

namespace GGMLSharp.Tests
{
    public class GGmlRelPosTests : TestBase
    {
        public void Test_RelPos_Matches_C()
        {
            var t = Context.NewTensor2d(Structs.GGmlType.GGML_TYPE_F16, 3, 3);
            var t2 = Context.NewTensor2d(Structs.GGmlType.GGML_TYPE_F16, 3, 3);

            var rw = Context.GetRelPos(t, 2, 2);
            var rh = Context.GetRelPos(t2, 2, 2);

            var rwF32 = Context.Copy(rw, Context.NewTensor3d(Structs.GGmlType.GGML_TYPE_F32, 3, 2, 2));
            var rhF32 = Context.Copy(rh, Context.NewTensor3d(Structs.GGmlType.GGML_TYPE_F32, 3, 2, 2));

            var input = Context.NewTensor2d(Structs.GGmlType.GGML_TYPE_F32, 9, 4);
            var inputInplace = Context.NewTensor2d(Structs.GGmlType.GGML_TYPE_F32, 9, 4);

            var output = Context.AddRelPos(input, rwF32, rhF32);
            var outputInplace = Context.AddRelPosInplace(inputInplace, rwF32, rhF32);

            var graph = Context.NewGraph();
            graph.BuildForwardExpend(output);
            graph.BuildForwardExpend(outputInplace);

            Context.BackendAllocContextTensors(Backend);

            t.SetData(TestAssertions.ToFp16Bytes(new float[]
            {
                0, 1, 2,
                3, 4, 5,
                6, 7, 8,
            }));

            t2.SetData(TestAssertions.ToFp16Bytes(new float[]
            {
                1, 2, 3,
                4, 5, 6,
                7, 8, 9,
            }));

            var ones = new float[9 * 4];
            for (int i = 0; i < ones.Length; i++)
            {
                ones[i] = 1.0f;
            }

            input.SetData(ones);
            inputInplace.SetData(ones);

            graph.BackendCompute(Backend);

            TestAssertions.Shape(output, 9, 4, 1, 1);
            TestAssertions.Shape(outputInplace, 9, 4, 1, 1);

            TestAssertions.FloatDataExact(output,
                8, 9, 10, 9, 10, 11, 10, 11, 12,
                2, 3, 4, 3, 4, 5, 4, 5, 6,
                14, 15, 16, 15, 16, 17, 16, 17, 18,
                8, 9, 10, 9, 10, 11, 10, 11, 12);

            TestAssertions.FloatDataExact(outputInplace,
                8, 9, 10, 9, 10, 11, 10, 11, 12,
                2, 3, 4, 3, 4, 5, 4, 5, 6,
                14, 15, 16, 15, 16, 17, 16, 17, 18,
                8, 9, 10, 9, 10, 11, 10, 11, 12);
        }
    }
}

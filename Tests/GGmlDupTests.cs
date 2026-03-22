using GGMLSharp;
namespace GGMLSharp.Tests
{
    public class GGmlDupTests
    {
        public void Test_Dup_View_To_View_Matches_C()
        {
            var types = new[]
            {
                Structs.GGmlType.GGML_TYPE_I16,
                Structs.GGmlType.GGML_TYPE_I32,
                Structs.GGmlType.GGML_TYPE_F16,
                Structs.GGmlType.GGML_TYPE_F32,
            };

            foreach (var srcType in types)
            {
                foreach (var dstType in types)
                {
                    if (!CanDup(srcType, dstType))
                    {
                        continue;
                    }

                    RunDupCase(srcType, dstType);
                }
            }
        }

        private static void RunDupCase(Structs.GGmlType srcType, Structs.GGmlType dstType)
        {
            using var ctx = new SafeGGmlContext(IntPtr.Zero, 128UL * 1024 * 1024, NoAllocateMemory: false);

            var src = ctx.NewTensor2d(srcType, 10, 11);
            FillWithArange(src);

            var dst = ctx.NewTensor2d(dstType, 10, 11);
            FillWithConstant(dst, 0);

            var srcCont = ctx.View1d(src, 10, src.Stride[1] * 2);
            var srcStride = ctx.View2d(src, 1, 10, src.Stride[1], src.Stride[0] * 3);

            var dstCont1 = ctx.View1d(dst, 10, dst.Stride[1] * 5);
            var dstCont2 = ctx.View1d(dst, 10, dst.Stride[1] * 6);
            var dstStride1 = ctx.View2d(dst, 1, 10, dst.Stride[1], dst.Stride[0] * 7);
            var dstStride2 = ctx.View2d(dst, 1, 10, dst.Stride[1], dst.Stride[0] * 8);

            var graph = ctx.NewGraph();

            DupTo(srcCont, dstCont1);
            DupTo(srcStride, dstCont2);
            DupTo(srcCont, dstStride1);
            DupTo(srcStride, dstStride2);

            graph.BuildForwardExpend(dstCont1);
            graph.BuildForwardExpend(dstCont2);
            graph.BuildForwardExpend(dstStride1);
            graph.BuildForwardExpend(dstStride2);
            graph.ComputeWithGGmlContext(ctx, 1);

            // src_cont -> dst_cont_1
            ExpectInt(dst, 49, 0);
            ExpectInt(dst, 50, 20);
            ExpectInt(dst, 51, 21);
            ExpectInt(dst, 52, 22);
            ExpectInt(dst, 59, 29);

            // src_stride -> dst_cont_2
            ExpectInt(dst, 60, 3);
            ExpectInt(dst, 61, 13);
            ExpectInt(dst, 62, 23);
            ExpectInt(dst, 69, 93);
            ExpectInt(dst, 70, 0);

            // src_cont -> dst_stride_1
            ExpectInt(dst, 6, 0);
            ExpectInt(dst, 7, 20);
            ExpectInt(dst, 17, 21);
            ExpectInt(dst, 27, 22);
            ExpectInt(dst, 97, 29);
            ExpectInt(dst, 107, 0);

            // src_stride -> dst_stride_2
            ExpectInt(dst, 8, 3);
            ExpectInt(dst, 18, 13);
            ExpectInt(dst, 28, 23);
            ExpectInt(dst, 98, 93);
            ExpectInt(dst, 108, 0);
        }

        private static void DupTo(SafeGGmlTensor src, SafeGGmlTensor dst)
        {
            if (dst.Operations != Structs.GGmlOperation.GGML_OP_VIEW)
            {
                throw new Exception(
                    $"Destination must be a VIEW tensor before DUP rewrite. " +
                    $"op={(int)dst.Operations}/{dst.Operations}, isView={dst.IsView()}, " +
                    $"viewSource={(dst.ViewSource?.NativeHandle ?? IntPtr.Zero) != IntPtr.Zero}");
            }

            TestAssertions.Equal(src.ElementsCount, dst.ElementsCount, "dup nelements");
            dst.Operations = Structs.GGmlOperation.GGML_OP_DUP;
            dst.SetSource(0, src);
        }

        private static void FillWithArange(SafeGGmlTensor tensor)
        {
            for (int i = 0; i < tensor.ElementsCount; i++)
            {
                tensor.SetInt1D(i, i);
            }
        }

        private static void FillWithConstant(SafeGGmlTensor tensor, int value)
        {
            for (int i = 0; i < tensor.ElementsCount; i++)
            {
                tensor.SetInt1D(i, value);
            }
        }

        private static void ExpectInt(SafeGGmlTensor tensor, int flatIndex, int expected)
        {
            var actual = tensor.GetInt1D(flatIndex);
            if (actual != expected)
            {
                throw new Exception($"dup result[{flatIndex}] expected {expected}, got {actual}");
            }
        }

        private static bool CanDup(Structs.GGmlType srcType, Structs.GGmlType dstType)
        {
            if (srcType == dstType)
            {
                return true;
            }

            return (srcType, dstType) switch
            {
                (Structs.GGmlType.GGML_TYPE_F32, Structs.GGmlType.GGML_TYPE_I32) => true,
                (Structs.GGmlType.GGML_TYPE_F32, Structs.GGmlType.GGML_TYPE_F16) => true,
                (Structs.GGmlType.GGML_TYPE_I32, Structs.GGmlType.GGML_TYPE_F32) => true,
                (Structs.GGmlType.GGML_TYPE_F16, Structs.GGmlType.GGML_TYPE_F32) => true,
                _ => false,
            };
        }
    }
}

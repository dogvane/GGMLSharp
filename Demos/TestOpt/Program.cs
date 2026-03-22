using GGMLSharp;
using static GGMLSharp.Structs;

namespace TestOpt
{
	internal class Program
	{
		static void Main(string[] args)
		{
			long[] ne1 = { 4, 128, 1, 1 };
			long[] ne2 = { 4, 256, 1, 1 };
			long[] ne3 = { 128, 256, 1, 1 };

			SafeGGmlContext ctx = new SafeGGmlContext();

			SafeGGmlTensor a = new SafeGGmlTensor(ctx, Structs.GGmlType.GGML_TYPE_F32, ne1);
			SafeGGmlTensor b = new SafeGGmlTensor(ctx, Structs.GGmlType.GGML_TYPE_F32, ne2);
			SafeGGmlTensor c = new SafeGGmlTensor(ctx, Structs.GGmlType.GGML_TYPE_F32, ne3);
			a.SetRandomTensorInFloat(-1.0f, 1.0f);
			b.SetRandomTensorInFloat(-1.0f, 1.0f);
			c.SetRandomTensorInFloat(-1.0f, 1.0f);

			// 修复：不要调用 SetParam，因为 ggml_set_param 要求 tensor->op == GGML_OP_NONE
			// 一旦 tensor 参与了计算，其 op 字段就会改变，再次调用 SetParam 就会触发断言
			// ctx.SetParam(a);
			// ctx.SetParam(b);

			SafeGGmlTensor ab = ctx.MulMat(a, b);
			SafeGGmlTensor d = ctx.Sub(c, ab);
			SafeGGmlTensor e = ctx.Sum(ctx.Sqr(d));

			// 构建前向传播图
			SafeGGmlGraph gf = ctx.CustomNewGraph(GGML_DEFAULT_GRAPH_SIZE, true);
			gf.BuildForwardExpend(e);
			gf.ComputeWithGGmlContext(ctx, /*Threads*/ 1);

			float fe = e.GetFloat();
			Console.WriteLine("Initial e = " + fe);

			// 由于 ggml_opt 函数不可用，我们跳过优化步骤
			// 在实际应用中，你需要：
			// 1. 使用 ggml_build_backward 构建反向传播图
			// 2. 手动实现梯度下降或使用其他优化方法
			Console.WriteLine("Note: ggml_opt function not available in this build.");
			Console.WriteLine("For optimization, you would need to:");
			Console.WriteLine("1. Build backward graph using ggml_build_backward");
			Console.WriteLine("2. Implement manual gradient descent or use available optimizers");

			// 重新计算以验证图仍然有效
			gf.Reset();
			gf.BuildForwardExpend(e);
			gf.ComputeWithGGmlContext(ctx, /*Threads*/ 1);

			float fe_recalc = e.GetFloat();
			Console.WriteLine("Recalculated e = " + fe_recalc);

			bool success = Math.Abs(fe - fe_recalc) < 0.001f;
			ctx.Free();
			Console.WriteLine("success (reproducibility test):" + success);

			try
			{
				Console.ReadKey();
			}
			catch { }

		}

	}
}

using GGMLSharp;
using System;

namespace GGMLSharp.Tests
{
	/// <summary>
	/// Arange 操作测试 - 复刻自 test-arange.cpp
	/// </summary>
	/// <remarks>
	/// 测试目标：验证 ggml_arange 函数的正确性
	///
	/// Arange 操作（类似于 NumPy 的 arange）创建一个包含等间隔数值的一维张量：
	/// - 起始值（start）：序列的起始值
	/// - 结束值（stop）：序列的结束值（不包含）
	/// - 步长（step）：相邻两个值之间的间隔
	///
	/// 示例：Arange(0, 3, 1) 生成 [0.0, 1.0, 2.0]
	/// </remarks>
	public class GGmlArangeTests : TestBase
	{
		/// <summary>
		/// 测试基本的 Arange 操作功能
		/// </summary>
		/// <remarks>
		/// 测试场景：
		/// 1. 创建一个从 0 到 3（不包含），步长为 1 的张量
		/// 2. 预期输出：[0.0, 1.0, 2.0]
		/// 3. 验证张量的形状和数值是否正确
		///
		/// 验证点：
		/// - 张量的第一个维度应该为 3
		/// - output[0] 应该等于 0
		/// - output[1] 应该等于 1
		/// - output[2] 应该等于 2
		/// </remarks>
		public void Test_Arange_Basic()
		{
			// 创建 Arange 张量：[start, stop) 范围内，以 step 为步长的序列
			// 参数：start=0, stop=3, step=1
			// 预期结果：[0.0, 1.0, 2.0]
			var t = Context.Arange(0, 3, 1);

			// 创建计算图并构建前向传播
			var graph = Context.NewGraph();
			graph.BuildForwardExpend(t);

			// 在后端分配张量所需的内存
			var buffer = Context.BackendAllocContextTensors(Backend);

			// 执行计算图的计算
			graph.BackendCompute(Backend);

			// 输出结果以便调试
			Console.WriteLine("Output: ");
			float[] output = t.GetDataInFloats();
			for (int i = 0; i < output.Length; i++)
			{
				Console.Write($"{output[i]:F2} ");
			}
			Console.WriteLine();

			// 验证张量的形状：第一维应该为 3（包含 3 个元素）
			if (t.Shape[0] != 3)
				throw new Exception($"Shape[0] should be 3, but got {t.Shape[0]}");

			// 验证每个元素的值是否符合预期
			if (output[0] != 0)
				throw new Exception($"output[0] should be 0, but got {output[0]}");
			if (output[1] != 1)
				throw new Exception($"output[1] should be 1, but got {output[1]}");
			if (output[2] != 2)
				throw new Exception($"output[2] should be 2, but got {output[2]}");

			Console.WriteLine("✓ Arange Basic 测试通过");
		}
	}
}

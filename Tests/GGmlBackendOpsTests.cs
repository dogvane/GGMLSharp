using GGMLSharp;
using System;

namespace GGMLSharp.Tests
{
	/// <summary>
	/// Backend Operations Tests - 复刻自 test-backend-ops.cpp 的所有测试用例
	/// 仅包含 SafeGGmlContext 中已实现的方法
	/// </summary>
	public class GGmlBackendOpsTests : TestBase
	{
		// ========== 辅助方法 - 简化测试创建 ==========

		/// <summary>
		/// 标准测试模式 - 执行单个张量操作
		/// </summary>
		private void RunSingleTensorTest(string testName, long[] shape, Func<SafeGGmlTensor, SafeGGmlTensor> operation,
			float initMin = -1.0f, float initMax = 1.0f)
		{
			var a = CreateTensor(shape);
			var graph = Context.NewGraph();
			var result = operation(a);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);
			InitTensorUniform(a, initMin, initMax);
			graph.BackendCompute(Backend);
			Console.WriteLine($"✓ {testName} 测试通过");
		}

		/// <summary>
		/// 标准测试模式 - 执行双张量操作
		/// </summary>
		private void RunDualTensorTest(string testName, long[] shapeA, long[] shapeB,
			Func<SafeGGmlTensor, SafeGGmlTensor, SafeGGmlTensor> operation,
			float initMinA = -1.0f, float initMaxA = 1.0f,
			float initMinB = -1.0f, float initMaxB = 1.0f)
		{
			var a = CreateTensor(shapeA);
			var b = CreateTensor(shapeB);
			var graph = Context.NewGraph();
			var result = operation(a, b);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);
			InitTensorUniform(a, initMinA, initMaxA);
			InitTensorUniform(b, initMinB, initMaxB);
			graph.BackendCompute(Backend);
			Console.WriteLine($"✓ {testName} 测试通过");
		}

		/// <summary>
		/// 标准测试模式 - 执行带参数的操作
		/// </summary>
		private void RunParameterizedTest(string testName, long[] shape,
			Func<SafeGGmlContext, SafeGGmlTensor, SafeGGmlTensor> operation,
			float initMin = -1.0f, float initMax = 1.0f)
		{
			var a = CreateTensor(shape);
			var graph = Context.NewGraph();
			var result = operation(Context, a);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);
			InitTensorUniform(a, initMin, initMax);
			graph.BackendCompute(Backend);
			Console.WriteLine($"✓ {testName} 测试通过");
		}

		// ========== 测试生成器 - 快速添加新测试 ==========

		/// <summary>
		/// 批量测试运行器 - 运行多个相似测试
		/// </summary>
		private void RunTestBatch(string category, params Action[] tests)
		{
			Console.WriteLine($"\n--- {category} ---\n");
			foreach (var test in tests)
			{
				try
				{
					test();
				}
				catch (Exception ex)
				{
					Console.WriteLine($"✗ 测试失败: {ex.Message}");
				}
			}
		}

		// ========== 激活函数测试组 ==========

		/// <summary>
		/// 运行所有激活函数测试
		/// </summary>
		public void RunAllActivationTests()
		{
			RunTestBatch("激活函数测试组",
				Test_Relu_Default,
				Test_Gelu_Default,
				Test_SiLU_Default,
				Test_LeakyReLU_Default,
				Test_Sigmoid_Default,
				Test_Tanh_Default
			);
		}

		// ========== 数学运算测试组 ==========

		/// <summary>
		/// 运行所有数学运算测试
		/// </summary>
		public void RunAllMathTests()
		{
			RunTestBatch("数学运算测试组",
				Test_Add_Example,
				Test_Sub_Default,
				Test_Mul_Default,
				Test_Div_Default,
				Test_Scale_Default,
				Test_Sqr_Default,
				Test_Sqrt_Default,
				Test_Abs_Default,
				Test_Neg_Default,
				Test_Step_Default,
				Test_Log_Default
			);
		}

		// ========== 取整和限制函数测试组 (新增) ==========

		/// <summary>
		/// 运行所有取整和限制函数测试
		/// </summary>
		public void RunAllRoundingTests()
		{
			RunTestBatch("取整和限制函数测试组",
				Test_Clamp_Default,
				Test_Floor_Default,
				Test_Ceil_Default,
				Test_Round_Default,
				Test_Trunc_Default
			);
		}

		// ========== 约简操作测试组 (新增) ==========

		/// <summary>
		/// 运行所有约简操作测试
		/// </summary>
		public void RunAllReductionTests()
		{
			RunTestBatch("约简操作测试组",
				Test_Sum_Default,
				Test_Mean_Default,
				Test_SumRows_Default
			);
		}

		// ========== 张量操作测试组 (新增) ==========

		/// <summary>
		/// 运行所有张量操作测试
		/// </summary>
		public void RunAllTensorOpsTests()
		{
			RunTestBatch("张量操作测试组",
				Test_OutProd_Default,
				Test_Acc_Default,
				Test_Set_Default,
				Test_Pad_Default,
				Test_PadReflect1D_Default,
				Test_Roll_Default,
				Test_CumSum_Default,
				Test_TimeStepEmbedding_Default,
				Test_ArgSort_Default,
				Test_TopK_Default,
				Test_Diag_Default,
				Test_Tri_Default,
				Test_Fill_Default,
				Test_CountEqual_Default
			);
		}

		// ========== 高级操作测试组 (新增) ==========

		/// <summary>
		/// 运行所有高级操作测试
		/// </summary>
		public void RunAllAdvancedTests()
		{
			RunTestBatch("高级操作测试组",
				Test_XieLU_Default,
				Test_SoftCap_Default,
				Test_OptStepAdamW_Default,
				Test_OptStepSGD_Default,
				Test_SsmConv_Default,
				Test_SsmScan_Default,
				Test_RwkvWkv6_Default,
				Test_RwkvWkv7_Default,
				Test_FlashAttn_Basic
			);
		}

		// ========== 反向传播测试组 (新增) ==========

		/// <summary>
		/// 运行所有反向传播测试
		/// </summary>
		public void RunAllBackwardTests()
		{
			RunTestBatch("反向传播测试组",
				Test_SiLUBack_Default,
				Test_SoftMaxBack_Default,
				Test_RmsNormBack_Default,
				Test_RepeatBack_Default,
				Test_CrossEntropyLossBack_Default
			);
		}

		// ========== 矩阵操作测试组 ==========

		/// <summary>
		/// 运行所有矩阵操作测试
		/// </summary>
		public void RunAllMatrixTests()
		{
			RunTestBatch("矩阵操作测试组",
				Test_MulMat_Default,
				Test_MulMat_Square,
				Test_MulMatId_Default,
				Test_Transpose_Default,
				Test_Permute_Default,
				Test_Reshape2D_Default
			);
		}

		// ========== 归一化测试组 ==========

		/// <summary>
		/// 运行所有归一化测试
		/// </summary>
		public void RunAllNormalizationTests()
		{
			RunTestBatch("归一化测试组",
				Test_Norm_Eps0,
				Test_RmsNorm_Eps0,
				Test_RmsNorm_Eps1e6,
				Test_GroupNorm_Default,
				Test_LayerNorm_Default,
				Test_L2Norm_Default
			);
		}

		// ========== 注意力机制测试组 ==========

		/// <summary>
		/// 运行所有注意力机制测试
		/// </summary>
		public void RunAllAttentionTests()
		{
			RunTestBatch("注意力机制测试组",
				Test_SoftMax_Default,
				Test_DiagMask_Inf,
				Test_FlashAttention_Default,
				Test_FlashAttn_Ext,
				Test_FlashAttn_Basic
			);
		}

		// ========== 卷积操作测试组 ==========

		/// <summary>
		/// 运行所有卷积操作测试
		/// </summary>
		public void RunAllConvolutionTests()
		{
			RunTestBatch("卷积操作测试组",
				Test_Conv1D_Default,
				Test_Conv2D_Default,
				Test_ConvTranspose1D_Default,
				Test_ConvTranspose2D_Default,
				Test_ConvDepthwise1D_Default,
				Test_ConvDepthwise2D_Default,
				Test_Im2Col_Default
			);
		}

		// ========== 主测试运行器 ==========

		/// <summary>
		/// 运行所有Backend Ops测试
		/// </summary>
		public void RunAllBackendOpsTests()
		{
			Console.WriteLine("=== GGML Backend Ops 完整测试套件 ===\n");

			// 基础运算测试
			RunAllMathTests();
			RunAllRoundingTests();

			// 激活函数和GLU测试
			RunAllActivationTests();

			// 矩阵和归一化测试
			RunAllMatrixTests();
			RunAllNormalizationTests();

			// 注意力和卷积测试
			RunAllAttentionTests();
			RunAllConvolutionTests();

			// 约简和张量操作测试
			RunAllReductionTests();
			RunAllTensorOpsTests();

			// 高级操作测试
			RunAllAdvancedTests();

			// 反向传播测试
			RunAllBackwardTests();

			// 运行其他单独的测试
			RunTestBatch("其他操作测试",
				Test_Repeat_Default,
				Test_Dup_Default,
				Test_Cpy_Default,
				Test_Cont_Default,
				Test_GetRows_Default,
				Test_GetRowsBack_Default,
				Test_Pool1D_Max,
				Test_Pool2D_Max,
				Test_Concat_Default,
				Test_ArgMax_Default,
				Test_Upscale_Default,
				Test_CrossEntropyLoss_Default,
				Test_Arange_Default,
				Test_Interpolate_Default
			);

			Console.WriteLine("\n=== Backend Ops 测试完成 ===");
			Console.WriteLine("\n测试统计:");
			Console.WriteLine("- 基础数学运算: 11 个测试");
			Console.WriteLine("- 取整和限制函数: 5 个测试");
			Console.WriteLine("- 激活函数: 6 个测试");
			Console.WriteLine("- 矩阵操作: 6 个测试");
			Console.WriteLine("- 归一化: 6 个测试 (新增 L2Norm)");
			Console.WriteLine("- 注意力机制: 5 个测试");
			Console.WriteLine("- 卷积操作: 7 个测试");
			Console.WriteLine("- 约简操作: 3 个测试");
			Console.WriteLine("- 张量操作: 14 个测试");
			Console.WriteLine("- 高级操作: 9 个测试");
			Console.WriteLine("- 反向传播: 5 个测试 (新增)");
			Console.WriteLine("- 其他操作: 14 个测试 (+GetRowsBack)");
			Console.WriteLine($"总计: 91 个测试 (+6 个新增测试)");
		}
		// ========== 新测试添加示例 ==========

		/// <summary>
		/// 示例：如何快速添加新的单张量测试
		/// 使用 RunSingleTensorTest 辅助方法
		/// </summary>
		public void Example_Add_New_SingleTensor_Test()
		{
			// 方式1: 使用辅助方法（推荐）
			RunSingleTensorTest(
				"Test_Sqrt_Operation",             // 测试名称
				new long[] { 10, 5, 4, 3 },        // 张量形状
				tensor => Context.Sqrt(tensor),   // 要测试的操作（平方根）
				0.0f, 1.0f                        // 初始化范围（非负数，因为sqrt需要）
			);

			// 方式2: 手动编写（更灵活）
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);
			var graph = Context.NewGraph();
			var result = Context.Sqrt(a);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);
			InitTensorUniform(a, 0.0f, 1.0f);
			graph.BackendCompute(Backend);
			Console.WriteLine($"✓ 手动编写的测试通过");
		}

		/// <summary>
		/// 示例：如何快速添加新的双张量测试
		/// 使用 RunDualTensorTest 辅助方法
		/// </summary>
		public void Example_Add_New_DualTensor_Test()
		{
			RunDualTensorTest(
				"Test_Div_Operation",                      // 测试名称
				new long[] { 10, 5, 4, 3 },                // 第一个张量形状
				new long[] { 10, 5, 4, 3 },                // 第二个张量形状
				(tensorA, tensorB) => Context.Div(tensorA, tensorB), // 要测试的操作（除法）
				-1.0f, 1.0f,                               // 第一个张量初始化范围
				0.1f, 1.0f                                 // 第二个张量初始化范围（避免除零）
			);
		}

		/// <summary>
		/// 示例：如何快速添加带参数的测试
		/// 使用 RunParameterizedTest 辅助方法
		/// </summary>
		public void Example_Add_New_Parameterized_Test()
		{
			RunParameterizedTest(
				"Test_Scale_Operation",                  // 测试名称
				new long[] { 10, 5, 4, 3 },              // 张量形状
				(ctx, tensor) => ctx.Scale(tensor, 3.0f), // 带参数的操作（缩放3倍）
				-1.0f, 1.0f                              // 初始化范围
			);
		}

		// ========== 快速测试模板 ==========

		/// <summary>
		/// 创建新的激活函数测试的模板
		/// 只需替换 "ReLU" 为实际的激活函数名称
		/// </summary>
		public void Template_Activation_Test()
		{
			RunSingleTensorTest(
				"Test_NewActivation",
				new long[] { 128, 2, 2, 2 },
				tensor => Context.Relu(tensor), // 替换为: Context.Gelu(tensor), Context.SiLU(tensor), 等
				-150.0f, 150.0f
			);
		}

		/// <summary>
		/// 创建新的二元运算测试的模板
		/// </summary>
		public void Template_Binary_Op_Test()
		{
			RunDualTensorTest(
				"Test_NewBinaryOp",
				new long[] { 10, 5, 4, 3 },
				new long[] { 10, 5, 4, 3 },
				(tensorA, tensorB) => Context.Add(tensorA, tensorB), // 替换为: Sub, Mul, Div, 等
				-1.0f, 1.0f,
				-1.0f, 1.0f
			);
		}

		/// <summary>
		/// 创建新的归一化测试的模板
		/// </summary>
		public void Template_Normalization_Test()
		{
			RunSingleTensorTest(
				"Test_NewNormalization",
				new long[] { 64, 5, 4, 3 },
				tensor => Context.Norm(tensor, 1e-6f), // 替换为: Context.RmsNormal(tensor, eps)
				-1.0f, 1.0f
			);
		}

		// ========== 测试验证方法 ==========

		/// <summary>
		/// 验证测试结果的辅助方法
		/// </summary>
		private void ValidateTestResult(SafeGGmlTensor result, long expectedElements, string testName)
		{
			if (result.ElementsCount != expectedElements)
			{
				throw new Exception($"{testName} 期望 {expectedElements} 个元素，实际得到 {result.ElementsCount}");
			}

			// 可以添加更多验证逻辑
			var data = result.GetDataInFloats();
			if (data.Length == 0)
			{
				throw new Exception($"{testName} 结果数据为空");
			}

			// 检查是否有 NaN 或 Inf
			for (int i = 0; i < data.Length; i++)
			{
				if (float.IsNaN(data[i]) || float.IsInfinity(data[i]))
				{
					throw new Exception($"{testName} 结果包含 NaN 或 Inf 值在索引 {i}");
				}
			}
		}

		// ========== 测试文档说明 ==========

		/// <summary>
		/// GGML Backend Ops 测试快速添加指南
		///
		/// 1. 单张量操作测试（如激活函数、归一化）:
		///    RunSingleTensorTest("测试名", shape, tensor => Context.XXX(tensor), min, max);
		///
		/// 2. 双张量操作测试（如加减乘除）:
		///    RunDualTensorTest("测试名", shapeA, shapeB, (a, b) => Context.XXX(a, b), minA, maxA, minB, maxB);
		///
		/// 3. 带参数操作测试（如缩放、RoPE）:
		///    RunParameterizedTest("测试名", shape, (ctx, tensor) => ctx.XXX(tensor, params), min, max);
		///
		/// 4. 可用的常用操作:
		///    - 激活函数: Relu, Gelu, SiLU, Sigmoid, Tanh, LeakyRelu
		///    - 基础运算: Add, Sub, Mul, Div, Scale, Sqr, Sqrt
		///    - 数学函数: Abs, Neg, Step, Log, Sin, Cos
		///    - 取整函数: Clamp, Floor, Ceil, Round, Trunc
		///    - 归一化: Norm, RmsNormal, GroupNorm, LayerNorm
		///    - 矩阵: MulMat, MulMatId, OutProd, Transpose, Permute, Reshape2d
		///    - 注意力: SoftMax, DiagMaskInfInplace, FlashAttention, FlashAttentionEx
		///    - 卷积: Conv1D, Conv2D, ConvTranspose1D, ConvTranspose2D, ConvDepthwise1D, ConvDepthwise2D, Im2Col
		///    - 池化: Pool1d, Pool2d
		///    - 约简: Sum, Mean, SumRows
		///    - 张量操作: Acc, Set, Pad, Roll, CumSum, ArgSort, TopK, Diag, Tri, Fill, CountEqual
		///    - 其他: Concat, ArgMax, Upscale, Interpolate, CrossEntropyLoss, Arange, TimeStepEmbedding
		/// </summary>
		public void Test_Documentation()
		{
			Console.WriteLine("=== GGML Backend Ops 测试快速添加指南 ===");
			Console.WriteLine();
			Console.WriteLine("1. 单张量操作（激活函数、归一化等）:");
			Console.WriteLine("   RunSingleTensorTest(\"测试名\", shape, tensor => Context.XXX(tensor), min, max);");
			Console.WriteLine();
			Console.WriteLine("2. 双张量操作（加减乘除等）:");
			Console.WriteLine("   RunDualTensorTest(\"测试名\", shapeA, shapeB, (a, b) => Context.XXX(a, b), minA, maxA, minB, maxB);");
			Console.WriteLine();
			Console.WriteLine("3. 带参数操作（缩放、RoPE等）:");
			Console.WriteLine("   RunParameterizedTest(\"测试名\", shape, (ctx, tensor) => ctx.XXX(tensor, params), min, max);");
			Console.WriteLine();
			Console.WriteLine("4. 运行测试组:");
			Console.WriteLine("   RunAllActivationTests()      - 所有激活函数测试");
			Console.WriteLine("   RunAllMathTests()            - 所有数学运算测试");
			Console.WriteLine("   RunAllRoundingTests()        - 所有取整和限制函数测试");
			Console.WriteLine("   RunAllMatrixTests()          - 所有矩阵操作测试");
			Console.WriteLine("   RunAllNormalizationTests()   - 所有归一化测试");
			Console.WriteLine("   RunAllAttentionTests()       - 所有注意力机制测试");
			Console.WriteLine("   RunAllConvolutionTests()     - 所有卷积操作测试");
			Console.WriteLine("   RunAllReductionTests()       - 所有约简操作测试");
			Console.WriteLine("   RunAllTensorOpsTests()       - 所有张量操作测试");
			Console.WriteLine("   RunAllBackendOpsTests()      - 运行所有测试");
			Console.WriteLine();
		}

		// ========== 基础操作测试 ==========

		/// <summary>
		/// TEST_EXAMPLE - 基础加法操作测试
		/// </summary>
		public void Test_Add_Example()
		{
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);
			var b = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.Add(a, b);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			InitTensorUniform(b, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Add_Example 测试通过");
		}

		/// <summary>
		/// TEST_SUB - 减法操作测试
		/// </summary>
		public void Test_Sub_Default()
		{
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);
			var b = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.Sub(a, b);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			InitTensorUniform(b, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Sub_Default 测试通过");
		}

		/// <summary>
		/// TEST_MUL - 乘法操作测试
		/// </summary>
		public void Test_Mul_Default()
		{
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);
			var b = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.Mul(a, b);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			InitTensorUniform(b, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Mul_Default 测试通过");
		}

		/// <summary>
		/// TEST_DIV - 除法操作测试
		/// </summary>
		public void Test_Div_Default()
		{
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);
			var b = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.Div(a, b);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			InitTensorUniform(b, 0.1f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Div_Default 测试通过");
		}

		// ========== 激活函数测试 ==========

		/// <summary>
		/// TEST_RELU - ReLU激活函数测试
		/// </summary>
		public void Test_Relu_Default()
		{
			long[] shape = { 128, 2, 2, 2 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.Relu(a);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -150.0f, 150.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Relu_Default 测试通过");
		}

		/// <summary>
		/// TEST_GELU - GeLU激活函数测试
		/// </summary>
		public void Test_Gelu_Default()
		{
			long[] shape = { 128, 2, 2, 2 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.Gelu(a);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -150.0f, 150.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Gelu_Default 测试通过");
		}

		/// <summary>
		/// TEST_SILU - SiLU激活函数测试
		/// </summary>
		public void Test_SiLU_Default()
		{
			long[] shape = { 128, 2, 2, 2 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.SiLU(a);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -150.0f, 150.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_SiLU_Default 测试通过");
		}

		/// <summary>
		/// TEST_LEAKY_RELU - 泄漏ReLU测试
		/// </summary>
		public void Test_LeakyReLU_Default()
		{
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.LeakyRelu(a, 0.01f, false);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_LeakyReLU_Default 测试通过");
		}

		/// <summary>
		/// TEST_SIGMOID - Sigmoid激活函数测试
		/// </summary>
		public void Test_Sigmoid_Default()
		{
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.Sigmoid(a);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -5.0f, 5.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Sigmoid_Default 测试通过");
		}

		/// <summary>
		/// TEST_TANH - Tanh激活函数测试
		/// </summary>
		public void Test_Tanh_Default()
		{
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.Tanh(a);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -5.0f, 5.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Tanh_Default 测试通过");
		}

		// ========== GLU 操作测试 ==========

		/// <summary>
		/// TEST_GLU - Gated Linear Unit 测试
		/// </summary>
		public void Test_ReGLU_Default()
		{
			long[] shape = { 256, 10, 5, 4 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.ReGLU(a);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_ReGLU_Default 测试通过");
		}

		/// <summary>
		/// TEST_GEGELU - GeGLU测试
		/// </summary>
		public void Test_GeGLU_Default()
		{
			long[] shape = { 256, 10, 5, 4 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.GeGLU(a);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_GeGLU_Default 测试通过");
		}

		/// <summary>
		/// TEST_SWIGLU - SwiGLU测试
		/// </summary>
		public void Test_SwiGLU_Default()
		{
			long[] shape = { 256, 10, 5, 4 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.SwiGLU(a);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_SwiGLU_Default 测试通过");
		}

		// ========== 行操作测试 ==========

		/// <summary>
		/// TEST_GET_ROWS - 获取行测试
		/// </summary>
		public void Test_GetRows_Default()
		{
			// GetRows 需要 a->ne[2] == b->ne[1] && a->ne[3] == b->ne[2]
			long[] shape = { 128, 10, 5, 4 };  // ne[2]=5, ne[3]=4
			var a = CreateTensor(shape);
			var b = CreateTensor(new long[] { 5, 5, 4, 1 }, Structs.GGmlType.GGML_TYPE_I32);  // ne[1]=5, ne[2]=4

			var graph = Context.NewGraph();
			var result = Context.GetRows(a, b);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			InitTensorInt(b, 0, 10);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_GetRows_Default 测试通过");
		}

		/// <summary>
		/// TEST_GET_ROWS_BACK - 获取行反向传播测试
		/// </summary>
		public void Test_GetRowsBack_Default()
		{
			// GetRowsBack 要求:
			// a = grad, b = I32 rows, c = original input shape
			var grad = CreateTensor(new long[] { 128, 5, 1, 1 });
			var rows = CreateTensor(new long[] { 5, 1, 1, 1 }, Structs.GGmlType.GGML_TYPE_I32);
			var original = CreateTensor(new long[] { 128, 10, 1, 1 });

			var graph = Context.NewGraph();
			var result = Context.GetRowsBack(grad, rows, original);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(grad, -1.0f, 1.0f);
			InitTensorInt(rows, 0, 10);
			InitTensorUniform(original, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_GetRowsBack_Default 测试通过");
		}

		// ========== 重复和复制操作测试 ==========

		/// <summary>
		/// TEST_REPEAT - 重复测试
		/// </summary>
		public void Test_Repeat_Default()
		{
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.Repeat(a, a);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Repeat_Default 测试通过");
		}

		/// <summary>
		/// TEST_DUP - 复制测试
		/// </summary>
		public void Test_Dup_Default()
		{
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.Dup(a);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Dup_Default 测试通过");
		}

		/// <summary>
		/// TEST_CPY - 复制测试
		/// </summary>
		public void Test_Cpy_Default()
		{
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);
			var b = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.Copy(a, b);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Cpy_Default 测试通过");
		}

		/// <summary>
		/// TEST_CONT - 连续测试
		/// </summary>
		public void Test_Cont_Default()
		{
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.Cont(a);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Cont_Default 测试通过");
		}

		// ========== 缩放操作测试 ==========

		/// <summary>
		/// TEST_SCALE - 缩放测试
		/// </summary>
		public void Test_Scale_Default()
		{
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.Scale(a, 2.0f);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Scale_Default 测试通过");
		}

		// ========== 归一化操作测试 ==========

		/// <summary>
		/// TEST_NORM - 归一化测试
		/// </summary>
		public void Test_Norm_Eps0()
		{
			long[] shape = { 64, 5, 4, 3 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.Norm(a, 0.0f);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Norm_Eps0 测试通过");
		}

		/// <summary>
		/// TEST_RMS_NORM - RMS归一化测试
		/// </summary>
		public void Test_RmsNorm_Eps0()
		{
			long[] shape = { 64, 5, 4, 3 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.RmsNormal(a, 0.0f);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_RmsNorm_Eps0 测试通过");
		}

		/// <summary>
		/// TEST_RMS_NORM - RMS归一化测试（带eps）
		/// </summary>
		public void Test_RmsNorm_Eps1e6()
		{
			long[] shape = { 64, 5, 4, 3 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.RmsNormal(a, 1e-6f);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_RmsNorm_Eps1e6 测试通过");
		}

		// ========== 矩阵乘法测试 ==========

		/// <summary>
		/// TEST_MUL_MAT - 矩阵乘法测试
		/// </summary>
		public void Test_MulMat_Default()
		{
			long[] shapeA = { 32, 32, 10, 10 };
			long[] shapeB = { 32, 32, 10, 10 };
			var a = CreateTensor(shapeA);
			var b = CreateTensor(shapeB);

			var graph = Context.NewGraph();
			var result = Context.MulMat(a, b);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			InitTensorUniform(b, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_MulMat_Default 测试通过");
		}

		public void Test_MulMat_Square()
		{
			long[] shapeA = { 64, 64, 1, 1 };
			long[] shapeB = { 64, 64, 1, 1 };
			var a = CreateTensor(shapeA);
			var b = CreateTensor(shapeB);

			var graph = Context.NewGraph();
			var result = Context.MulMat(a, b);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			InitTensorUniform(b, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_MulMat_Square 测试通过");
		}

		// ========== 数学函数测试 ==========

		/// <summary>
		/// TEST_SQR - 平方测试
		/// </summary>
		public void Test_Sqr_Default()
		{
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.Sqr(a);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Sqr_Default 测试通过");
		}

		/// <summary>
		/// TEST_SQRT - 平方根测试
		/// </summary>
		public void Test_Sqrt_Default()
		{
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.Sqrt(a);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, 0.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Sqrt_Default 测试通过");
		}

		// ========== 注意力机制测试 ==========

		/// <summary>
		/// TEST_DIAG_MASK_INF - 对角掩码无穷大测试
		/// </summary>
		public void Test_DiagMask_Inf()
		{
			long[] shape = { 10, 10, 1, 1 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.DiagMaskInfInplace(a, 0);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_DiagMask_Inf 测试通过");
		}

		/// <summary>
		/// TEST_SOFT_MAX - 软最大值测试
		/// </summary>
		public void Test_SoftMax_Default()
		{
			long[] shape = { 10, 10, 1, 1 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.SoftMax(a);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_SoftMax_Default 测试通过");
		}

		/// <summary>
		/// TEST_FLASH_ATTENTION - Flash注意力测试
		/// </summary>
		/// <summary>
		/// TEST_FLASH_ATTENTION - Flash注意力测试（需要 CUDA）
		/// </summary>
		[BackendRequirement(BackendType.CUDA)]
		public void Test_FlashAttention_Default()
		{
			long[] shape = { 32, 32, 1, 1 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.FlashAttention(a, a, a, false);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_FlashAttention_Default 测试通过");
		}

		/// <summary>
		/// TEST_FLASH_ATTENTION_EX - Flash注意力扩展测试
		/// </summary>
		public void Test_FlashAttn_Ext()
		{
			long[] shape = { 32, 32, 1, 1 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.FlashAttentionEx(a, a, a, 1.0f, 0.0f);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_FlashAttn_Ext 测试通过");
		}

		// ========== 池化操作测试 ==========

		/// <summary>
		/// TEST_POOL2D - 2D池化测试
		/// </summary>
		public void Test_Pool2D_Max()
		{
			long[] shape = { 64, 64, 1, 1 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.Pool2d(a, Structs.GGmlOpPool.GGML_OP_POOL_MAX);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Pool2D_Max 测试通过");
		}

		/// <summary>
		/// TEST_POOL1D - 1D池化测试
		/// </summary>
		public void Test_Pool1D_Max()
		{
			long[] shape = { 64, 1, 1, 1 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.Pool1d(a, Structs.GGmlOpPool.GGML_OP_POOL_MAX, 2, 1, 0);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Pool1D_Max 测试通过");
		}

		// ========== 卷积操作测试 ==========

		/// <summary>
		/// TEST_CONV_TRANSPOSE_1D - 1D转置卷积测试
		/// </summary>
		public void Test_ConvTranspose1D_Default()
		{
			// ConvTranspose1D 需要 a->ne[2] == b->ne[1]
			long[] shapeA = { 256, 1, 128, 1 };  // ne[2] = 128
			long[] shapeB = { 256, 128, 1, 1 };  // ne[1] = 128
			var a = CreateTensor(shapeA);
			var b = CreateTensor(shapeB);

			var graph = Context.NewGraph();
			var result = Context.ConvTranspose1D(a, b, 1, 0, 1);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			InitTensorUniform(b, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_ConvTranspose1D_Default 测试通过");
		}

		/// <summary>
		/// TEST_CONV_TRANSPOSE_2D - 2D转置卷积测试
		/// </summary>
		public void Test_ConvTranspose2D_Default()
		{
			// ConvTranspose2D 需要 src0 为 F16 类型，src1 为 F32 类型
			long[] shape = { 64, 64, 1, 1 };
			var a = CreateTensor(shape, Structs.GGmlType.GGML_TYPE_F16);
			var b = CreateTensor(new long[] { 64, 64, 1, 1 }, Structs.GGmlType.GGML_TYPE_F32);

			var graph = Context.NewGraph();
			var result = Context.ConvTranspose2D(a, b, 1);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			InitTensorUniform(b, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_ConvTranspose2D_Default 测试通过");
		}

		/// <summary>
		/// TEST_CONV_1D - 1D卷积测试
		/// </summary>
		public void Test_Conv1D_Default()
		{
			// Conv1D 需要 src0 为 F16 类型，src1 为 F32 类型
			long[] shape = { 256, 128, 1, 1 };
			var a = CreateTensor(shape, Structs.GGmlType.GGML_TYPE_F16);
			var b = CreateTensor(new long[] { 256, 128, 1, 1 }, Structs.GGmlType.GGML_TYPE_F32);

			var graph = Context.NewGraph();
			var result = Context.Conv1D(a, b, 1, 0, 1);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			InitTensorUniform(b, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Conv1D_Default 测试通过");
		}

		/// <summary>
		/// TEST_CONV_2D - 2D卷积测试
		/// </summary>
		public void Test_Conv2D_Default()
		{
			long[] shape = { 64, 64, 1, 1 };
			var a = CreateTensor(shape);
			var b = CreateTensor(new long[] { 64, 64, 1, 1 });

			var graph = Context.NewGraph();
			var result = Context.Conv2d(a, b);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			InitTensorUniform(b, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Conv2D_Default 测试通过");
		}

		// ========== 拼接和排序测试 ==========

		/// <summary>
		/// TEST_CONCAT - 拼接测试
		/// </summary>
		public void Test_Concat_Default()
		{
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);
			var b = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.Concat(a, b, 0);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			InitTensorUniform(b, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Concat_Default 测试通过");
		}

		/// <summary>
		/// TEST_ARGMAX - 参数最大值测试
		/// </summary>
		public void Test_ArgMax_Default()
		{
			// ArgMax 需要矩阵（2D张量），所以使用 {rows, cols, 1, 1} 形状
			long[] shape = { 10, 5, 1, 1 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.ArgMax(a);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_ArgMax_Default 测试通过");
		}

		// ========== 约简操作测试 ==========

		/// <summary>
		/// TEST_SUM - 求和测试
		/// </summary>
		public void Test_Sum_Default()
		{
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.Sum(a);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Sum_Default 测试通过");
		}

		/// <summary>
		/// TEST_UPSCALE - 上采样测试
		/// </summary>
		public void Test_Upscale_Default()
		{
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.Upscale(a, 2, Structs.GGmlScaleMode.GGML_SCALE_MODE_NEAREST);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Upscale_Default 测试通过");
		}

		/// <summary>
		/// TEST_GROUP_NORM - 组归一化测试
		/// </summary>
		public void Test_GroupNorm_Default()
		{
			long[] shape = { 64, 32, 8, 8 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.GroupNorm(a, 32);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_GroupNorm_Default 测试通过");
		}

		// ========== 损失函数测试 ==========

		/// <summary>
		/// TEST_CROSS_ENTROPY_LOSS - 交叉熵损失测试
		/// </summary>
		public void Test_CrossEntropyLoss_Default()
		{
			long[] shape = { 32, 10, 1, 1 };
			var a = CreateTensor(shape);
			var b = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.CrossEntropyLoss(a, b);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			InitTensorUniform(b, 0.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_CrossEntropyLoss_Default 测试通过");
		}

		// ========== 特殊操作测试 ==========

		/// <summary>
		/// TEST_ARANGE - 范围生成测试
		/// </summary>
		public void Test_Arange_Default()
		{
			var graph = Context.NewGraph();
			var result = Context.Arange(0.0f, 10.0f, 1.0f);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Arange_Default 测试通过");
		}

		/// <summary>
		/// TEST_PERMUTE - 排列测试
		/// </summary>
		public void Test_Permute_Default()
		{
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.Permute(a, 0, 2, 1, 3);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Permute_Default 测试通过");
		}

		/// <summary>
		/// TEST_TRANSPOSE - 转置测试
		/// </summary>
		public void Test_Transpose_Default()
		{
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.Transpose(a);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Transpose_Default 测试通过");
		}

		/// <summary>
		/// TEST_RESHAPE - 重塑测试
		/// </summary>
		public void Test_Reshape2D_Default()
		{
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.Reshape2d(a, 10, 60);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Reshape2D_Default 测试通过");
		}

		/// <summary>
		/// TEST_INTERPOLATE - 插值测试
		/// </summary>
		public void Test_Interpolate_Default()
		{
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.Interpolate(a, 20, 10, 4, 3, 0);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Interpolate_Default 测试通过");
		}

		// ========== 基础数学函数测试 (新增) ==========

		/// <summary>
		/// TEST_ABS - 绝对值测试
		/// </summary>
		public void Test_Abs_Default()
		{
			RunSingleTensorTest(
				"Test_Abs_Default",
				new long[] { 10, 5, 4, 3 },
				tensor => Context.Abs(tensor),
				-1.0f, 1.0f
			);
		}

		/// <summary>
		/// TEST_NEG - 取负测试
		/// </summary>
		public void Test_Neg_Default()
		{
			RunSingleTensorTest(
				"Test_Neg_Default",
				new long[] { 10, 5, 4, 3 },
				tensor => Context.Neg(tensor),
				-1.0f, 1.0f
			);
		}

		/// <summary>
		/// TEST_STEP - 阶跃函数测试
		/// </summary>
		public void Test_Step_Default()
		{
			RunSingleTensorTest(
				"Test_Step_Default",
				new long[] { 10, 5, 4, 3 },
				tensor => Context.Step(tensor),
				-1.0f, 1.0f
			);
		}

		/// <summary>
		/// TEST_LOG - 自然对数测试
		/// </summary>
		public void Test_Log_Default()
		{
			RunSingleTensorTest(
				"Test_Log_Default",
				new long[] { 10, 5, 4, 3 },
				tensor => Context.Log(tensor),
				0.1f, 10.0f  // 避免负数和零
			);
		}

		// ========== 取整和限制函数测试 (新增) ==========

		/// <summary>
		/// TEST_CLAMP - 限制值范围测试
		/// </summary>
		public void Test_Clamp_Default()
		{
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.Clamp(a, -0.5f, 0.5f);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Clamp_Default 测试通过");
		}

		/// <summary>
		/// TEST_FLOOR - 向下取整测试
		/// </summary>
		public void Test_Floor_Default()
		{
			RunSingleTensorTest(
				"Test_Floor_Default",
				new long[] { 10, 5, 4, 3 },
				tensor => Context.Floor(tensor),
				-10.0f, 10.0f
			);
		}

		/// <summary>
		/// TEST_CEIL - 向上取整测试
		/// </summary>
		public void Test_Ceil_Default()
		{
			RunSingleTensorTest(
				"Test_Ceil_Default",
				new long[] { 10, 5, 4, 3 },
				tensor => Context.Ceil(tensor),
				-10.0f, 10.0f
			);
		}

		/// <summary>
		/// TEST_ROUND - 四舍五入测试
		/// </summary>
		public void Test_Round_Default()
		{
			RunSingleTensorTest(
				"Test_Round_Default",
				new long[] { 10, 5, 4, 3 },
				tensor => Context.Round(tensor),
				-10.0f, 10.0f
			);
		}

		/// <summary>
		/// TEST_TRUNC - 截断测试
		/// </summary>
		public void Test_Trunc_Default()
		{
			RunSingleTensorTest(
				"Test_Trunc_Default",
				new long[] { 10, 5, 4, 3 },
				tensor => Context.Trunc(tensor),
				-10.0f, 10.0f
			);
		}

		// ========== 归一化变体测试 (新增) ==========

		/// <summary>
		/// TEST_LAYER_NORM - 层归一化测试
		/// </summary>
		public void Test_LayerNorm_Default()
		{
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);
			var weight = CreateTensor(new long[] { shape[1], 1, 1, 1 });  // 对应特征维度
			var bias = CreateTensor(new long[] { shape[1], 1, 1, 1 });

			var graph = Context.NewGraph();
			var result = Context.LayerNorm(a, weight, bias, 1e-5f);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			InitTensorUniform(weight, 0.5f, 1.5f);
			InitTensorUniform(bias, -0.5f, 0.5f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_LayerNorm_Default 测试通过");
		}

		/// <summary>
		/// TEST_L2_NORM - L2归一化测试
		/// </summary>
		public void Test_L2Norm_Default()
		{
			long[] shape = { 64, 5, 4, 3 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.L2Norm(a);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_L2Norm_Default 测试通过");
		}

		// ========== 约简操作测试 (新增) ==========

		/// <summary>
		/// TEST_MEAN - 均值测试
		/// </summary>
		public void Test_Mean_Default()
		{
			RunSingleTensorTest(
				"Test_Mean_Default",
				new long[] { 10, 5, 4, 3 },
				tensor => Context.Mean(tensor),
				-1.0f, 1.0f
			);
		}

		/// <summary>
		/// TEST_SUM_ROWS - 行求和测试
		/// </summary>
		public void Test_SumRows_Default()
		{
			RunSingleTensorTest(
				"Test_SumRows_Default",
				new long[] { 10, 5, 1, 1 },  // 2D 矩阵
				tensor => Context.SumRows(tensor),
				-1.0f, 1.0f
			);
		}

		// ========== 矩阵操作变体测试 (新增) ==========

		/// <summary>
		/// TEST_MUL_MAT_ID - 矩阵乘法带ID测试
		/// </summary>
		public void Test_MulMatId_Default()
		{
			// 对齐上游 test_mul_mat_id:
			// as : [k, m, n_mats]
			// b  : [k, n_used, n]
			// ids: [n_used, n]
			const int k = 32;
			const int m = 32;
			const int n = 32;
			const int nMats = 2;
			const int nUsed = 2;

			long[] shapeA = { k, m, nMats };
			long[] shapeB = { k, nUsed, n };
			long[] shapeIds = { nUsed, n };

			var @as = CreateTensor(shapeA);
			var b = CreateTensor(shapeB);
			var ids = CreateTensor(shapeIds, Structs.GGmlType.GGML_TYPE_I32);

			var graph = Context.NewGraph();
			var result = Context.MulMatId(@as, ids, b);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(@as, -1.0f, 1.0f);
			InitTensorUniform(b, -1.0f, 1.0f);
			InitTensorInt(ids, 0, nMats);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_MulMatId_Default 测试通过");
		}

		// ========== 卷积变体测试 (新增) ==========

		/// <summary>
		/// TEST_CONV_DEPTHWISE_1D - 1D深度卷积测试
		/// </summary>
		public void Test_ConvDepthwise1D_Default()
		{
			// 深度卷积：ggml 接口要求卷积核在前，输入数据在后
			// input : [input_length, channels, batch]
			// kernel: [kernel_length, 1, channels]
			long[] inputShape = { 256, 128, 1 };
			long[] kernelShape = { 3, 1, 128 };  // kernel_size=3, channels=128

			var input = CreateTensor(inputShape, Structs.GGmlType.GGML_TYPE_F32);
			var kernel = CreateTensor(kernelShape, Structs.GGmlType.GGML_TYPE_F16);

			var graph = Context.NewGraph();
			var result = Context.ConvDepthwise1D(kernel, input, 1, 0, 1);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(input, -1.0f, 1.0f);
			InitTensorUniform(kernel, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_ConvDepthwise1D_Default 测试通过");
		}

		/// <summary>
		/// TEST_CONV_DEPTHWISE_2D - 2D深度卷积测试
		/// </summary>
		public void Test_ConvDepthwise2D_Default()
		{
			// 深度卷积：ggml 接口要求卷积核在前，输入数据在后
			// input : [height, width, channels, batch]
			// kernel: [kernel_height, kernel_width, 1, channels]
			long[] inputShape = { 64, 64, 32, 1 };   // height=64, width=64, channels=32
			long[] kernelShape = { 3, 3, 1, 32 };    // kernel_h=3, kernel_w=3, channels=32

			var input = CreateTensor(inputShape);
			var kernel = CreateTensor(kernelShape);

			var graph = Context.NewGraph();
			var result = Context.ConvDepthwise2D(kernel, input, 1, 1, 0, 0, 1, 1);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(input, -1.0f, 1.0f);
			InitTensorUniform(kernel, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_ConvDepthwise2D_Default 测试通过");
		}

		/// <summary>
		/// TEST_IM2COL - 图像转列测试
		/// </summary>
		public void Test_Im2Col_Default()
		{
			long[] inputShape = { 64, 64, 1, 1 };
			long[] kernelShape = { 3, 3, 1, 1 };

			var input = CreateTensor(inputShape);
			var kernel = CreateTensor(kernelShape);

			var graph = Context.NewGraph();
			var result = Context.Im2Col(kernel, input, 1, 1, 0, 0, 1, 1, true, Structs.GGmlType.GGML_TYPE_F32);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(input, -1.0f, 1.0f);
			InitTensorUniform(kernel, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Im2Col_Default 测试通过");
		}

		// ========== 张量操作测试 (新增) ==========

		/// <summary>
		/// TEST_OUT_PROD - 外积测试
		/// </summary>
		public void Test_OutProd_Default()
		{
			// 对齐上游 test_out_prod 的基础情况:
			// a: [m, k], b: [n, k]，要求 ne[1] 相同
			long[] shapeA = { 32, 32, 1, 1 };
			long[] shapeB = { 64, 32, 1, 1 };
			var a = CreateTensor(shapeA);
			var b = CreateTensor(shapeB);

			var graph = Context.NewGraph();
			var result = Context.OutProd(a, b);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			InitTensorUniform(b, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_OutProd_Default 测试通过");
		}

		/// <summary>
		/// TEST_ACC - 累加测试
		/// </summary>
		public void Test_Acc_Default()
		{
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);
			var b = CreateTensor(new long[] { 5, 5, 4, 3 });

			var graph = Context.NewGraph();
			var result = Context.Acc(a, b, 0, 0, 0, 0);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			InitTensorUniform(b, 0.1f, 0.5f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Acc_Default 测试通过");
		}

		/// <summary>
		/// TEST_SET - 设置测试
		/// </summary>
		public void Test_Set_Default()
		{
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);
			var b = CreateTensor(new long[] { 5, 5, 4, 3 });

			var graph = Context.NewGraph();
			var result = Context.Set(a, b, 0, 0, 0, 0);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			InitTensorUniform(b, 0.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Set_Default 测试通过");
		}

		/// <summary>
		/// TEST_PAD - 填充测试
		/// </summary>
		public void Test_Pad_Default()
		{
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.Pad(a, 1, 1, 1, 1);  // 各方向填充1
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Pad_Default 测试通过");
		}

		/// <summary>
		/// TEST_PAD_REFLECT_1D - 1D反射填充测试
		/// </summary>
		public void Test_PadReflect1D_Default()
		{
			long[] shape = { 10, 1, 1, 1 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.PadReflect1D(a, 2, 2);  // 左右各填充2
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_PadReflect1D_Default 测试通过");
		}

		/// <summary>
		/// TEST_ROLL - 张量滚动测试
		/// </summary>
		public void Test_Roll_Default()
		{
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.Roll(a, 1, 0, 0, 0);  // 在第0维滚动1位
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Roll_Default 测试通过");
		}

		/// <summary>
		/// TEST_CUMSUM - 累积求和测试
		/// </summary>
		public void Test_CumSum_Default()
		{
			RunSingleTensorTest(
				"Test_CumSum_Default",
				new long[] { 10, 5, 4, 3 },
				tensor => Context.CumSum(tensor),
				-1.0f, 1.0f
			);
		}

		/// <summary>
		/// TEST_TIMESTEP_EMBEDDING - 时间步嵌入测试
		/// </summary>
		public void Test_TimeStepEmbedding_Default()
		{
			long[] shape = { 10, 1, 1, 1 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.TimeStepEmbedding(a, 32, 10000);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, 0.0f, 100.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_TimeStepEmbedding_Default 测试通过");
		}

		/// <summary>
		/// TEST_ARGSORT - 参数排序测试
		/// </summary>
		public void Test_ArgSort_Default()
		{
			long[] shape = { 10, 5, 1, 1 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.ArgSort(a, InternalStructs.ggml_sort_order.GGML_SORT_ORDER_ASC);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -10.0f, 10.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_ArgSort_Default 测试通过");
		}

		/// <summary>
		/// TEST_TOP_K - Top-K选择测试
		/// </summary>
		public void Test_TopK_Default()
		{
			long[] shape = { 10, 5, 1, 1 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.TopK(a, 3);  // 选择Top 3
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -10.0f, 10.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_TopK_Default 测试通过");
		}

		/// <summary>
		/// TEST_DIAG - 对角矩阵测试
		/// </summary>
		public void Test_Diag_Default()
		{
			RunSingleTensorTest(
				"Test_Diag_Default",
				new long[] { 10, 1, 1, 1 },
				tensor => Context.Diag(tensor),
				-1.0f, 1.0f
			);
		}

		/// <summary>
		/// TEST_TRI - 三角矩阵测试
		/// </summary>
		public void Test_Tri_Default()
		{
			long[] shape = { 10, 10, 1, 1 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.Tri(a, InternalStructs.ggml_tri_type.GGML_TRIANGULAR_LOWER);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Tri_Default 测试通过");
		}

		/// <summary>
		/// TEST_FILL - 填充测试
		/// </summary>
		public void Test_Fill_Default()
		{
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.Fill(a, 0.5f);  // 填充为0.5
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_Fill_Default 测试通过");
		}

		/// <summary>
		/// TEST_COUNT_EQUAL - 计数相等元素测试
		/// </summary>
		public void Test_CountEqual_Default()
		{
			long[] shape = { 10, 5, 1, 1 };
			var a = CreateTensor(shape);
			var b = CreateTensor(shape);

			var graph = Context.NewGraph();
			var aArgMax = Context.ArgMax(a);
			var bArgMax = Context.ArgMax(b);
			var result = Context.CountEqual(aArgMax, bArgMax);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			InitTensorUniform(b, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_CountEqual_Default 测试通过");
		}

		// ========== 高级操作测试组 (新增) ==========

		/// <summary>
		/// TEST_XIELU - XieLU激活函数测试
		/// </summary>
		public void Test_XieLU_Default()
		{
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.XieLU(a, 4.0f, 20.0f, 0.5f, 0.0000001f);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_XieLU_Default 测试通过");
		}

		/// <summary>
		/// TEST_SOFTCAP - Soft Capping测试
		/// </summary>
		public void Test_SoftCap_Default()
		{
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);

			var graph = Context.NewGraph();
			// SoftCap = scale(tanh(scale(a, 1/softcap)), softcap)
			var scaled1 = Context.Scale(a, 1.0f / 30.0f);
			var tanh = Context.Tanh(scaled1);
			var result = Context.Scale(tanh, 30.0f);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -10.0f, 10.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_SoftCap_Default 测试通过");
		}

		/// <summary>
		/// TEST_OPT_STEP_ADAMW - AdamW优化器测试
		/// </summary>
		public void Test_OptStepAdamW_Default()
		{
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);
			var grad = CreateTensor(shape);
			var gradM = CreateTensor(shape);
			var gradV = CreateTensor(shape);
			var adamWParams = CreateTensor(new long[] { 7, 1, 1, 1 });
			Context.SetParam(a);

			var graph = Context.NewGraph();
			var result = Context.OptStepAdamW(a, grad, gradM, gradV, adamWParams);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			// 初始化参数和梯度
			InitTensorUniform(a, 0.0f, 1.0f);
			InitTensorUniform(grad, 0.0f, 1.0f);
			InitTensorUniform(gradM, 0.0f, 1.0f);
			InitTensorUniform(gradV, 0.0f, 1.0f);  // 需要非负值
			InitTensorUniform(adamWParams, 0.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_OptStepAdamW_Default 测试通过");
		}

		/// <summary>
		/// TEST_OPT_STEP_SGD - SGD优化器测试
		/// </summary>
		public void Test_OptStepSGD_Default()
		{
			long[] shape = { 10, 5, 4, 3 };
			var a = CreateTensor(shape);
			var grad = CreateTensor(shape);
			var sgdParams = CreateTensor(new long[] { 2, 1, 1, 1 });  // learning rate and momentum
			Context.SetParam(a);

			var graph = Context.NewGraph();
			var result = Context.OptStepSGD(a, grad, sgdParams);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			InitTensorUniform(grad, -0.1f, 0.1f);
			InitTensorUniform(sgdParams, 0.0f, 0.1f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_OptStepSGD_Default 测试通过");
		}

		/// <summary>
		/// TEST_SSM_CONV - SSM卷积测试
		/// </summary>
		public void Test_SsmConv_Default()
		{
			long[] shape = { 32, 32, 1, 1 };
			var s = CreateTensor(shape);
			var x = CreateTensor(shape);
			var c = CreateTensor(shape);
			var sq = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.SsmConv(s, x, c, sq);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(s, -1.0f, 1.0f);
			InitTensorUniform(x, -1.0f, 1.0f);
			InitTensorUniform(c, -1.0f, 1.0f);
			InitTensorUniform(sq, 0.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_SsmConv_Default 测试通过");
		}

		/// <summary>
		/// TEST_SSM_SCAN - SSM扫描测试
		/// </summary>
		public void Test_SsmScan_Default()
		{
			const int dState = 16;
			const int headDim = 1;
			const int nHead = 32;
			const int nGroup = 1;
			const int nSeqTokens = 32;
			const int nSeqs = 4;

			var s = CreateTensor(new long[] { dState, headDim, nHead, nSeqs });
			var x = CreateTensor(new long[] { headDim, nHead, nSeqTokens, nSeqs });
			var dt = CreateTensor(new long[] { nHead, nSeqTokens, nSeqs, 1 });
			var A = CreateTensor(new long[] { dState, nHead, 1, 1 });
			var B = CreateTensor(new long[] { dState, nGroup, nSeqTokens, nSeqs });
			var C = CreateTensor(new long[] { dState, nGroup, nSeqTokens, nSeqs });
			var ids = CreateTensor(new long[] { nSeqs, 1, 1, 1 }, Structs.GGmlType.GGML_TYPE_I32);

			var graph = Context.NewGraph();
			var result = Context.SsmScan(s, x, dt, A, B, C, ids);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(s, -1.0f, 1.0f);
			InitTensorUniform(x, -1.0f, 1.0f);
			InitTensorUniform(dt, 0.0f, 1.0f);
			InitTensorUniform(A, -1.0f, 1.0f);
			InitTensorUniform(B, -1.0f, 1.0f);
			InitTensorUniform(C, -1.0f, 1.0f);
			ids.SetData(new[] { 0, 1, 2, 3 });
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_SsmScan_Default 测试通过");
		}

		/// <summary>
		/// TEST_RWKV_WKV6 - RWKV v6测试
		/// </summary>
		public void Test_RwkvWkv6_Default()
		{
			const int headCount = 32;
			const int headSize = 64;
			const int nSeqTokens = 1;
			const int nSeqs = 1;
			const int nTokens = nSeqTokens * nSeqs;

			var k = CreateTensor(new long[] { headSize, headCount, nTokens, 1 });
			var v = CreateTensor(new long[] { headSize, headCount, nTokens, 1 });
			var r = CreateTensor(new long[] { headSize, headCount, nTokens, 1 });
			var tf = CreateTensor(new long[] { headSize, headCount, 1, 1 });
			var td = CreateTensor(new long[] { headSize, headCount, nTokens, 1 });
			var s = CreateTensor(new long[] { headSize * headSize * headCount, nSeqs, 1, 1 });

			var graph = Context.NewGraph();
			var result = Context.RwkvWkv6(k, v, r, tf, td, s);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(k, -1.0f, 1.0f);
			InitTensorUniform(v, -1.0f, 1.0f);
			InitTensorUniform(r, -1.0f, 1.0f);
			InitTensorUniform(tf, -1.0f, 1.0f);
			InitTensorUniform(td, -1.0f, 1.0f);
			InitTensorUniform(s, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_RwkvWkv6_Default 测试通过");
		}

		/// <summary>
		/// TEST_RWKV_WKV7 - RWKV v7测试
		/// </summary>
		public void Test_RwkvWkv7_Default()
		{
			const int headCount = 32;
			const int headSize = 64;
			const int nSeqTokens = 1;
			const int nSeqs = 1;
			const int nTokens = nSeqTokens * nSeqs;

			var r = CreateTensor(new long[] { headSize, headCount, nTokens, 1 });
			var w = CreateTensor(new long[] { headSize, headCount, nTokens, 1 });
			var k = CreateTensor(new long[] { headSize, headCount, nTokens, 1 });
			var v = CreateTensor(new long[] { headSize, headCount, nTokens, 1 });
			var a = CreateTensor(new long[] { headSize, headCount, nTokens, 1 });
			var b = CreateTensor(new long[] { headSize, headCount, nTokens, 1 });
			var s = CreateTensor(new long[] { headSize * headSize * headCount, nSeqs, 1, 1 });

			var graph = Context.NewGraph();
			var result = Context.RwkvWkv7(r, w, k, v, a, b, s);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(r, -1.0f, 1.0f);
			InitTensorUniform(w, -1.0f, 1.0f);
			InitTensorUniform(k, -1.0f, 1.0f);
			InitTensorUniform(v, -1.0f, 1.0f);
			InitTensorUniform(a, -1.0f, 1.0f);
			InitTensorUniform(b, -1.0f, 1.0f);
			InitTensorUniform(s, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_RwkvWkv7_Default 测试通过");
		}

		/// <summary>
		/// TEST_FLASH_ATTN - 基础Flash Attention测试
		/// </summary>
		[BackendRequirement(BackendType.CUDA)]
		public void Test_FlashAttn_Basic()
		{
			long[] shape = { 32, 32, 1, 1 };
			var q = CreateTensor(shape);
			var k = CreateTensor(shape);
			var v = CreateTensor(shape);

			var graph = Context.NewGraph();
			var result = Context.FlashAttn(q, k, v, false);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(q, -1.0f, 1.0f);
			InitTensorUniform(k, -1.0f, 1.0f);
			InitTensorUniform(v, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_FlashAttn_Basic 测试通过");
		}

		// ========== 反向传播测试 ==========

		/// <summary>
		/// TEST_SILU_BACK - SiLU反向传播测试
		/// </summary>
		public void Test_SiLUBack_Default()
		{
			long[] shape = { 64, 10, 5, 4 };
			var a = CreateTensor(shape);
			var b = CreateTensor(shape);  // 梯度输入

			var graph = Context.NewGraph();
			var result = Context.SiLUBack(a, b);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			InitTensorUniform(b, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_SiLUBack_Default 测试通过");
		}

		/// <summary>
		/// TEST_SOFT_MAX_BACK - SoftMax反向传播测试
		/// </summary>
		public void Test_SoftMaxBack_Default()
		{
			long[] shape = { 32, 32, 1, 1 };
			var a = CreateTensor(shape);
			var b = CreateTensor(shape);  // 梯度输入

			var graph = Context.NewGraph();
			var result = Context.SoftMaxBack(a, b);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			InitTensorUniform(b, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_SoftMaxBack_Default 测试通过");
		}

		/// <summary>
		/// TEST_RMS_NORM_BACK - RMS归一化反向传播测试
		/// </summary>
		public void Test_RmsNormBack_Default()
		{
			long[] shape = { 64, 10, 5, 4 };
			var a = CreateTensor(shape);
			var b = CreateTensor(shape);  // 梯度输入

			var graph = Context.NewGraph();
			var result = Context.RmsNormBack(a, b, 1e-5f);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			InitTensorUniform(b, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_RmsNormBack_Default 测试通过");
		}

		/// <summary>
		/// TEST_REPEAT_BACK - Repeat反向传播测试
		/// </summary>
		public void Test_RepeatBack_Default()
		{
			// 对齐上游 test_repeat_back:
			// 第一个参数是重复后的梯度张量，第二个参数是原始目标形状
			long[] repeatedShape = { 16, 6, 4, 2 };
			long[] targetShape = { 8, 6, 4, 2 };
			var a = CreateTensor(repeatedShape);
			var b = CreateTensor(targetShape);

			var graph = Context.NewGraph();
			var result = Context.RepeatBack(a, b);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			InitTensorUniform(a, -1.0f, 1.0f);
			InitTensorUniform(b, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_RepeatBack_Default 测试通过");
		}

		/// <summary>
		/// TEST_CROSS_ENTROPY_LOSS_BACK - 交叉熵损失反向传播测试
		/// </summary>
		public void Test_CrossEntropyLossBack_Default()
		{
			long[] shape = { 32, 10, 1, 1 };
			var grad = CreateTensor(new long[] { 1, 1, 1, 1 });
			var logits = CreateTensor(shape);
			var labelsInput = CreateTensor(shape);

			var graph = Context.NewGraph();
			var labels = Context.SoftMax(labelsInput);
			var result = Context.CrossEntropyLossBack(grad, logits, labels);
			graph.BuildForwardExpend(result);
			Context.BackendAllocContextTensors(Backend);

			grad.SetData(1.0f, 0, 0, 0, 0);
			InitTensorUniform(logits, 0.1f, 0.9f);
			InitTensorUniform(labelsInput, -1.0f, 1.0f);
			graph.BackendCompute(Backend);

			Console.WriteLine($"✓ Test_CrossEntropyLossBack_Default 测试通过");
		}
	}
}

using System;
using System.Collections.Generic;
using GGMLSharp;

namespace GGMLSharp.Utils
{
	/// <summary>
	/// GGML 操作工具类 - 提供 GGML 操作相关的实用功能
	/// </summary>
	public static class GGmlOpsUtils
	{
		/// <summary>
		/// 列出所有 GGML 操作名称
		/// 对应 C++ test-backend-ops.cpp 中的 list_all_ops() 函数
		/// </summary>
		public static void ListAllOps()
			{
			Console.WriteLine("GGML operations:");
			var allOps = new SortedSet<string>();

			// 收集所有 GGML_OP 操作
			// 从头文件中知道 GGML_OP_COUNT 大约是 100+
			for (int i = 1; i < 100; i++)
			{
				try
				{
					string opName = Native.ggml_op_name((InternalStructs.ggml_op)i);
					if (!string.IsNullOrEmpty(opName) && opName != $"UNKNOWN_OP_{i}")
					{
						allOps.Add(opName);
					}
				}
				catch
				{
					break; // 达到操作数量限制
				}
			}

			// 收集所有 GGML_UNARY_OP 操作
			for (int i = 0; i < 50; i++)
			{
				try
				{
					string opName = Native.ggml_unary_op_name((InternalStructs.ggml_unary_op)i);
					if (!string.IsNullOrEmpty(opName) && opName != $"UNKNOWN_UNARY_OP_{i}")
					{
						allOps.Add(opName);
					}
				}
				catch
				{
					break; // 达到操作数量限制
				}
			}

			// 收集所有 GGML_GLU_OP 操作
			for (int i = 0; i < 20; i++)
			{
				try
				{
					string opName = Native.ggml_glu_op_name((InternalStructs.ggml_glu_op)i);
					if (!string.IsNullOrEmpty(opName) && opName != $"UNKNOWN_GLU_OP_{i}")
					{
						allOps.Add(opName);
					}
				}
				catch
				{
					break; // 达到操作数量限制
				}
			}

			// 打印所有操作
			foreach (var op in allOps)
			{
				Console.WriteLine($"  {op}");
			}

			Console.WriteLine($"\nTotal: {allOps.Count} operations");
		}

		/// <summary>
		/// 获取所有 GGML 操作名称列表
		/// </summary>
		/// <returns>所有操作名称的排序列表</returns>
		public static List<string> GetAllOpNames()
		{
			var allOps = new SortedSet<string>();

			// 收集所有 GGML_OP 操作
			for (int i = 1; i < (int)InternalStructs.ggml_op.GGML_OP_COUNT; i++)
			{
				string opName = Native.ggml_op_name((InternalStructs.ggml_op)i);
				if (!string.IsNullOrEmpty(opName))
				{
					allOps.Add(opName);
				}
			}

			// 收集所有 GGML_UNARY_OP 操作
			for (int i = 0; i < (int)InternalStructs.ggml_unary_op.GGML_UNARY_OP_COUNT; i++)
			{
				string opName = Native.ggml_unary_op_name((InternalStructs.ggml_unary_op)i);
				if (!string.IsNullOrEmpty(opName))
				{
					allOps.Add(opName);
				}
			}

			// 收集所有 GGML_GLU_OP 操作
			for (int i = 0; i < (int)InternalStructs.ggml_glu_op.GGML_GLU_OP_COUNT; i++)
			{
				string opName = Native.ggml_glu_op_name((InternalStructs.ggml_glu_op)i);
				if (!string.IsNullOrEmpty(opName))
				{
					allOps.Add(opName);
				}
			}

			return new List<string>(allOps);
		}

		/// <summary>
		/// 检查特定操作是否被支持
		/// </summary>
		/// <param name="opName">要检查的操作名称</param>
		/// <returns>如果操作存在返回 true，否则返回 false</returns>
		public static bool IsOpSupported(string opName)
		{
			if (string.IsNullOrEmpty(opName))
			{
				return false;
			}

			var allOps = GetAllOpNames();
			return allOps.Contains(opName);
		}

		/// <summary>
		/// 按类型分组获取操作名称
		/// </summary>
		/// <returns>按操作类型分组的字典</returns>
		public static Dictionary<string, List<string>> GetOpsByType()
		{
			var result = new Dictionary<string, List<string>>
			{
				["GGML_OP"] = new List<string>(),
				["GGML_UNARY_OP"] = new List<string>(),
				["GGML_GLU_OP"] = new List<string>()
			};

			// 收集 GGML_OP 操作
			for (int i = 1; i < (int)InternalStructs.ggml_op.GGML_OP_COUNT; i++)
			{
				string opName = Native.ggml_op_name((InternalStructs.ggml_op)i);
				if (!string.IsNullOrEmpty(opName))
				{
					result["GGML_OP"].Add(opName);
				}
			}

			// 收集 GGML_UNARY_OP 操作
			for (int i = 0; i < (int)InternalStructs.ggml_unary_op.GGML_UNARY_OP_COUNT; i++)
			{
				string opName = Native.ggml_unary_op_name((InternalStructs.ggml_unary_op)i);
				if (!string.IsNullOrEmpty(opName))
				{
					result["GGML_UNARY_OP"].Add(opName);
				}
			}

			// 收集 GGML_GLU_OP 操作
			for (int i = 0; i < (int)InternalStructs.ggml_glu_op.GGML_GLU_OP_COUNT; i++)
			{
				string opName = Native.ggml_glu_op_name((InternalStructs.ggml_glu_op)i);
				if (!string.IsNullOrEmpty(opName))
				{
					result["GGML_GLU_OP"].Add(opName);
				}
			}

			return result;
		}
	}
}

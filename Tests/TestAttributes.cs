using System;

namespace GGMLSharp.Tests
{
	/// <summary>
	/// 后端类型枚举
	/// </summary>
	public enum BackendType
	{
		CPU,
		CUDA,
		Both
	}

	/// <summary>
	/// 后端需求属性 - 用于标记测试需要特定后端
	/// 使用方法: [BackendRequirement(BackendType.CUDA)]
	/// </summary>
	[AttributeUsage(AttributeTargets.Method)]
	public class BackendRequirementAttribute : Attribute
	{
		public BackendType RequiredBackend { get; }

		public BackendRequirementAttribute(BackendType backend)
		{
			RequiredBackend = backend;
		}
	}
}

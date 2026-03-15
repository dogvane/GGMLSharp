using System;

namespace GGMLSharp
{
    /// <summary>
    /// GGML 配置类，用于设置运行时选项
    /// </summary>
    public static class GGMLConfig
    {
        /// <summary>
        /// 设置 GGML 后端
        /// </summary>
        /// <param name="backend">后端类型: "cpu" 或 "cuda"</param>
        /// <exception cref="ArgumentException">当后端类型不支持时抛出</exception>
        /// <exception cref="InvalidOperationException">当无法加载后端 DLL 时抛出</exception>
        /// <remarks>
        /// 此方法必须在调用任何 GGML 功能之前调用。
        /// CPU 后端使用 ggml-cpu.dll
        /// CUDA 后端使用 ggml-cuda.dll（需要 NVIDIA GPU 和 CUDA 驱动）
        ///
        /// 示例:
        /// <code>
        /// // 使用 CPU 后端（默认）
        /// GGMLConfig.SetBackend("cpu");
        ///
        /// // 使用 CUDA 后端
        /// GGMLConfig.SetBackend("cuda");
        /// </code>
        /// </remarks>
        public static void SetBackend(string backend)
        {
            Native.SetBackend(backend);
        }

        /// <summary>
        /// 获取当前使用的后端名称
        /// </summary>
        /// <returns>后端名称 ("cpu" 或 "cuda")</returns>
        public static string GetBackendName()
        {
            return Native.GetBackendName();
        }
    }
}

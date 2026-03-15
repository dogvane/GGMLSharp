using System;
using System.Runtime.InteropServices;

namespace GGMLSharp
{
    /// <summary>
    /// GGML 原生代码断言失败时抛出的异常
    /// </summary>
    public class GGmlAssertException : Exception
    {
        public GGmlAssertException(string message) : base(message) { }
    }

    /// <summary>
    /// GGML Abort 回调委托
    /// </summary>
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate void GGmlAbortCallback([MarshalAs(UnmanagedType.LPStr)] string errorMessage);

    /// <summary>
    /// GGML 断言处理器 - 仅在 Debug 模式下启用 C 堆栈输出
    /// </summary>
    internal static class GGmlAssertHandler
    {
        private static GGmlAbortCallback? s_abortCallback;

        /// <summary>
        /// 初始化断言处理器（仅在 Debug 模式下）
        /// </summary>
        public static void Initialize()
        {
#if DEBUG
            if (s_abortCallback == null)
            {
                s_abortCallback = new GGmlAbortCallback(OnAbort);
                // 将委托转换为函数指针
                IntPtr callbackPtr = Marshal.GetFunctionPointerForDelegate(s_abortCallback);
                Native.ggml_set_abort_callback(callbackPtr);
            }
#endif
        }

        /// <summary>
        /// 原生 abort 回调处理函数（仅 Debug 模式）
        /// </summary>
#if DEBUG
        private static void OnAbort(string errorMessage)
        {
            // 打印错误消息
            Console.Error.WriteLine($"[GGML ASSERT FAILED] {errorMessage}");

            // 调用 C 端的堆栈打印函数（如果可用）
            try
            {
                Native.ggml_print_backtrace();
            }
            catch { }
        }
#endif
    }
}

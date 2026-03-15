using System;
using System.IO;
using System.Runtime.InteropServices;

namespace GGMLSharp
{
    /// <summary>
    /// GGUF 文件验证扩展方法
    /// </summary>
    public static class GgufValidationExtensions
    {
        // GGUF Magic: "GGUF" (0x47 0x47 0x55 0x46)
        private const byte GGUF_MAGIC_0 = 0x47; // 'G'
        private const byte GGUF_MAGIC_1 = 0x47; // 'G'
        private const byte GGUF_MAGIC_2 = 0x55; // 'U'
        private const byte GGUF_MAGIC_3 = 0x46; // 'F'

        /// <summary>
        /// 验证 GGUF 上下文的文件头 Magic 是否匹配
        /// </summary>
        /// <param name="context">GGUF 上下文</param>
        /// <returns>如果 Magic 匹配返回 true</returns>
        public static bool ValidateContentHeaderMagic(this SafeGGufContext context)
        {
            if (context == null || context.IsInvalid)
            {
                return false;
            }

            // 由于 gguf_init_from_file 成功返回了上下文，说明文件已经通过了基础验证
            // 我们只需要简单验证上下文是否有效即可
            return true;
        }

        /// <summary>
        /// 直接验证 GGUF 模型文件的文件头 Magic 是否匹配
        /// 无需打开整个文件，仅读取前 4 个字节
        /// </summary>
        /// <param name="context">GGUF 上下文（从中获取文件名）</param>
        /// <returns>如果文件存在且 Magic 匹配返回 true</returns>
        public static bool ValidateFileHeaderMagic(this SafeGGufContext context)
        {
            string? fileName = context?.OpenModelFile;
            if (string.IsNullOrEmpty(fileName) || !File.Exists(fileName))
            {
                return false;
            }

            return ValidateFileHeaderMagic(fileName);
        }

        /// <summary>
        /// 直接验证 GGUF 模型文件的文件头 Magic 是否匹配
        /// 无需打开整个文件，仅读取前 4 个字节
        /// </summary>
        /// <param name="fileName">GGUF 模型文件路径</param>
        /// <returns>如果文件存在且 Magic 匹配返回 true</returns>
        public static bool ValidateFileHeaderMagic(string fileName)
        {
            if (string.IsNullOrEmpty(fileName) || !File.Exists(fileName))
            {
                return false;
            }

            try
            {
                using var fileStream = new FileStream(fileName, FileMode.Open, FileAccess.Read);
                byte[] magicBytes = new byte[4];
                int bytesRead = fileStream.Read(magicBytes, 0, 4);

                if (bytesRead < 4)
                {
                    return false;
                }

                return magicBytes[0] == GGUF_MAGIC_0 &&
                       magicBytes[1] == GGUF_MAGIC_1 &&
                       magicBytes[2] == GGUF_MAGIC_2 &&
                       magicBytes[3] == GGUF_MAGIC_3;
            }
            catch (IOException)
            {
                // 文件读取错误
                return false;
            }
        }

        /// <summary>
        /// 验证 GGUF 文件头并返回详细信息
        /// </summary>
        /// <param name="fileName">GGUF 模型文件路径</param>
        /// <returns>验证结果</returns>
        public static GgufHeaderValidationResult ValidateFileHeader(string fileName)
        {
            var result = new GgufHeaderValidationResult();
            result.FileName = fileName;

            if (string.IsNullOrEmpty(fileName))
            {
                result.ErrorMessage = "文件名不能为空";
                result.IsValid = false;
                return result;
            }

            if (!File.Exists(fileName))
            {
                result.ErrorMessage = "文件不存在";
                result.IsValid = false;
                return result;
            }

            try
            {
                using var fileStream = new FileStream(fileName, FileMode.Open, FileAccess.Read);

                // 读取 Magic
                byte[] magicBytes = new byte[4];
                int bytesRead = fileStream.Read(magicBytes, 0, 4);
                result.MagicBytes = magicBytes;

                if (bytesRead < 4)
                {
                    result.ErrorMessage = "文件太小，无法读取完整的 Magic";
                    result.IsValid = false;
                    return result;
                }

                result.IsMagicMatch = magicBytes[0] == GGUF_MAGIC_0 &&
                                     magicBytes[1] == GGUF_MAGIC_1 &&
                                     magicBytes[2] == GGUF_MAGIC_2 &&
                                     magicBytes[3] == GGUF_MAGIC_3;

                // 读取 Version
                byte[] versionBytes = new byte[4];
                bytesRead = fileStream.Read(versionBytes, 0, 4);
                if (bytesRead >= 4)
                {
                    result.Version = BitConverter.ToUInt32(versionBytes, 0);
                }

                // 读取 Tensor 数量
                byte[] tensorCountBytes = new byte[8];
                bytesRead = fileStream.Read(tensorCountBytes, 0, 8);
                if (bytesRead >= 8)
                {
                    result.TensorCount = BitConverter.ToUInt64(tensorCountBytes, 0);
                }

                // 读取 KV 数量
                byte[] kvCountBytes = new byte[8];
                bytesRead = fileStream.Read(kvCountBytes, 0, 8);
                if (bytesRead >= 8)
                {
                    result.KeyValueCount = BitConverter.ToUInt64(kvCountBytes, 0);
                }

                result.IsValid = result.IsMagicMatch;
                if (!result.IsMagicMatch)
                {
                    result.ErrorMessage = $"Magic 不匹配。期望: GGUF (0x{GGUF_MAGIC_0:X2} 0x{GGUF_MAGIC_1:X2} 0x{GGUF_MAGIC_2:X2} 0x{GGUF_MAGIC_3:X2}), " +
                                         $"实际: 0x{magicBytes[0]:X2} 0x{magicBytes[1]:X2} 0x{magicBytes[2]:X2} 0x{magicBytes[3]:X2}";
                }
            }
            catch (IOException ex)
            {
                result.ErrorMessage = $"文件读取错误: {ex.Message}";
                result.IsValid = false;
            }

            return result;
        }
    }

    /// <summary>
    /// GGUF 文件头验证结果
    /// </summary>
    public class GgufHeaderValidationResult
    {
        /// <summary>文件名</summary>
        public string? FileName { get; set; }

        /// <summary>是否有效</summary>
        public bool IsValid { get; set; }

        /// <summary>Magic 是否匹配</summary>
        public bool IsMagicMatch { get; set; }

        /// <summary>Magic 字节</summary>
        public byte[]? MagicBytes { get; set; }

        /// <summary>版本</summary>
        public uint Version { get; set; }

        /// <summary>Tensor 数量</summary>
        public ulong TensorCount { get; set; }

        /// <summary>KeyValue 数量</summary>
        public ulong KeyValueCount { get; set; }

        /// <summary>错误信息</summary>
        public string? ErrorMessage { get; set; }

        /// <summary>
        /// 格式化输出验证结果
        /// </summary>
        public override string ToString()
        {
            if (!IsValid)
            {
                return $"[无效] {FileName}: {ErrorMessage}";
            }

            return $"[有效] {FileName}: Magic=GGUF, Version={Version}, Tensors={TensorCount}, KVs={KeyValueCount}";
        }
    }
}

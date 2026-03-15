using System;
using System.Runtime.InteropServices;
using System.Text;

namespace GGMLSharp
{
	public unsafe class SafeGGufContext : SafeGGmlHandleBase, IDisposable
	{
		private bool IsInitialized => handle != IntPtr.Zero;

		public uint Version
		{
			get
			{
				ThrowIfNotInitialized();
				return (uint)Native.gguf_get_version(this);
			}
		}

		public ulong KeyValuesCount
		{
			get
			{
				ThrowIfNotInitialized();
				return (ulong)Native.gguf_get_n_kv(this);
			}
		}

		public ulong TensorsCount
		{
			get
			{
				ThrowIfNotInitialized();
				return (ulong)Native.gguf_get_n_tensors(this);
			}
		}

		public ulong Alignment
		{
			get
			{
				ThrowIfNotInitialized();
				return Native.gguf_get_alignment(this);
			}
		}

		public ulong DataOffset
		{
			get
			{
				ThrowIfNotInitialized();
				return Native.gguf_get_data_offset(this);
			}
		}

		public SafeGGufContext()
		{
			this.handle = IntPtr.Zero;
		}

		public SafeGGufContext(IntPtr handle, bool ownsHandle) : base(handle, ownsHandle)
		{
			SetHandle(handle);
		}

		public static SafeGGufContext Initialize()
		{
			return Native.gguf_init_empty();
		}

#if DEBUG
		public static SafeGGufContext UnittestInitialize()
		{
			return Native.unittest_gguf_init();
		}
#endif

		public string OpenModelFile { get; internal set; } = string.Empty;

		internal new IntPtr DangerousGetHandle()
		{
			return handle;
		}

		public static SafeGGufContext InitFromFile(string fileName, SafeGGmlContext ggmlContext, bool noAlloc)
		{
			if (!File.Exists(fileName))
			{
				throw new FileNotFoundException($"File not found: {fileName}");
			}

			var ret = Native.gguf_init_from_file(fileName, ggmlContext, noAlloc);
			ret.OpenModelFile = fileName;
			return ret;
		}


		private void ThrowIfNotInitialized()
		{
			if (!IsInitialized)
			{
				throw new ObjectDisposedException("Not initialized or disposed");
			}
		}

		// 注意：KeyValues 属性不再直接访问内存，因为 gguf_context 内部使用了 std::vector
		// 如果需要遍历 key-value，建议使用 gguf_get_key, gguf_get_val_* 等 API 逐个获取

		// 注意：GGufTensorInfos 属性不再直接访问内存
		// 如果需要 tensor 信息，建议使用 gguf_get_tensor_name, gguf_get_tensor_type 等 API 逐个获取

		public void SetValueString(string key, string value)
		{
			ThrowIfNotInitialized();
			Native.gguf_set_val_str(this, key, value);
		}

		public void SetValueBool(string key, bool value)
		{
			ThrowIfNotInitialized();
			Native.gguf_set_val_bool(this, key, value);
		}

		public void SetValueFloat(string key, float value)
		{
			ThrowIfNotInitialized();
			Native.gguf_set_val_f32(this, key, value);
		}

		public void SetValueFloat64(string key, double value)
		{
			if (IsInitialized)
			{
				Native.gguf_set_val_f64(this, key, value);
			}
		}

		public void SetValueInt16(string key, Int16 value)
		{
			ThrowIfNotInitialized();
			Native.gguf_set_val_i16(this, key, value);
		}

		public void SetValueInt32(string key, int value)
		{
			ThrowIfNotInitialized();
			Native.gguf_set_val_i32(this, key, value);
		}

		public void SetValueInt64(string key, Int64 value)
		{
			ThrowIfNotInitialized();
			Native.gguf_set_val_i64(this, key, value);
		}

		public void SetValueInt8(string key, sbyte value)
		{
			ThrowIfNotInitialized();
			Native.gguf_set_val_i8(this, key, value);
		}

		public void SetValueUInt8(string key, byte value)
		{
			ThrowIfNotInitialized();
			Native.gguf_set_val_u8(this, key, value);
		}

		public void SetValueUInt16(string key, UInt16 value)
		{
			ThrowIfNotInitialized();
			Native.gguf_set_val_u16(this, key, value);
		}

		public void SetValueUInt32(string key, uint value)
		{
			if (IsInitialized)
			{
				Native.gguf_set_val_u32(this, key, value);
			}
		}

		public void SetValueUInt64(string key, UInt64 value)
		{
			ThrowIfNotInitialized();
			Native.gguf_set_val_u64(this, key, value);
		}

		public void SetValueArrayData(string key, Structs.GGufType type, Array data)
		{
			ThrowIfNotInitialized();
			if (type == Structs.GGufType.GGUF_TYPE_ARRAY)
			{
				throw new ArgumentException("Array not support");
			}
			else if (type == Structs.GGufType.GGUF_TYPE_STRING)
			{
				gguf_str[] ggufStrs = new gguf_str[data.Length];
				for (int i = 0; i < data.Length; i++)
				{
					ggufStrs[i] = new gguf_str { data = StringToCoTaskMemUTF8((string)data.GetValue(i)), n = (ulong)data.GetValue(i).ToString().Length };
				}
				IntPtr ptr = Marshal.UnsafeAddrOfPinnedArrayElement(ggufStrs, 0);
				Native.gguf_set_arr_data(this, key, type, ptr, data.Length);
			}
			else
			{
				IntPtr ptr = Marshal.UnsafeAddrOfPinnedArrayElement(data, 0);
				Native.gguf_set_arr_data(this, key, type, ptr, data.Length);
			}
		}

		public void SetValueArrayString(string key, string[] strs)
		{
			ThrowIfNotInitialized();
			IntPtr[] dataPtrs = new IntPtr[strs.Length];
			for (int i = 0; i < strs.Length; i++)
			{
				dataPtrs[i] = StringToCoTaskMemUTF8(strs[i]);
			}
			Native.gguf_set_arr_str(this, key, dataPtrs, strs.Length);
		}

		public void Free()
		{
			if (IsInitialized)
			{
				Native.gguf_free(handle);
				handle = IntPtr.Zero;
			}
		}

		public static unsafe IntPtr StringToCoTaskMemUTF8(string s)
		{
			if (s is null)
			{
				return IntPtr.Zero;
			}
			int nb = Encoding.UTF8.GetMaxByteCount(s.Length);
			IntPtr ptr = Marshal.AllocCoTaskMem(checked(nb + 1));
			byte* pbMem = (byte*)ptr;
			char[] chars = s.ToCharArray();
			Marshal.Copy(chars, 0, ptr, chars.Length);
			fixed (char* chr = chars)
			{
				int nbWritten = Encoding.UTF8.GetBytes(chr, chars.Length, pbMem, nb);
				pbMem[nbWritten] = 0;
			}
			return ptr;
		}

		public void WriteToFile(string filename, bool metaOnly = false)
		{
			ThrowIfNotInitialized();
			Native.gguf_write_to_file(this, filename, metaOnly);
		}

		public void AddTensor(SafeGGmlTensor tensor)
		{
			ThrowIfNotInitialized();
			Native.gguf_add_tensor(this, tensor);
		}

		public string GetTensorName(int index)
		{
			ThrowIfNotInitialized();
			IntPtr ptr = Native.gguf_get_tensor_name(this, (long)index);
			if (ptr == IntPtr.Zero)
			{
				return string.Empty;
			}
			return Marshal.PtrToStringUTF8(ptr) ?? string.Empty;
		}

		// 为了兼容性保留这个方法，但内部通过 C API 获取
		public ulong GetDataOffset()
		{
			return DataOffset;
		}

		public ulong GetTensorOffset(int i)
		{
			ThrowIfNotInitialized();
			return Native.gguf_get_tensor_offset(this, i);
		}

		public ulong GetTensorSize(int i)
		{
			ThrowIfNotInitialized();
			return Native.gguf_get_tensor_size(this, (long)i);
		}

		public Structs.GGmlType GetTensorType(int i)
		{
			ThrowIfNotInitialized();
			return Native.gguf_get_tensor_type(this, i);
		}

		// 添加一些便捷方法来获取单个 key-value 信息
		public string GetKey(int keyId)
		{
			ThrowIfNotInitialized();
			IntPtr ptr = Native.gguf_get_key(this, (long)keyId);
			if (ptr == IntPtr.Zero)
			{
				return string.Empty;
			}
			return Marshal.PtrToStringUTF8(ptr) ?? string.Empty;
		}

		public int FindKey(string key)
		{
			ThrowIfNotInitialized();
			return Native.gguf_find_key(this, key) ? 0 : -1;
		}
	}

	// 为了避免编译错误，保留这些类型声明，但不建议直接使用
	[StructLayout(LayoutKind.Sequential)]
	internal struct gguf_str
	{
		public ulong n;
		public IntPtr data;
	}
}
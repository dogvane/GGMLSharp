using System;
using System.Runtime.InteropServices;
using static GGMLSharp.InternalStructs;

namespace GGMLSharp
{
	public unsafe class SafeGGmlTensor : SafeGGmlHandleBase
	{
		private ggml_tensor* tensor => (ggml_tensor*)handle;

		private bool IsInitialized => handle != IntPtr.Zero;

	/// <summary>
	/// Gets the native handle for this tensor (useful for interop with other native functions)
	/// </summary>
	public IntPtr NativeHandle => handle;

		public SafeGGmlTensor()
		{
			handle = IntPtr.Zero;
		}

		public SafeGGmlTensor(IntPtr intPtr)
		{
			this.handle = intPtr;
		}

		internal SafeGGmlTensor(ggml_tensor* tensor)
		{
			this.handle = (IntPtr)(tensor);
		}

		public SafeGGmlTensor(SafeGGmlContext context, Structs.GGmlType type, long[] shape)
		{
			this.handle = Native.ggml_new_tensor(context, type, shape.Length, shape).handle;
		}
		private void ThrowIfNotInitialized()
		{
			if (!IsInitialized)
			{
				throw new ObjectDisposedException("Not initialized or disposed");
			}
		}

		public string Name
		{
			get { return Marshal.PtrToStringAnsi((IntPtr)tensor->name); }
			set { Native.ggml_set_name(this, value); }
		}

		public void FormatName(string name)
		{
			Native.ggml_format_name(this, name);
		}

		public string ViewName { get; set; }

		public long[] Shape
		{
			get
			{
				long[] shape = new long[4];
				shape[0] = tensor->ne[0];
				shape[1] = tensor->ne[1];
				shape[2] = tensor->ne[2];
				shape[3] = tensor->ne[3];
				return shape;
			}
		}

		public ulong[] Stride
		{
			get
			{
				ulong[] stride = new ulong[4];
				stride[0] = tensor->nb[0];
				stride[1] = tensor->nb[1];
				stride[2] = tensor->nb[2];
				stride[3] = tensor->nb[3];
				return stride;
			}
		}

		/// <summary>
		/// Stride (bytes per dimension) - alias for Stride
		/// </summary>
		public ulong[] Nb => Stride;

		public Structs.GGmlType Type => (Structs.GGmlType)tensor->type;

		// NOTE: 'backend' field removed in ggml v0.9.7+
		// Backend information is now managed separately through the buffer system
		// public Structs.GGmlBackendType Backend => (Structs.GGmlBackendType)tensor->backend;

		public IntPtr Data => tensor->data;

		// NOTE: 'grad' field removed from ggml_tensor structure
		// Gradients are now managed separately through the computation graph
		// public SafeGGmlTensor Grad => new SafeGGmlTensor((IntPtr)tensor->grad);

		public SafeGGmlTensor ViewSource => new SafeGGmlTensor(tensor->view_src);


		public Structs.GGmlOperation Operations
		{
			get { return (Structs.GGmlOperation)tensor->op; }
			set { tensor->op = (InternalStructs.ggml_op)value; }
		}


		public SafeGGmlTensor[] Sources
		{
			get
			{
				SafeGGmlTensor[] src = new SafeGGmlTensor[GGML_MAX_SRC];
				for (int i = 0; i < GGML_MAX_SRC; i++)
				{
					// 使用 ggml_tensor_array_src 的索引器访问 src 数组，直接返回 ggml_tensor*
					src[i] = new SafeGGmlTensor((IntPtr)(tensor->src[i]));
				}
				return src;
			}
		}

		public void SetSource(int index, SafeGGmlTensor? source)
		{
			ThrowIfNotInitialized();
			if (index < 0 || index >= GGML_MAX_SRC)
			{
				throw new ArgumentOutOfRangeException(nameof(index));
			}

			tensor->src[index] = (ggml_tensor*)(source?.NativeHandle ?? IntPtr.Zero);
		}

		public void SetData(byte[] data)
		{
			ThrowIfNotInitialized();
			if (data == null || data.Length == 0)
			{
				throw new ArgumentNullException(nameof(data), "Data array is null or empty");
			}

			ulong size = (ulong)data.Length;
			ulong tensorSize = ElementsSize * (ulong)ElementsCount;

			if (size != tensorSize)
			{
				throw new ArgumentOutOfRangeException(nameof(data), $"Data size {size} bytes does not match tensor size {tensorSize} bytes");
			}

			// 分配临时非托管内存
			IntPtr tempBuffer = Marshal.AllocHGlobal((int)size);
			try
			{
				// 复制数据到临时缓冲区
				Marshal.Copy(data, 0, tempBuffer, data.Length);
				// 使用 backend API 来设置数据
				Native.ggml_backend_tensor_set(this, tempBuffer, 0, size);
			}
			finally
			{
				Marshal.FreeHGlobal(tempBuffer);
			}
		}

		public void SetData(float[] data)
		{
			ThrowIfNotInitialized();
			if (data == null || data.Length == 0)
			{
				throw new ArgumentNullException(nameof(data), "Data array is null or empty");
			}

			// 计算需要的字节数
			ulong size = (ulong)(data.Length * sizeof(float));
			ulong tensorSize = ElementsSize * (ulong)ElementsCount;

			if (size != tensorSize)
			{
				throw new ArgumentOutOfRangeException(nameof(data), $"Data size {size} bytes does not match tensor size {tensorSize} bytes");
			}

			// 分配临时非托管内存
			IntPtr tempBuffer = Marshal.AllocHGlobal((int)size);
			try
			{
				// 复制数据到临时缓冲区
				Marshal.Copy(data, 0, tempBuffer, data.Length);
				// 使用 backend API 来设置数据（这个方法可以处理所有情况）
				Native.ggml_backend_tensor_set(this, tempBuffer, 0, size);
			}
			finally
			{
				Marshal.FreeHGlobal(tempBuffer);
			}
		}

		public void SetData(int[] data)
		{
			ThrowIfNotInitialized();
			if (data == null || data.Length == 0)
			{
				throw new ArgumentNullException(nameof(data), "Data array is null or empty");
			}

			// 计算需要的字节数
			ulong size = (ulong)(data.Length * sizeof(int));
			ulong tensorSize = ElementsSize * (ulong)ElementsCount;

			if (size != tensorSize)
			{
				throw new ArgumentOutOfRangeException(nameof(data), $"Data size {size} bytes does not match tensor size {tensorSize} bytes");
			}

			// 分配临时非托管内存
			IntPtr tempBuffer = Marshal.AllocHGlobal((int)size);
			try
			{
				// 复制数据到临时缓冲区
				Marshal.Copy(data, 0, tempBuffer, data.Length);
				// 使用 backend API 来设置数据
				Native.ggml_backend_tensor_set(this, tempBuffer, 0, size);
			}
			finally
			{
				Marshal.FreeHGlobal(tempBuffer);
			}
		}

		public void SetData(int data, int ne0, int ne1 = 0, int ne2 = 0, int ne3 = 0)
		{
			ThrowIfNotInitialized();
			Native.ggml_set_i32_nd(this, ne0, ne1, ne2, ne3, data);
		}

		public void SetInt1D(int index, int value)
		{
			ThrowIfNotInitialized();
			Native.ggml_set_i32_1d(this, index, value);
		}

		public int GetInt(int n0 = 0, int n1 = 0, int n2 = 0, int n3 = 0)
		{
			return Native.ggml_get_i32_nd(this, n0, n1, n2, n3);
		}

		public int GetInt1D(int index)
		{
			ThrowIfNotInitialized();
			return Native.ggml_get_i32_1d(this, index);
		}

		//public void SetData(float data, int index)
		//{
		//	ThrowIfNotInitialized();
		//	Native.ggml_set_f32_1d(this, index, data);
		//}

		public void SetData(float data, int ne0, int ne1 = 0, int ne2 = 0, int ne3 = 0)
		{
			ThrowIfNotInitialized();
			Native.ggml_set_f32_nd(this, ne0, ne1, ne2, ne3, data);
		}

		public float GetFloat(int n0 = 0, int n1 = 0, int n2 = 0, int n3 = 0)
		{
			return Native.ggml_get_f32_nd(this, n0, n1, n2, n3);
		}

        public float[] GetDataInFloat16()
        {
            long length = ElementsCount;
            IntPtr dataPtr = Native.ggml_get_data(this);
            float[] floats = new float[length];

            unsafe
            {
                ushort* fp16Data = (ushort*)dataPtr;
                for (long i = 0; i < length; i++)
                {
                    floats[i] = Native.ggml_fp16_to_fp32(fp16Data[i]);
                }
            }

            return floats;
        }

        public float[] GetDataInFloats()
		{
			long length = Shape[0] * Shape[1] * Shape[2] * Shape[3];
			float* f = Native.ggml_get_data_f32(this);
			float[] floats = new float[length];

			for (long i = 0; i < length; i++)
			{
				floats[i] = f[i];
			}
			return floats;
		}

		public byte[] GetData()
		{
			ulong size = ElementsSize * (ulong)ElementsCount;
			IntPtr ptr = (IntPtr)Native.ggml_get_data(this);
			byte[] bytes = new byte[size];
			Marshal.Copy(ptr, bytes, 0, (int)size);
			return bytes;
		}

		public void SetBackend(IntPtr data, ulong offset = 0, ulong size = 0)
		{
			ThrowIfNotInitialized();
			size = size == 0 ? ElementsSize * (ulong)ElementsCount : size;
			Native.ggml_backend_tensor_set(this, data, offset, size);
		}

		public void SetDataToBackend()
		{
			ThrowIfNotInitialized();
			if (Data == IntPtr.Zero)
			{
				throw new ArgumentNullException("data is null");
			}
			SetBackend(GetData());
		}

		public void SetBackend(Array data, ulong offset = 0, ulong size = 0)
		{
			ThrowIfNotInitialized();

			IntPtr ptr = Marshal.UnsafeAddrOfPinnedArrayElement(data, 0);
			size = size == 0 ? ElementsSize * (ulong)ElementsCount : size;
			int sz = Marshal.SizeOf(data.GetType().GetElementType());
			if ((ulong)(sz * data.Length) != size)
			{
				throw new ArgumentOutOfRangeException("data size is not fit");
			}
			Native.ggml_backend_tensor_set(this, ptr, offset, size);
		}



		public long ElementsCount => Native.ggml_nelements(this);
		public ulong ElementsSize => Native.ggml_element_size(this);

		public byte[] GetBackend()
		{
			ThrowIfNotInitialized();
			ulong size = (ulong)ElementsCount * ElementsSize;
			IntPtr ptr = Marshal.AllocHGlobal((int)size);
			Native.ggml_backend_tensor_get(this, ptr, 0, size);
			byte[] bytes = new byte[size];
			Marshal.Copy(ptr, bytes, 0, (int)size);
			return bytes;
		}

		public void SetInput()
		{
			ThrowIfNotInitialized();
			Native.ggml_set_input(this);
		}

		public void SetOutput()
		{
			ThrowIfNotInitialized();
			Native.ggml_set_output(this);
		}

		public void SetRandomTensorInFloat(float max, float min)
		{
			ThrowIfNotInitialized();
			Random random = new Random();

			long size = Shape[0] * Shape[1] * Shape[2] * Shape[3];
			for (long i = 0; i < size; i++)
			{
				float f = (float)random.NextDouble() * (max - min) + min;
				SetData(f, (int)i);
			}
		}


		public void SetBackendRandomTensorInFloat(float max, float min)
		{
			ThrowIfNotInitialized();
			Random random = new Random();

			long size = Shape[0] * Shape[1] * Shape[2] * Shape[3];
			float[] floats = new float[size];
			for (long i = 0; i < size; i++)
			{
				floats[i] = (float)random.NextDouble() * (max - min) + min;
			}
			SetBackend(floats);
		}

		public void SetZero()
		{
			ThrowIfNotInitialized();
			Native.ggml_set_zero(this);
		}

		public bool AreSameShape(SafeGGmlTensor tensor)
		{
			ThrowIfNotInitialized();
			return Native.ggml_are_same_shape(this, tensor);
		}

		public bool IsContiguous()
		{
			ThrowIfNotInitialized();
			return Native.ggml_is_contiguous(this);
		}

		public bool IsView()
		{
			ThrowIfNotInitialized();
			return Native.ggml_is_view(this);
		}

		public bool IsQuantized()
		{
			ThrowIfNotInitialized();
			return Native.ggml_is_quantized(this.Type);
		}

		public ulong NBytes
		{
			get { return Native.ggml_nbytes(this); }
		}

		public long NElements
		{
			get { return Native.ggml_nelements(this); }
		}

		public float[] GetDataF32()
		{
			ThrowIfNotInitialized();
			float* data = Native.ggml_get_data_f32(this);
			if (data == null)
			{
				return new float[0];
			}
			long nElements = NElements;
			float[] result = new float[nElements];
			for (long i = 0; i < nElements; i++)
			{
				result[i] = data[i];
			}
			return result;
		}

		public void TensorSet(float[] data, ulong dataOffset, ulong dataSize)
		{
			ThrowIfNotInitialized();
			IntPtr dataPtr = Marshal.AllocHGlobal(data.Length * 4);
			Marshal.Copy(data, 0, dataPtr, data.Length);
			Native.ggml_backend_tensor_set(this, dataPtr, dataOffset, dataSize);
			Marshal.FreeHGlobal(dataPtr);
		}

		public void TensorGet(float[] data, ulong dataOffset, ulong dataSize)
		{
			ThrowIfNotInitialized();
			IntPtr dataPtr = Marshal.AllocHGlobal(data.Length * 4);
			Native.ggml_backend_tensor_get(this, dataPtr, dataOffset, dataSize);
			Marshal.Copy(dataPtr, data, 0, data.Length);
			Marshal.FreeHGlobal(dataPtr);
		}
		}
}

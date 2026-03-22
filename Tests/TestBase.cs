using GGMLSharp;

namespace GGMLSharp.Tests
{
	/// <summary>
	/// Base class for GGML tests with common initialization and helper methods
	/// </summary>
	public abstract class TestBase : IDisposable
	{
		protected SafeGGmlContext Context { get; private set; }
		protected SafeGGmlBackend Backend { get; private set; }

		private bool _disposed = false;

		/// <summary>
		/// Default constructor initializes with no-allocate context and CPU backend
		/// </summary>
		protected TestBase() : this(initNoAlloc: true, initBackend: true)
		{
		}

		/// <summary>
		/// Constructor with options for what to initialize
		/// </summary>
		protected TestBase(bool initNoAlloc = true, bool initBackend = true)
		{
			if (initNoAlloc)
			{
				Context = new SafeGGmlContext(IntPtr.Zero, 10 * 1024 * 1024, NoAllocateMemory: true);
				if (Context == null || Context.IsInvalid)
					throw new Exception("无法初始化 SafeGGmlContext");
			}
			if (initBackend)
			{
				Backend = SafeGGmlBackend.CpuInit();
				if (Backend == null || Backend.IsInvalid)
					throw new Exception("无法初始化 SafeGGmlBackend");
			}
		}

		/// <summary>
		/// Creates a new tensor with the given shape and type
		/// </summary>
		protected SafeGGmlTensor CreateTensor(long[] shape, Structs.GGmlType type = Structs.GGmlType.GGML_TYPE_F32)
		{
			if (Context == null)
			{
				throw new InvalidOperationException("Context is not initialized");
			}

			var tensor = Context.NewTensor(type, shape);
			if (tensor.IsInvalid)
			{
				throw new Exception("无法创建 tensor");
			}
			return tensor;
		}

		/// <summary>
		/// Initializes tensor with uniform random values
		/// </summary>
		protected void InitTensorUniform(SafeGGmlTensor tensor, float min = -1.0f, float max = 1.0f)
		{
			Random random = new Random(0);
			long nelements = tensor.ElementsCount;

			for (long i = 0; i < nelements; i++)
			{
				float value = (float)random.NextDouble() * (max - min) + min;
				long ne0 = tensor.Shape[0];
				long ne1 = tensor.Shape[1];
				long ne2 = tensor.Shape[2];
				long ne3 = tensor.Shape[3];

				long n3 = (i / (ne2 * ne1 * ne0)) % ne3;
				long n2 = (i / (ne1 * ne0)) % ne2;
				long n1 = (i / ne0) % ne1;
				long n0 = i % ne0;

				tensor.SetData(value, (int)n0, (int)n1, (int)n2, (int)n3);
			}
		}

		/// <summary>
		/// Initializes I32 tensor with random integer values
		/// </summary>
		protected void InitTensorInt(SafeGGmlTensor tensor, int min = 0, int max = 100)
		{
			Random random = new Random(0);
			long nelements = tensor.ElementsCount;
			int[] data = new int[nelements];

			for (long i = 0; i < nelements; i++)
			{
				data[i] = random.Next(min, max);
			}

			tensor.SetData(data);
		}

		/// <summary>
		/// Initializes tensor with sequential values starting from 'start'
		/// </summary>
		protected void InitTensorWithSequence(SafeGGmlTensor tensor, float start = 1.0f)
		{
			long nelements = tensor.ElementsCount;
			for (long i = 0; i < nelements; i++)
			{
				float value = start + (float)i;
				long ne0 = tensor.Shape[0];
				long ne1 = tensor.Shape[1];
				long ne2 = tensor.Shape[2];
				long ne3 = tensor.Shape[3];

				long n3 = (i / (ne2 * ne1 * ne0)) % ne3;
				long n2 = (i / (ne1 * ne0)) % ne2;
				long n1 = (i / ne0) % ne1;
				long n0 = i % ne0;

				tensor.SetData(value, (int)n0, (int)n1, (int)n2, (int)n3);
			}
		}

		/// <summary>
		/// Calculates RMS error between two float arrays
		/// </summary>
		protected float CalculateRMSError(float[] a, float[] b)
		{
			if (a.Length != b.Length)
			{
				throw new ArgumentException("数组长度不匹配");
			}

			double sumSquaredError = 0;
			for (int i = 0; i < a.Length; i++)
			{
				double error = a[i] - b[i];
				sumSquaredError += error * error;
			}

			double mse = sumSquaredError / a.Length;
			return (float)Math.Sqrt(mse);
		}

		/// <summary>
		/// Executes a standard backend compute workflow: builds graph, allocates tensors, executes
		/// </summary>
		protected void ExecuteBackendCompute(SafeGGmlTensor resultTensor)
		{
			if (Context == null || Backend == null)
			{
				throw new InvalidOperationException("Context or Backend is not initialized");
			}

			var graph = Context.NewGraph();
			graph.BuildForwardExpend(resultTensor);
			Context.BackendAllocContextTensors(Backend);
			graph.BackendCompute(Backend);
		}

		/// <summary>
		/// Dispose pattern implementation
		/// </summary>
		public void Dispose()
		{
			Dispose(true);
			GC.SuppressFinalize(this);
		}

		/// <summary>
		/// Dispose pattern implementation
		/// </summary>
		protected virtual void Dispose(bool disposing)
		{
			if (_disposed)
				return;

			if (disposing)
			{
				// Dispose managed resources
				Backend?.Free();
				Context?.Free();
			}

			_disposed = true;
		}

		/// <summary>
		/// Finalizer
		/// </summary>
		~TestBase()
		{
			Dispose(false);
		}
	}
}

using System.Runtime.InteropServices;
using System;

namespace GGMLSharp
{
	public class Structs
	{

		/// <summary>
		/// "ggml"
		/// </summary>
		public const int GGML_FILE_MAGIC = 0x67676d6c;
		public const int GGML_FILE_VERSION = 1;

		/// <summary>
		/// bump this on quantization format changes
		/// </summary>
		public const int GGML_QNT_VERSION = 2;

		/// <summary>
		/// do not change this
		/// </summary>
		public const int GGML_QNT_VERSION_FACTOR = 1000;

		public const int GGML_MAX_DIMS = 4;
		public const int GGML_MAX_PARAMS = 2048;
		public const int GGML_MAX_CONTEXTS = 64;
		public const int GGML_MAX_SRC = 10;
		public const int GGML_MAX_NAME = 128; // 64?
		public const int GGML_MAX_OP_PARAMS = 64;
		public const int GGML_DEFAULT_N_THREADS = 4;
		public const int GGML_DEFAULT_GRAPH_SIZE = 2048;

		/// <summary>
		///  x64 only
		/// </summary>
		public const int GGML_MEM_ALIGN = 16;

		public const int GGML_EXIT_SUCCESS = 0;
		public const int GGML_EXIT_ABORTED = 1;

		public const string GGUF_MAGIC = "GGUF";

		public const uint GGUF_VERSION = 3;

		public const ulong GGUF_DEFAULT_ALIGNMENT = 32;
		public const int GGML_N_TASKS_MAX = -1;
		public const int GGML_KQ_MASK_PAD = 32;
		public const int MAX_FREE_BLOCKS = 256;

		public enum GGmlStatus
		{
			GGML_STATUS_ALLOC_FAILED = -2,
			GGML_STATUS_FAILED = -1,
			GGML_STATUS_SUCCESS = 0,
			GGML_STATUS_ABORTED = 1,
		};

		public enum GGmlType
		{
			GGML_TYPE_F32 = 0,
			GGML_TYPE_F16 = 1,
			GGML_TYPE_Q4_0 = 2,
			GGML_TYPE_Q4_1 = 3,
			// GGML_TYPE_Q4_2 = 4, support has been removed
			// GGML_TYPE_Q4_3 = 5, support has been removed
			GGML_TYPE_Q5_0 = 6,
			GGML_TYPE_Q5_1 = 7,
			GGML_TYPE_Q8_0 = 8,
			GGML_TYPE_Q8_1 = 9,
			GGML_TYPE_Q2_K = 10,
			GGML_TYPE_Q3_K = 11,
			GGML_TYPE_Q4_K = 12,
			GGML_TYPE_Q5_K = 13,
			GGML_TYPE_Q6_K = 14,
			GGML_TYPE_Q8_K = 15,
			GGML_TYPE_IQ2_XXS = 16,
			GGML_TYPE_IQ2_XS = 17,
			GGML_TYPE_IQ3_XXS = 18,
			GGML_TYPE_IQ1_S = 19,
			GGML_TYPE_IQ4_NL = 20,
			GGML_TYPE_IQ3_S = 21,
			GGML_TYPE_IQ2_S = 22,
			GGML_TYPE_IQ4_XS = 23,
			GGML_TYPE_I8 = 24,
			GGML_TYPE_I16 = 25,
			GGML_TYPE_I32 = 26,
			GGML_TYPE_I64 = 27,
			GGML_TYPE_F64 = 28,
			GGML_TYPE_IQ1_M = 29,
			GGML_TYPE_BF16 = 30,
			// New types in latest ggml
			GGML_TYPE_TQ1_0 = 34,
			GGML_TYPE_TQ2_0 = 35,
			GGML_TYPE_MXFP4 = 39,
			GGML_TYPE_COUNT = 40,
		};

		public enum GGufType
		{
			GGUF_TYPE_UINT8 = 0,
			GGUF_TYPE_INT8 = 1,
			GGUF_TYPE_UINT16 = 2,
			GGUF_TYPE_INT16 = 3,
			GGUF_TYPE_UINT32 = 4,
			GGUF_TYPE_INT32 = 5,
			GGUF_TYPE_FLOAT32 = 6,
			GGUF_TYPE_BOOL = 7,
			GGUF_TYPE_STRING = 8,
			GGUF_TYPE_ARRAY = 9,
			GGUF_TYPE_UINT64 = 10,
			GGUF_TYPE_INT64 = 11,
			GGUF_TYPE_FLOAT64 = 12,
			/// <summary>
			/// marks the end of the enum
			/// </summary>
			GGUF_TYPE_COUNT,
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct OptimizerParameters
		{
			public OptimizerType Type;

			public UInt64 GraphSize;

			public int Threads;

			// Delta-based convergence test
			//
			//   if Past == 0 - disabled
			//   if Past > 0:
			//     stop if |f(x) - f(x_past)| < Delta * max(1, |f(x)|)
			//
			public int Past;
			public float Delta;

			// maximum number of iterations without improvement
			//
			//   if 0 - disabled
			//   if > 0:
			//     assume convergence if no cost improvement in this number of iterations
			//
			public int MaxNoImprovement;

			[MarshalAs(UnmanagedType.I1)]
			public byte PrintForwarGraph;
			[MarshalAs(UnmanagedType.I1)]
			public byte PrintBackwardGraph;

			public int GradientAccumulationCount;

			public AdamParams Adam;
			public LbfgsParams Lbfgs;

			[StructLayout(LayoutKind.Sequential)]
			// ADAM parameters
			public struct AdamParams
			{
				public int Iter;

				public float Sched; // schedule multiplier (fixed, Decay or warmup)
				public float Decay; // Weight Decay for AdamW, use 0.0f to disable
				public int DecayMinDim; // minimum number of tensor dimension to apply Weight Decay
				public float Alpha; // learning rate
				public float Beta1;
				public float Beta2;
				public float Eps;   // epsilon for numerical stability
				public float EpsF; // epsilon for convergence test
				public float EpsG; // epsilon for convergence test
				public float GradientClipping; // gradient clipping
			}

			// LBFGS parameters
			[StructLayout(LayoutKind.Sequential, Pack = 8)]
			public struct LbfgsParams
			{
				public int NumberOfCorrections; // number of corrections to approximate the inv. Hessian
				public int Iter;
				public int MaxLinesearch;

				public float Eps;      // convergence tolerance
				public float ftol;     // line search tolerance
				public float Wolfe;
				public float MinStep;
				public float MaxStep;

				public Linesearch linesearch;
			}
		};

		public enum OptimizerType
		{
			ADAM,
			LBFGS,
		};

		/// <summary>
		/// linesearch methods
		/// </summary>
		public enum Linesearch
		{
			GGML_LINESEARCH_DEFAULT = 1,

			GGML_LINESEARCH_BACKTRACKING_ARMIJO = 0,
			GGML_LINESEARCH_BACKTRACKING_WOLFE = 1,
			GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 2,
		};

		/// <summary>
		/// optimization return values
		/// </summary>
		public enum OptimizationResult
		{
			GGML_OPT_RESULT_OK = 0,
			GGML_OPT_RESULT_DID_NOT_CONVERGE,
			GGML_OPT_RESULT_NO_CONTEXT,
			GGML_OPT_RESULT_INVALID_WOLFE,
			GGML_OPT_RESULT_FAIL,
			GGML_OPT_RESULT_CANCEL,

			GGML_LINESEARCH_FAIL = -128,
			GGML_LINESEARCH_MINIMUM_STEP,
			GGML_LINESEARCH_MAXIMUM_STEP,
			GGML_LINESEARCH_MAXIMUM_ITERATIONS,
			GGML_LINESEARCH_INVALID_PARAMETERS,
		};


		public delegate void Custom1OpDelegate([MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(GGmlCustomMarshaler))] SafeGGmlTensor dst, [MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(GGmlCustomMarshaler))] SafeGGmlTensor a, int ith, int nth, IntPtr userdata);
		public delegate void Custom2OpDelegate([MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(GGmlCustomMarshaler))] SafeGGmlTensor dst, [MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(GGmlCustomMarshaler))] SafeGGmlTensor a, [MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(GGmlCustomMarshaler))] SafeGGmlTensor b, int ith, int nth, IntPtr userdata);
		public delegate void Custom3OpDelegate([MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(GGmlCustomMarshaler))] SafeGGmlTensor dst, [MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(GGmlCustomMarshaler))] SafeGGmlTensor a, [MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(GGmlCustomMarshaler))] SafeGGmlTensor b, [MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(GGmlCustomMarshaler))] SafeGGmlTensor c, int ith, int nth, IntPtr userdata);
		public delegate void CustomOpDelegate([MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(GGmlCustomMarshaler))] SafeGGmlTensor dst, int ith, int nth, IntPtr userdata);

		public sealed class GGmlCustomMarshaler : ICustomMarshaler
		{
			private static GGmlCustomMarshaler _instance = new GGmlCustomMarshaler();

			public void CleanUpManagedData(object o)
			{

			}

			public void CleanUpNativeData(IntPtr ptr)
			{

			}

			public int GetNativeDataSize()
			{
				return IntPtr.Size;
			}

			public IntPtr MarshalManagedToNative(object o)
			{
				return IntPtr.Zero;
			}

			public object MarshalNativeToManaged(IntPtr ptr)
			{
				return new SafeGGmlTensor(ptr);
			}

			public static ICustomMarshaler GetInstance(string s)
			{
				return _instance;
			}

			
		}

		/// <summary>
		/// available tensor operations:
		/// </summary>
		public enum GGmlOperation
		{
			GGML_OP_NONE = 0,

			GGML_OP_DUP,
			GGML_OP_ADD,
			GGML_OP_ADD_ID,
			GGML_OP_ADD1,
			GGML_OP_ACC,
			GGML_OP_SUB,
			GGML_OP_MUL,
			GGML_OP_DIV,
			GGML_OP_SQR,
			GGML_OP_SQRT,
			GGML_OP_LOG,
			GGML_OP_SIN,
			GGML_OP_COS,
			GGML_OP_SUM,
			GGML_OP_SUM_ROWS,
			GGML_OP_CUMSUM,
			GGML_OP_MEAN,
			GGML_OP_ARGMAX,
			GGML_OP_COUNT_EQUAL,
			GGML_OP_REPEAT,
			GGML_OP_REPEAT_BACK,
			GGML_OP_CONCAT,
			GGML_OP_SILU_BACK,
			/// <summary>
			/// normalize
			/// </summary>
			GGML_OP_NORM,
			GGML_OP_RMS_NORM,
			GGML_OP_RMS_NORM_BACK,
			GGML_OP_GROUP_NORM,
			GGML_OP_L2_NORM,

			GGML_OP_MUL_MAT,
			GGML_OP_MUL_MAT_ID,
			GGML_OP_OUT_PROD,

			GGML_OP_SCALE,
			GGML_OP_SET,
			GGML_OP_CPY,
			GGML_OP_CONT,
			GGML_OP_RESHAPE,
			GGML_OP_VIEW,
			GGML_OP_PERMUTE,
			GGML_OP_TRANSPOSE,
			GGML_OP_GET_ROWS,
			GGML_OP_GET_ROWS_BACK,
			GGML_OP_SET_ROWS,
			GGML_OP_DIAG,
			GGML_OP_DIAG_MASK_INF,
			GGML_OP_DIAG_MASK_ZERO,
			GGML_OP_SOFT_MAX,
			GGML_OP_SOFT_MAX_BACK,
			GGML_OP_ROPE,
			GGML_OP_ROPE_BACK,
			GGML_OP_CLAMP,
			GGML_OP_CONV_TRANSPOSE_1D,
			GGML_OP_IM2COL,
			GGML_OP_IM2COL_BACK,
			GGML_OP_IM2COL_3D,
			GGML_OP_CONV_2D,
			GGML_OP_CONV_3D,
			GGML_OP_CONV_2D_DW,
			GGML_OP_CONV_TRANSPOSE_2D,
			GGML_OP_POOL_1D,
			GGML_OP_POOL_2D,
			GGML_OP_POOL_2D_BACK,
			/// <summary>
			/// nearest interpolate
			/// </summary>
			GGML_OP_UPSCALE,
			GGML_OP_PAD,
			GGML_OP_PAD_REFLECT_1D,
			GGML_OP_ROLL,
			GGML_OP_ARANGE,
			GGML_OP_TIMESTEP_EMBEDDING,
			GGML_OP_ARGSORT,
			GGML_OP_TOP_K,
			GGML_OP_LEAKY_RELU,
			GGML_OP_TRI,
			GGML_OP_FILL,

			GGML_OP_FLASH_ATTN_EXT,
			GGML_OP_FLASH_ATTN_BACK,
			GGML_OP_SSM_CONV,
			GGML_OP_SSM_SCAN,
			GGML_OP_WIN_PART,
			GGML_OP_WIN_UNPART,
			GGML_OP_GET_REL_POS,
			GGML_OP_ADD_REL_POS,
			GGML_OP_RWKV_WKV6,
			GGML_OP_GATED_LINEAR_ATTN,
			GGML_OP_RWKV_WKV7,
			GGML_OP_SOLVE_TRI,

			GGML_OP_UNARY,

			GGML_OP_MAP_CUSTOM1,
			GGML_OP_MAP_CUSTOM2,
			GGML_OP_MAP_CUSTOM3,
			GGML_OP_CUSTOM,

			GGML_OP_CROSS_ENTROPY_LOSS,
			GGML_OP_CROSS_ENTROPY_LOSS_BACK,
			GGML_OP_OPT_STEP_ADAMW,
			GGML_OP_OPT_STEP_SGD,
			GGML_OP_GLU,
			GGML_OP_COUNT,
		};

		public enum GGmlGraphEvalOrder
		{
			GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT = 0,
			GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT,
			GGML_CGRAPH_EVAL_ORDER_COUNT
		};

		public enum GGmlObjectType
		{
			GGML_OBJECT_TYPE_TENSOR,
			GGML_OBJECT_TYPE_GRAPH,
			GGML_OBJECT_TYPE_WORK_BUFFER
		};

		public enum GGmlBackendType
		{
			GGML_BACKEND_TYPE_CPU = 0,
			GGML_BACKEND_TYPE_GPU = 10,
			GGML_BACKEND_TYPE_GPU_SPLIT = 20,
		};


		public enum GGmlOpPool
		{
			GGML_OP_POOL_MAX,
			GGML_OP_POOL_AVG,
			GGML_OP_POOL_COUNT,
		};

		/// <summary>
		/// GLU (Gated Linear Unit) operations
		/// </summary>
		public enum GGmlGluOp
		{
			GGML_GLU_OP_REGLU,
			GGML_GLU_OP_GEGLU,
			GGML_GLU_OP_SWIGLU,
			GGML_GLU_OP_SWIGLU_OAI,
			GGML_GLU_OP_GEGLU_ERF,
			GGML_GLU_OP_GEGLU_QUICK,
		};

		/// <summary>
		/// Sort order for argsort operation
		/// </summary>
		public enum GGmlSortOrder
		{
			GGML_SORT_ORDER_ASC,
			GGML_SORT_ORDER_DESC,
		};

		/// <summary>
		/// Scale mode for interpolation operations
		/// </summary>
		public enum GGmlScaleMode
		{
			GGML_SCALE_MODE_NEAREST = 0,
			GGML_SCALE_MODE_BILINEAR = 1,
			GGML_SCALE_MODE_BICUBIC = 2,
		};

		/// <summary>
		/// Triangular matrix type
		/// </summary>
		public enum GGmlTriType
		{
			GGML_TRIANGULAR_LOWER,
			GGML_TRIANGULAR_UPPER,
		};

		/// <summary>
		/// Tensor flags for marking input/output/parameter
		/// </summary>
		[Flags]
		public enum GGmlTensorFlag
		{
			GGML_TENSOR_FLAG_INPUT = 1,
			GGML_TENSOR_FLAG_OUTPUT = 2,
			GGML_TENSOR_FLAG_PARAM = 4,
		};

		// ========== New GGML Optimizer Structures (ggml-opt.h) ==========

		/// <summary>
		/// Loss types for the new optimizer API
		/// </summary>
		public enum GGmlOptLossType
		{
			GGML_OPT_LOSS_TYPE_MEAN,
			GGML_OPT_LOSS_TYPE_SUM,
			GGML_OPT_LOSS_TYPE_CROSS_ENTROPY,
			GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR,
		}

		/// <summary>
		/// Optimizer types for the new optimizer API
		/// </summary>
		public enum GGmlOptOptimizerType
		{
			GGML_OPT_OPTIMIZER_TYPE_ADAMW,
			GGML_OPT_OPTIMIZER_TYPE_SGD,
			GGML_OPT_OPTIMIZER_TYPE_COUNT
		}

		/// <summary>
		/// Build types for the new optimizer API
		/// </summary>
		public enum GGmlOptBuildType
		{
			GGML_OPT_BUILD_TYPE_FORWARD = 10,
			GGML_OPT_BUILD_TYPE_GRAD = 20,
			GGML_OPT_BUILD_TYPE_OPT = 30,
		}

		/// <summary>
		/// Optimizer parameters for the new optimizer API
		/// </summary>
		[StructLayout(LayoutKind.Sequential)]
		public struct GGmlOptOptimizerParams
		{
			[StructLayout(LayoutKind.Sequential)]
			public struct AdamwParams
			{
				public float alpha; // learning rate
				public float beta1; // first AdamW momentum
				public float beta2; // second AdamW momentum
				public float eps;   // epsilon for numerical stability
				public float wd;    // weight decay - 0.0f to disable
			}

			[StructLayout(LayoutKind.Sequential)]
			public struct SgdParams
			{
				public float alpha; // learning rate
				public float wd;    // weight decay
			}

			public AdamwParams adamw;
			public SgdParams sgd;
		}

		/// <summary>
		/// Result type for optimizer operations
		/// </summary>
		public enum GGmlOptResult
		{
			GGML_OPT_RESULT_OK = 0,
			GGML_OPT_RESULT_DID_NOT_CONVERGE,
			GGML_OPT_RESULT_NO_CONTEXT,
			GGML_OPT_RESULT_INVALID_WOLFE,
			GGML_OPT_RESULT_FAIL,
			GGML_OPT_RESULT_CANCEL,
		}

		/// <summary>
		/// Delegate for getting optimizer parameters
		/// </summary>
		[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
		public delegate GGmlOptOptimizerParams GGmlOptGetOptimizerParamsDelegate(IntPtr userdata);

		/// <summary>
		/// Parameters for initializing an optimization context
		/// </summary>
		[StructLayout(LayoutKind.Sequential)]
		public struct GGmlOptParams
		{
			public IntPtr backend_sched; // ggml_backend_sched_t
			public IntPtr ctx_compute;   // ggml_context*
			public IntPtr inputs;        // ggml_tensor*
			public IntPtr outputs;       // ggml_tensor*
			public GGmlOptLossType loss_type;
			public GGmlOptBuildType build_type;
			public int opt_period;
			public IntPtr get_opt_pars;  // ggml_opt_get_optimizer_params
			public IntPtr get_opt_pars_ud; // void*
			public GGmlOptOptimizerType optimizer;
		}

	}
}

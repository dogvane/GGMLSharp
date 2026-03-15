using System;
using System.Runtime.InteropServices;
using ggml_backend_buffer_type_context_t = System.IntPtr;
using ggml_backend_context_t = System.IntPtr;
using ggml_backend_graph_plan_t = System.IntPtr;
using int16_t = System.Int16;
using int32_t = System.Int32;
using int64_t = System.Int64;
using int8_t = System.SByte;
using size_t = System.UInt64;
using uint16_t = System.UInt16;
using uint64_t = System.UInt64;
using uint8_t = System.Byte;



namespace GGMLSharp
{
	internal unsafe class InternalStructs
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

		public const int GGUF_VERSION = 3;

		public const int GGUF_DEFAULT_ALIGNMENT = 32;
		public const int GGML_N_TASKS_MAX = -1;
		public const int GGML_KQ_MASK_PAD = 32;
		public const int MAX_FREE_BLOCKS = 256;


		#region ggml.h

		public enum ggml_status
		{
			GGML_STATUS_ALLOC_FAILED = -2,
			GGML_STATUS_FAILED = -1,
			GGML_STATUS_SUCCESS = 0,
			GGML_STATUS_ABORTED = 1,
		};

		public enum ggml_type
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
			// GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
			// GGML_TYPE_Q4_0_4_8 = 32,
			// GGML_TYPE_Q4_0_8_8 = 33,
			GGML_TYPE_TQ1_0 = 34,
			GGML_TYPE_TQ2_0 = 35,
			// GGML_TYPE_IQ4_NL_4_4 = 36,
			// GGML_TYPE_IQ4_NL_4_8 = 37,
			// GGML_TYPE_IQ4_NL_8_8 = 38,
			GGML_TYPE_MXFP4 = 39,
			GGML_TYPE_COUNT = 40,
		};

		/// <summary>
		/// precision
		/// </summary>
		public enum ggml_prec
		{
			GGML_PREC_DEFAULT,
			GGML_PREC_F32,
		};

		public enum ggml_backend_type
		{
			GGML_BACKEND_TYPE_CPU = 0,
			GGML_BACKEND_TYPE_GPU = 10,
			GGML_BACKEND_TYPE_GPU_SPLIT = 20,
		};

		/// <summary>
		/// model file types
		/// </summary>
		public enum ggml_ftype
		{
			GGML_FTYPE_UNKNOWN = -1,
			GGML_FTYPE_ALL_F32 = 0,

			/// <summary>
			/// except 1d tensors
			/// </summary>
			GGML_FTYPE_MOSTLY_F16 = 1,

			/// <summary>
			/// except 1d tensors
			/// </summary>
			GGML_FTYPE_MOSTLY_Q4_0 = 2,

			/// <summary>
			/// except 1d tensors
			/// </summary>
			GGML_FTYPE_MOSTLY_Q4_1 = 3,

			/// <summary>
			/// tok_embeddings.Weight and output.Weight are F16
			/// </summary>
			GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,

			/// <summary>
			/// except 1d tensors
			/// </summary>
			GGML_FTYPE_MOSTLY_Q8_0 = 7,

			/// <summary>
			/// except 1d tensors
			/// </summary>
			GGML_FTYPE_MOSTLY_Q5_0 = 8,  // except 1d tensors
			GGML_FTYPE_MOSTLY_Q5_1 = 9,  // except 1d tensors
			GGML_FTYPE_MOSTLY_Q2_K = 10, // except 1d tensors
			GGML_FTYPE_MOSTLY_Q3_K = 11, // except 1d tensors
			GGML_FTYPE_MOSTLY_Q4_K = 12, // except 1d tensors
			GGML_FTYPE_MOSTLY_Q5_K = 13, // except 1d tensors
			GGML_FTYPE_MOSTLY_Q6_K = 14, // except 1d tensors
			GGML_FTYPE_MOSTLY_IQ2_XXS = 15, // except 1d tensors
			GGML_FTYPE_MOSTLY_IQ2_XS = 16, // except 1d tensors
			GGML_FTYPE_MOSTLY_IQ3_XXS = 17, // except 1d tensors
			GGML_FTYPE_MOSTLY_IQ1_S = 18, // except 1d tensors
			GGML_FTYPE_MOSTLY_IQ4_NL = 19, // except 1d tensors
			GGML_FTYPE_MOSTLY_IQ3_S = 20, // except 1d tensors
			GGML_FTYPE_MOSTLY_IQ2_S = 21, // except 1d tensors
			GGML_FTYPE_MOSTLY_IQ4_XS = 22, // except 1d tensors
			GGML_FTYPE_MOSTLY_IQ1_M = 23, // except 1d tensors
			GGML_FTYPE_MOSTLY_BF16 = 24, // except 1d tensors
			GGML_FTYPE_MOSTLY_MXFP4 = 25, // except 1d tensors
		};

		/// <summary>
		/// available tensor operations:
		/// </summary>
		public enum ggml_op
		{
			GGML_OP_NONE = 0,

			GGML_OP_DUP,
			GGML_OP_ADD,
			GGML_OP_ADD1,
			GGML_OP_ACC,
			GGML_OP_SUB,
			GGML_OP_MUL,
			GGML_OP_DIV,
			GGML_OP_SQR,
			GGML_OP_SQRT,
			GGML_OP_LOG,
			GGML_OP_SUM,
			GGML_OP_SUM_ROWS,
			GGML_OP_MEAN,
			GGML_OP_ARGMAX,
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
			GGML_OP_CONV_TRANSPOSE_2D,
			GGML_OP_POOL_1D,
			GGML_OP_POOL_2D,
			/// <summary>
			/// nearest interpolate
			/// </summary>
			GGML_OP_UPSCALE,
			GGML_OP_PAD,
			GGML_OP_ARANGE,
			GGML_OP_TIMESTEP_EMBEDDING,
			GGML_OP_ARGSORT,
			GGML_OP_LEAKY_RELU,

			GGML_OP_FLASH_ATTN,
			GGML_OP_FLASH_ATTN_EXT,
			GGML_OP_FLASH_FF,
			GGML_OP_FLASH_ATTN_BACK,
			GGML_OP_SSM_CONV,
			GGML_OP_SSM_SCAN,
			GGML_OP_WIN_PART,
			GGML_OP_WIN_UNPART,
			GGML_OP_GET_REL_POS,
			GGML_OP_ADD_REL_POS,

			GGML_OP_UNARY,

			GGML_OP_MAP_UNARY,
			GGML_OP_MAP_BINARY,

			GGML_OP_MAP_CUSTOM1_F32,
			GGML_OP_MAP_CUSTOM2_F32,
			GGML_OP_MAP_CUSTOM3_F32,

			GGML_OP_MAP_CUSTOM1,
			GGML_OP_MAP_CUSTOM2,
			GGML_OP_MAP_CUSTOM3,

			GGML_OP_CROSS_ENTROPY_LOSS,
			GGML_OP_CROSS_ENTROPY_LOSS_BACK,

			GGML_OP_COUNT,
		};

		public enum ggml_unary_op
		{
			GGML_UNARY_OP_ABS,
			GGML_UNARY_OP_SGN,
			GGML_UNARY_OP_NEG,
			GGML_UNARY_OP_STEP,
			GGML_UNARY_OP_TANH,
			GGML_UNARY_OP_ELU,
			GGML_UNARY_OP_RELU,
			GGML_UNARY_OP_SIGMOID,
			GGML_UNARY_OP_GELU,
			GGML_UNARY_OP_GELU_QUICK,
			GGML_UNARY_OP_SILU,
			GGML_UNARY_OP_HARDSWISH,
			GGML_UNARY_OP_HARDSIGMOID,
			// New unary operations in latest ggml
			GGML_UNARY_OP_EXP,
			GGML_UNARY_OP_EXPM1,
			GGML_UNARY_OP_SOFTPLUS,
			GGML_UNARY_OP_GELU_ERF,
			GGML_UNARY_OP_XIELU,
			GGML_UNARY_OP_FLOOR,
			GGML_UNARY_OP_CEIL,
			GGML_UNARY_OP_ROUND,
			GGML_UNARY_OP_TRUNC,
			GGML_UNARY_OP_COUNT,
		};

		public enum ggml_object_type
		{
			GGML_OBJECT_TYPE_TENSOR,
			GGML_OBJECT_TYPE_GRAPH,
			GGML_OBJECT_TYPE_WORK_BUFFER
		};

		public enum ggml_log_level
		{
			GGML_LOG_LEVEL_ERROR = 2,
			GGML_LOG_LEVEL_WARN = 3,
			GGML_LOG_LEVEL_INFO = 4,
			GGML_LOG_LEVEL_DEBUG = 5
		};

		public enum ggml_tensor_flag
		{
			GGML_TENSOR_FLAG_INPUT = 1,
			GGML_TENSOR_FLAG_OUTPUT = 2,
			GGML_TENSOR_FLAG_PARAM = 4,
		};

		public enum ggml_task_type
		{
			GGML_TASK_TYPE_INIT = 0,
			GGML_TASK_TYPE_COMPUTE,
			GGML_TASK_TYPE_FINALIZE,
		};

		public enum ggml_cgraph_eval_order
		{
			GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT = 0,
			GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT,
			GGML_CGRAPH_EVAL_ORDER_COUNT
		};

		public enum ggml_numa_strategy
		{
			GGML_NUMA_STRATEGY_DISABLED = 0,
			GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
			GGML_NUMA_STRATEGY_ISOLATE = 2,
			GGML_NUMA_STRATEGY_NUMACTL = 3,
			GGML_NUMA_STRATEGY_MIRROR = 4,
			GGML_NUMA_STRATEGY_COUNT
		};

		public enum ggml_sort_order
		{
			GGML_SORT_ORDER_ASC,
			GGML_SORT_ORDER_DESC,
		};

		public enum ggml_op_pool
		{
			GGML_OP_POOL_MAX,
			GGML_OP_POOL_AVG,
			GGML_OP_POOL_COUNT,
		};

		/// <summary>
		/// GLU (Gated Linear Unit) operations
		/// </summary>
		public enum ggml_glu_op
		{
			GGML_GLU_OP_REGLU,
			GGML_GLU_OP_GEGLU,
			GGML_GLU_OP_SWIGLU,
			GGML_GLU_OP_SWIGLU_OAI,
			GGML_GLU_OP_GEGLU_ERF,
			GGML_GLU_OP_GEGLU_QUICK,
		};

		/// <summary>
		/// Triangular matrix type
		/// </summary>
		public enum ggml_tri_type
		{
			GGML_TRIANGULAR_LOWER,
			GGML_TRIANGULAR_UPPER,
		};

		public enum ggml_opt_type
		{
			GGML_OPT_TYPE_ADAM,
			GGML_OPT_TYPE_LBFGS,
		};

		/// <summary>
		/// linesearch methods
		/// </summary>
		public enum ggml_linesearch
		{
			GGML_LINESEARCH_DEFAULT = 1,

			GGML_LINESEARCH_BACKTRACKING_ARMIJO = 0,
			GGML_LINESEARCH_BACKTRACKING_WOLFE = 1,
			GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 2,
		};

		/// <summary>
		/// optimization return values
		/// </summary>
		public enum ggml_opt_result
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

		//
		// gguf
		//
		public enum gguf_type
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
		public struct ggml_object
		{
			public size_t offs;
			public size_t size;

			public ggml_object* next;

			public ggml_object_type type;

			public fixed byte padding[4];
		};

		/// <summary>
		/// ggml_tensor structure matching the C implementation exactly
		/// Based on ggml/include/ggml.h v0.9.7-43
		/// Total size: 396 bytes
		/// </summary>
		[StructLayout(LayoutKind.Sequential, Pack = 8)]
		public unsafe struct ggml_tensor
		{
			/// <summary>Offset +0, Size: 4 bytes</summary>
			public ggml_type type;

			/// <summary>Offset +4, Size: 4 bytes (padding for alignment)</summary>
			private int _padding_after_type;

			/// <summary>Offset +8, Size: 8 bytes</summary>
			public ggml_backend_buffer* buffer;

			/// <summary>Offset +16, Size: 32 bytes (4 × 8 bytes)</summary>
			public fixed int64_t ne[GGML_MAX_DIMS]; // number of elements

			/// <summary>Offset +48, Size: 32 bytes (4 × 8 bytes)</summary>
			public fixed size_t nb[GGML_MAX_DIMS]; // stride in bytes:
												   // nb[0] = ggml_type_size(type)
												   // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
												   // nb[i] = nb[i-1] * ne[i-1]

			/// <summary>Offset +80, Size: 4 bytes</summary>
			public ggml_op op; // compute data

			/// <summary>Offset +84, Size: 4 bytes (padding for alignment)</summary>
			private int _padding_after_op;

			/// <summary>Offset +88, Size: 64 bytes (16 × 4 bytes)</summary>
			/// op params - allocated as int32_t for alignment
			public fixed int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];

			/// <summary>Offset +152, Size: 4 bytes</summary>
			public int32_t flags;

			/// <summary>Offset +156, Size: 4 bytes (padding for alignment)</summary>
			private int _padding_after_flags;

			/// <summary>Offset +160, Size: 80 bytes (10 × 8 bytes)</summary>
			/// Source tensors for operations
			/// Matches C implementation: struct ggml_tensor * src[GGML_MAX_SRC]
			public ggml_tensor_array_src src;

			/// <summary>Offset +240, Size: 8 bytes</summary>
			public ggml_tensor* view_src; // source tensor for views

			/// <summary>Offset +248, Size: 8 bytes</summary>
			public size_t view_offs; // offset for views

			/// <summary>Offset +256, Size: 8 bytes</summary>
			public IntPtr data; // pointer to tensor data

			/// <summary>Offset +264, Size: 128 bytes</summary>
			public fixed byte name[GGML_MAX_NAME]; // tensor name

			/// <summary>Offset +392, Size: 8 bytes</summary>
			public IntPtr extra; // extra things e.g. for ggml-cuda.cu

			/// <summary>Offset +400, Size: 8 bytes</summary>
			/// Padding to match C struct size
			public fixed byte padding[8];
		}
		
		/// <summary>
		/// Array of ggml_tensor pointers
		/// Matches C implementation: struct ggml_tensor * src[GGML_MAX_SRC]
		/// Total size: GGML_MAX_SRC × 8 bytes (currently 10 × 8 = 80 bytes)
		///
		/// IMPORTANT: The number of src_N fields MUST exactly match GGML_MAX_SRC.
		/// If GGML_MAX_SRC changes in the future, you MUST add/remove corresponding src_N fields.
		/// </summary>
		[StructLayout(LayoutKind.Sequential, Pack = 8)]
		public unsafe struct ggml_tensor_array_src
		{
			// 当前硬编码的字段数量，必须与 GGML_MAX_SRC 匹配
			private const int LOCAL_FIELD_COUNT = 10;

			/// <summary>Array size derived from GGML_MAX_SRC</summary>
			public const int GGML_MAX_SRC_Size = GGML_MAX_SRC;

			/// <summary>
			/// Array elements stored as ggml_tensor pointers
			/// Field count: LOCAL_FIELD_COUNT (must equal GGML_MAX_SRC)
			/// </summary>
			public ggml_tensor* src_0, src_1, src_2, src_3, src_4, src_5, src_6, src_7, src_8, src_9;

			/// <summary>
			/// Static constructor to validate array size matches GGML_MAX_SRC
			/// Throws exception if mismatch detected
			/// </summary>
			static ggml_tensor_array_src()
			{
				if (LOCAL_FIELD_COUNT != GGML_MAX_SRC)
				{
					throw new InvalidOperationException(
						$"ggml_tensor_array_src field count mismatch! " +
						$"LOCAL_FIELD_COUNT={LOCAL_FIELD_COUNT} but GGML_MAX_SRC={GGML_MAX_SRC}. " +
						$"Please update src_N fields to match GGML_MAX_SRC. " +
						$"Current range: src_0 to src_{LOCAL_FIELD_COUNT - 1}, " +
						$"required range: src_0 to src_{GGML_MAX_SRC - 1}."
					);
				}
			}

			/// <summary>
			/// Gets the total size in bytes based on GGML_MAX_SRC
			/// </summary>
			public static readonly int BytesSize = GGML_MAX_SRC * sizeof(ggml_tensor*);

			/// <summary>
			/// Gets the size of this array (alias for GGML_MAX_SRC_Size)
			/// </summary>
			public int Length => GGML_MAX_SRC_Size;

			/// <summary>
			/// Indexer for array-style access using GGML_MAX_SRC bounds
			/// Matches C array syntax: src[i]
			/// </summary>
			public ggml_tensor* this[int index]
			{
				get
				{
					if (index < 0 || index >= GGML_MAX_SRC_Size)
						throw new ArgumentOutOfRangeException(
							nameof(index),
							index,
							$"Index must be between 0 and {GGML_MAX_SRC_Size - 1} (GGML_MAX_SRC)"
						);

					return index switch
					{
						0 => src_0,
						1 => src_1,
						2 => src_2,
						3 => src_3,
						4 => src_4,
						5 => src_5,
						6 => src_6,
						7 => src_7,
						8 => src_8,
						9 => src_9,
						_ => null
					};
				}
				set
				{
					if (index < 0 || index >= GGML_MAX_SRC_Size)
						throw new ArgumentOutOfRangeException(
							nameof(index),
							index,
							$"Index must be between 0 and {GGML_MAX_SRC_Size - 1} (GGML_MAX_SRC)"
						);

					switch (index)
					{
						case 0: src_0 = value; break;
						case 1: src_1 = value; break;
						case 2: src_2 = value; break;
						case 3: src_3 = value; break;
						case 4: src_4 = value; break;
						case 5: src_5 = value; break;
						case 6: src_6 = value; break;
						case 7: src_7 = value; break;
						case 8: src_8 = value; break;
						case 9: src_9 = value; break;
					}
				}
			}

			/// <summary>
			/// Clear all elements to null within GGML_MAX_SRC range
			/// </summary>
			public void Clear_GGML_MAX_SRC()
			{
				src_0 = src_1 = src_2 = src_3 = src_4 = src_5 = src_6 = src_7 = src_8 = src_9 = null;
			}

			/// <summary>
			/// Copy from another array within GGML_MAX_SRC range
			/// </summary>
			public void CopyFrom_GGML_MAX_SRC(ggml_tensor_array_src other)
			{
				for (int i = 0; i < GGML_MAX_SRC_Size; i++)
				{
					this[i] = other[i];
				}
			}

			/// <summary>
			/// Convert to managed array of ggml_tensor pointers sized by GGML_MAX_SRC
			/// </summary>
			public ggml_tensor*[] ToArray_GGML_MAX_SRC()
			{
				ggml_tensor*[] array = new ggml_tensor*[GGML_MAX_SRC_Size];
				for (int i = 0; i < GGML_MAX_SRC_Size; i++)
				{
					array[i] = this[i];
				}
				return array;
			}

			/// <summary>
			/// Check if all elements are null within GGML_MAX_SRC range
			/// </summary>
			public bool IsAllNull_GGML_MAX_SRC()
			{
				for (int i = 0; i < GGML_MAX_SRC_Size; i++)
				{
					if (this[i] != null)
						return false;
				}
				return true;
			}

			/// <summary>
			/// Count non-null elements within GGML_MAX_SRC range
			/// </summary>
			public int CountNonNull_GGML_MAX_SRC()
			{
				int count = 0;
				for (int i = 0; i < GGML_MAX_SRC_Size; i++)
				{
					if (this[i] != null)
						count++;
				}
				return count;
			}

			/// <summary>
			/// Get string representation showing GGML_MAX_SRC usage
			/// </summary>
			public override string ToString()
			{
				return $"ggml_tensor_array_src[{GGML_MAX_SRC_Size}] (BytesSize={BytesSize})";
			}
		}


		public size_t GGML_TENSOR_SIZE => (ulong)sizeof(ggml_tensor);
		
		public delegate bool ggml_abort_callback(void* data);

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_cplan
		{
			/// <summary>
			/// size of work buffer, calculated by `ggml_graph_plan()`
			/// </summary>
			public size_t work_size;
			/// <summary>
			/// work buffer, to be allocated by caller before calling to `ggml_graph_compute()`
			/// </summary>
			public IntPtr work_data;

			public int n_threads;

			/// <summary>
			/// abort ggml_graph_compute when true
			/// </summary>
			public IntPtr abort_callback;
			public IntPtr abort_callback_data;
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_hash_set
		{
			public size_t size;
			public ggml_tensor** keys;
		}

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_cgraph
		{
			/// <summary>size of the graph</summary>
			public int size;

			/// <summary>number of nodes</summary>
			public int n_nodes;

			/// <summary>number of leaf nodes</summary>
			public int n_leafs;

			/// <summary>[size] graph nodes</summary>
			public ggml_tensor** nodes;

			/// <summary>[size] gradients</summary>
			public ggml_tensor** grads;

			/// <summary>[size] gradient accumulators</summary>
			public ggml_tensor** grad_accs;

			/// <summary>[size] leaf nodes</summary>
			public ggml_tensor** leafs;

			/// <summary>[hash_size] use counts</summary>
			public int32_t* use_counts;

			/// <summary>hash set for visited nodes</summary>
			public ggml_hash_set visited_hash_set;

			/// <summary>evaluation order</summary>
			public ggml_cgraph_eval_order order;
		};

		/// <summary>
		/// scratch buffer
		/// </summary>
		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_scratch
		{
			public size_t offs;
			public size_t size;
			public IntPtr data;
		};

		/// <summary>
		/// memory pool
		/// </summary>
		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_init_params
		{
			/// <summary>
			/// bytes
			/// </summary>
			public size_t mem_size;

			/// <summary>
			/// if NULL, memory will be allocated publicly
			/// </summary>
			public IntPtr mem_buffer;

			/// <summary>
			/// don't allocate memory for the tensor data
			/// </summary>
			public bool no_alloc;
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_compute_params
		{
			public ggml_task_type type;

			/// <summary>
			/// ith = thread index, nth = number of threads
			/// </summary>
			public int ith, nth;

			/// <summary>
			/// work buffer for all threads
			/// </summary>
			public size_t wsize;
			public IntPtr wdata;
		};

		public struct ggml_guid_t
		{
			public byte[] Value;

			public ggml_guid_t(byte[] value)
			{
				if (value.Length != 16)
				{
					throw new ArgumentException("GUID must be 16 bytes long.");
				}
				Value = value;
			}
		}

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_context
		{
			public size_t mem_size;
			public IntPtr mem_buffer;
			public bool mem_buffer_owned;
			public bool no_alloc;

			public int n_objects;

			public ggml_object* objects_begin;
			public ggml_object* objects_end;
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_context_container
		{
			public bool used;

			public ggml_context context;
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_opt_params
		{
			public ggml_opt_type type;

			public size_t graph_size;

			public int n_threads;

			// Delta-based convergence test
			//
			//   if Past == 0 - disabled
			//   if Past > 0:
			//     stop if |f(x) - f(x_past)| < Delta * max(1, |f(x)|)
			//
			public int past;
			public float delta;

			// maximum number of iterations without improvement
			//
			//   if 0 - disabled
			//   if > 0:
			//     assume convergence if no cost improvement in this number of iterations
			//
			public int max_no_improvement;

			[MarshalAs(UnmanagedType.I1)]
			public byte print_forward_graph;
			[MarshalAs(UnmanagedType.I1)]
			public byte print_backward_graph;

			public int n_gradient_accumulation;

			public AdamParams adam;
			public LbfgsParams lbfgs;

			[StructLayout(LayoutKind.Sequential)]
			// ADAM parameters
			public struct AdamParams
			{
				public int n_iter;

				public float sched; // schedule multiplier (fixed, Decay or warmup)
				public float decay; // Weight Decay for AdamW, use 0.0f to disable
				public int decay_min_ndim; // minimum number of tensor dimension to apply Weight Decay
				public float alpha; // learning rate
				public float beta1;
				public float beta2;
				public float eps;   // epsilon for numerical stability
				public float eps_f; // epsilon for convergence test
				public float eps_g; // epsilon for convergence test
				public float gclip; // gradient clipping
			}

			// LBFGS parameters
			[StructLayout(LayoutKind.Sequential, Pack = 8)]
			public struct LbfgsParams
			{
				public int m; // number of corrections to approximate the inv. Hessian
				public int n_iter;
				public int max_linesearch;

				public float eps;      // convergence tolerance
				public float ftol;     // line search tolerance
				public float wolfe;
				public float min_step;
				public float max_step;

				public ggml_linesearch linesearch;
			}
		};



		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_opt_context
		{
			public ggml_context* ctx;
			public ggml_opt_params @params;

			public int iter;

			/// <summary>
			/// number of parameter elements
			/// </summary>
			public int64_t nx;

			public bool just_initialized;

			public float loss_before;
			public float loss_after;
			public Adam adam;
			public Lbfgs lbfgs;

			[StructLayout(LayoutKind.Sequential)]
			public struct Adam
			{
				/// <summary>
				/// current gradient
				/// </summary>
				public ggml_tensor* g;

				/// <summary>
				/// first moment
				/// </summary>
				public ggml_tensor* m;

				/// <summary>
				/// second moment
				/// </summary>
				public ggml_tensor* v;

				/// <summary>
				/// Past function values
				/// </summary>
				public ggml_tensor* pf;
				public float fx_best;
				public float fx_prev;
				public int n_no_improvement;
			}

			[StructLayout(LayoutKind.Sequential)]
			public struct Lbfgs
			{
				/// <summary>
				/// current parameters
				/// </summary>
				public ggml_tensor* x;

				/// <summary>
				/// previous parameters
				/// </summary>
				public ggml_tensor* xp;

				/// <summary>
				/// current gradient
				/// </summary>
				public ggml_tensor* g;

				/// <summary>
				/// previous gradient
				/// </summary>
				public ggml_tensor* gp;

				/// <summary>
				/// search direction
				/// </summary>
				public ggml_tensor* d;

				/// <summary>
				/// Past function values
				/// </summary>
				public ggml_tensor* pf;

				/// <summary>
				/// the L-BFGS memory Alpha
				/// </summary>
				public ggml_tensor* lmal;

				/// <summary>
				/// the L-BFGS memory ys
				/// </summary>
				public ggml_tensor* lmys;

				/// <summary>
				/// the L-BFGS memory s
				/// </summary>
				public ggml_tensor* lms;

				/// <summary>
				/// the L-BFGS memory y
				/// </summary>
				public ggml_tensor* lmy;
				public float fx_best;
				public float step;
				public int j;
				public int k;
				public int end;
				public int n_no_improvement;
			}
		};

		// 警告：以下 gguf_header 和 gguf_context 结构体定义与实际 C++ 实现不匹配！
		// C++ 中的 gguf_context 使用了 std::vector，无法直接在 C# 中通过内存布局访问
		// 请使用 SafeGGufContext 中提供的 C API 函数（如 gguf_get_version, gguf_get_n_kv 等）
		// 保留此定义仅为了兼容性，但强烈建议不要直接使用
		[StructLayout(LayoutKind.Sequential)]
		public struct gguf_header
		{
			public fixed byte magic[4];

			public uint version;

			/// <summary>
			/// GGUFv2
			/// </summary>
			public uint64_t n_tensors;

			/// <summary>
			/// GGUFv2
			/// </summary>
			public uint64_t n_kv;
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct gguf_kv
		{
			public gguf_str key;
			public gguf_type type;
			public gguf_value value;
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct gguf_str
		{
			/// <summary>
			/// GGUFv2
			/// </summary>
			public uint64_t n;
			public IntPtr data;
		};

		[StructLayout(LayoutKind.Explicit, Size = 24)]
		public struct gguf_value
		{
			[FieldOffset(0)] public uint8_t uint8;
			[FieldOffset(0)] public int8_t int8;
			[FieldOffset(0)] public uint16_t uint16;
			[FieldOffset(0)] public int16_t int16;
			[FieldOffset(0)] public uint uint32;
			[FieldOffset(0)] public int32_t int32;
			[FieldOffset(0)] public float float32;
			[FieldOffset(0)] public uint64_t uint64;
			[FieldOffset(0)] public int64_t int64;
			[FieldOffset(0)] public double float64;
			[FieldOffset(0)] public bool bool_;

			[FieldOffset(0)] public gguf_str str;

			[FieldOffset(0)] public arr _arr;

			[StructLayout(LayoutKind.Sequential)]
			public struct arr
			{
				public gguf_type type;

				/// <summary>
				/// GGUFv2
				/// </summary>
				public uint64_t n;
				public IntPtr data;
			}

		};

		[StructLayout(LayoutKind.Sequential)]
		public struct gguf_tensor_info
		{
			public gguf_str name;

			public uint n_dims;
			public fixed uint64_t ne[GGML_MAX_DIMS];

			public ggml_type type;

			/// <summary>
			/// offset from start of `data`, must be a multiple of `ALIGNMENT`
			/// </summary>
			public uint64_t offset;

			// for writing API
			public IntPtr data;
			public size_t size;
		};

		/// <summary>
		/// gguf_context 结构体 - 与 C++ 端内存布局一致
		///
		/// 警告：此结构体仅用于 P/Invoke 互操作，不要直接访问字段！
		/// C++ 端使用 std::vector，其内存布局为 {begin, end, capacity} 三个指针
		/// 请使用 SafeGGufContext 提供的 C API 函数访问数据
		/// </summary>
		[StructLayout(LayoutKind.Sequential)]
		public unsafe struct gguf_context
		{
			public uint version;

			// std::vector<gguf_kv> 的内存布局（3个指针）
			private IntPtr kv_begin;     // 指向数据起始位置
			private IntPtr kv_end;       // 指向数据结束位置
			private IntPtr kv_capacity;  // 容量

			// std::vector<gguf_tensor_info> 的内存布局（3个指针）
			private IntPtr info_begin;   // 指向数据起始位置
			private IntPtr info_end;     // 指向数据结束位置
			private IntPtr info_capacity;// 容量

			public UIntPtr alignment;    // size_t
			public UIntPtr offset;       // offset of `data` from beginning of file
			public UIntPtr size;         // size of `data` in bytes
			public IntPtr data;          // void*
		}

		[StructLayout(LayoutKind.Sequential)]
		public struct gguf_init_params
		{
			public bool no_alloc;

			/// <summary>
			/// if not NULL, create a ggml_context and allocate the tensor data in it
			/// </summary>
			public ggml_context** ctx;
		};


		[StructLayout(LayoutKind.Sequential)]
		public struct gguf_buf
		{
			public IntPtr data;
			public size_t size;
			public size_t offset;
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_bf16_t
		{
			public uint16_t bits;
		}

		#endregion

		#region ggml-backend-impl.h



		/// <summary>
		/// buffer
		/// </summary>


		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_backend_buffer
		{
			public ggml_backend_buffer_type_i iface;
			public ggml_backend_buffer_type* buft;
			public IntPtr context;
			public size_t size;
			public ggml_backend_buffer_usage usage;
		};

		public delegate string get_name(ggml_backend_buffer* buft);
		public delegate ggml_backend_buffer* alloc_buffer(ggml_backend_buffer_type* buft, size_t size);
		public delegate size_t get_alignment(ggml_backend_buffer_type* buft);
		public delegate size_t get_max_size(ggml_backend_buffer_type* buft);
		public delegate size_t get_alloc_size(ggml_backend_buffer_type* buft, ggml_tensor* tensor);
		public delegate bool supports_backend(ggml_backend_buffer_type* buft, ggml_backend* backend);
		public delegate bool is_host(ggml_backend_buffer_type* buft);

		public delegate void free_buffer(ggml_backend_buffer* buffer);
		public delegate IntPtr get_base(ggml_backend_buffer* buffer);
		public delegate void init_tensor(ggml_backend_buffer* buffer, ggml_tensor* tensor);
		public delegate void set_tensor(ggml_backend_buffer* buffer, ggml_tensor* tensor, IntPtr data, size_t offset, size_t size);
		public delegate void get_tensor(ggml_backend_buffer* buffer, ggml_tensor* tensor, IntPtr data, size_t offset, size_t size);

		/// <summary>
		/// // dst is in the buffer, src may be in any buffer
		/// </summary>
		/// <param name="buffer"></param>
		/// <param name="src"></param>
		/// <param name="dst"></param>
		/// <returns></returns>
		public delegate bool cpy_tensor(ggml_backend_buffer* buffer, ggml_tensor* src, ggml_tensor* dst);
		public delegate void clear(ggml_backend_buffer* buffer, uint8_t value);

		/// <summary>
		/// // reset any internal state due to tensor initialization, such as tensor extras
		/// </summary>
		/// <param name="buffer"></param>
		public delegate void reset(ggml_backend_buffer* buffer);

		public delegate void free(ggml_backend* backend);

		/// <summary>
		/// buffer allocation
		/// </summary>
		/// <param name="backend"></param>
		/// <returns></returns>
		public delegate ggml_backend_buffer_type* get_default_buffer_type(ggml_backend* backend);

		/// <summary>
		/// ( ) asynchronous tensor data access
		/// </summary>
		/// <param name="backend"></param>
		/// <param name="tensor"></param>
		/// <param name="data"></param>
		/// <param name="offset"></param>
		/// <param name="size"></param>
		public delegate void set_tensor_async(ggml_backend* backend, ggml_tensor* tensor, IntPtr data, size_t offset, size_t size);
		public delegate void get_tensor_async(ggml_backend* backend, ggml_tensor* tensor, IntPtr data, size_t offset, size_t size);
		public delegate void cpy_tensor_async(ggml_backend* backend_src, ggml_backend* backend_dst, ggml_tensor* src, ggml_tensor* dst);

		/// <summary>
		/// ( ) complete all pending operations
		/// </summary>
		/// <param name="backend"></param>
		public delegate void synchronize(ggml_backend* backend);

		/// <summary>
		/// compute graph with a plan (not used currently)
		/// </summary>
		/// <param name="backend"></param>
		/// <param name="cgraph"></param>
		/// <returns></returns>
		public delegate ggml_backend_graph_plan_t graph_plan_create(ggml_backend* backend, ggml_cgraph* cgraph);
		public delegate void graph_plan_free(ggml_backend* backend, ggml_backend_graph_plan_t plan);

		/// <summary>
		/// compute graph with a plan
		/// </summary>
		/// <param name="backend"></param>
		/// <param name="plan"></param>
		/// <returns></returns>
		public delegate ggml_status graph_plan_compute(ggml_backend* backend, ggml_backend_graph_plan_t plan);

		/// <summary>
		/// compute graph without a plan (async)
		/// </summary>
		/// <param name="backend"></param>
		/// <param name="cgraph"></param>
		/// <returns></returns>
		public delegate ggml_status graph_compute(ggml_backend* backend, ggml_cgraph* cgraph);

		/// <summary>
		/// check if the backend supports an operation
		/// </summary>
		/// <param name="backend"></param>
		/// <param name="op"></param>
		/// <returns></returns>
		public delegate bool supports_op(ggml_backend* backend, ggml_tensor* op);

		// check if the backend wants to run an operation, even if the weights are allocated in a CPU buffer
		// these should be expensive operations with large batch sizes that may benefit from running on this backend
		// even if the Weight has to be copied from the CPU temporarily
		public delegate void offload_op(ggml_backend* backend, ggml_tensor* op);

		/// <summary>
		/// ( ) event synchronization
		/// </summary>
		/// <param name="backend"></param>
		/// <returns></returns>
		public delegate ggml_backend_event* event_new(ggml_backend* backend);

		public delegate void event_free(ggml_backend_event* @event);
		public delegate void event_record(ggml_backend_event* @event);

		public delegate void event_wait(ggml_backend* backend, ggml_backend_event* @event);

		public delegate void event_synchronize(ggml_backend_event* @event);


		public struct ggml_backend_buffer_type_i
		{
			public get_name get_name;
			public alloc_buffer alloc_buffer;
			public get_alignment get_alignment;
			public get_max_size get_max_size;
			public get_alloc_size get_alloc_size;
			public supports_backend support_backend;
			public is_host is_hose;

		};

		// Backend device interface
		[StructLayout(LayoutKind.Sequential)]
		public unsafe struct ggml_backend_device_i
		{
			// device name: short identifier for this device
			public IntPtr get_name; // function pointer - will be set at runtime
			// device description
			public IntPtr get_description;
			// device memory
			public IntPtr get_memory;
			// device type
			public IntPtr get_type;
			// device properties
			public IntPtr get_props;
			// backend initialization
			public IntPtr init_backend;
			// preferred buffer type
			public IntPtr get_buffer_type;
			// host buffer type
			public IntPtr get_host_buffer_type;
			// buffer from pointer
			public IntPtr buffer_from_host_ptr;
			// check if supports operation
			public IntPtr supports_op;
			// check if supports buffer type
			public IntPtr supports_buft;
			// offload operation
			public IntPtr offload_op;
			// event synchronization
			public IntPtr event_new;
			public IntPtr event_free;
			public IntPtr event_synchronize;
		};

		// Backend device
		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_backend_device
		{
			public ggml_backend_device_i iface;
			public ggml_backend_reg* reg;
			public IntPtr context;
		};

		// Backend reg interface
		[StructLayout(LayoutKind.Sequential)]
		public unsafe struct ggml_backend_reg_i
		{
			public IntPtr get_name;
			public IntPtr get_device_count;
			public IntPtr get_device;
			public IntPtr get_proc_address;
		};

		// Backend reg
		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_backend_reg
		{
			public int api_version;
			public ggml_backend_reg_i iface;
			public IntPtr context;
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_backend_buffer_type
		{
			public ggml_backend_buffer_type_i iface;
			public ggml_backend_device* device;
			public IntPtr context;
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_backend
		{
			public ggml_guid_t guid;

			public ggml_backend_i iface;
			public ggml_backend_device* device;
			public IntPtr context;
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_backend_buffer_i
		{
			public get_name get_name;
			public free_buffer free_buffer;
			public get_base get_base;
			public init_tensor init_tensor;
			public set_tensor set_tensor;
			public get_tensor get_tensor;
			public cpy_tensor cpy_tensor; // dst is in the buffer, src may be in any buffer
			public clear clear;
			public reset reset; // reset any internal state due to tensor initialization, such as tensor extras
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_backend_i
		{
			public get_name get_name;

			public free free;

			// buffer allocation
			public get_default_buffer_type get_default_buffer_type;

			// ( ) asynchronous tensor data access
			public set_tensor_async set_tensor_async;
			public get_tensor_async get_tensor_async;
			public cpy_tensor_async cpy_tensor_async;

			// ( ) complete all pending operations
			public synchronize synchronize;

			// compute graph with a plan (not used currently)
			public graph_plan_create graph_plan_create;
			public graph_plan_free graph_plan_free;

			// compute graph with a plan
			public graph_plan_compute graph_plan_compute;
			// compute graph without a plan (async)
			public graph_compute graph_compute;

			// check if the backend supports an operation
			public supports_op supports_op;

			// check if the backend wants to run an operation, even if the weights are allocated in a CPU buffer
			// these should be expensive operations with large batch sizes that may benefit from running on this backend
			// even if the Weight has to be copied from the CPU temporarily
			public offload_op offload_op;

			// ( ) event synchronization
			public event_new event_new;
			public event_free event_free;

			public event_record event_record;

			public event_wait event_wait;

			public event_synchronize event_synchronize;
		}

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_backend_event
		{
			public ggml_backend_device* device;
			public IntPtr context;
		};


		#endregion


		#region ggml-backend.h




		public delegate bool ggml_backend_sched_eval_callback(ggml_tensor* t, bool ask, IntPtr user_data);
		public delegate bool ggml_backend_eval_callback(int node_index, ggml_tensor* t1, ggml_tensor* t2, IntPtr user_data);

		/// <summary>
		/// buffer
		/// </summary>
		public enum ggml_backend_buffer_usage
		{
			GGML_BACKEND_BUFFER_USAGE_ANY = 0,
			GGML_BACKEND_BUFFER_USAGE_WEIGHTS = 1,
		};

		/// <summary>
		/// Backend device type
		/// </summary>
		public enum ggml_backend_dev_type
		{
			/// <summary>CPU device using system memory</summary>
			GGML_BACKEND_DEVICE_TYPE_CPU,
			/// <summary>GPU device using dedicated memory</summary>
			GGML_BACKEND_DEVICE_TYPE_GPU,
			/// <summary>integrated GPU device using host memory</summary>
			GGML_BACKEND_DEVICE_TYPE_IGPU,
			/// <summary>accelerator devices intended to be used together with the CPU backend (e.g. BLAS or AMX)</summary>
			GGML_BACKEND_DEVICE_TYPE_ACCEL
		};

		/// <summary>
		/// Functionality supported by the device
		/// </summary>
		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_backend_dev_caps
		{
			/// <summary>asynchronous operations</summary>
			public bool async;
			/// <summary>pinned host buffer</summary>
			public bool host_buffer;
			/// <summary>creating buffers from host ptr</summary>
			public bool buffer_from_host_ptr;
			/// <summary>event synchronization</summary>
			public bool events;
		};

		/// <summary>
		/// All the device properties
		/// </summary>
		[StructLayout(LayoutKind.Sequential)]
		public unsafe struct ggml_backend_dev_props
		{
			/// <summary>device name</summary>
			public IntPtr name; // const char *
			/// <summary>device description</summary>
			public IntPtr description; // const char *
			/// <summary>device free memory in bytes</summary>
			public size_t memory_free;
			/// <summary>device total memory in bytes</summary>
			public size_t memory_total;
			/// <summary>device type</summary>
			public ggml_backend_dev_type type;
			/// <summary>
			/// device id:
			///   for PCI devices, this should be the PCI bus id formatted as "domain:bus:device.function" (e.g. "0000:01:00.0")
			///   if the id is unknown, this should be NULL
			/// </summary>
			public IntPtr device_id; // const char *
			/// <summary>device capabilities</summary>
			public ggml_backend_dev_caps caps;

			/// <summary>
			/// Gets the device_id as a string (returns null if device_id is NULL)
			/// </summary>
			public string? DeviceIdString
			{
				get
				{
					if (device_id == IntPtr.Zero)
						return null;
					return System.Runtime.InteropServices.Marshal.PtrToStringAnsi(device_id);
				}
			}

			/// <summary>
			/// Gets the name as a string (returns null if name is NULL)
			/// </summary>
			public string? NameString
			{
				get
				{
					if (name == IntPtr.Zero)
						return null;
					return System.Runtime.InteropServices.Marshal.PtrToStringAnsi(name);
				}
			}

			/// <summary>
			/// Gets the description as a string (returns null if description is NULL)
			/// </summary>
			public string? DescriptionString
			{
				get
				{
					if (description == IntPtr.Zero)
						return null;
					return System.Runtime.InteropServices.Marshal.PtrToStringAnsi(description);
				}
			}
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_backend_graph_copy
		{
			ggml_backend_buffer* buffer;
			ggml_context* ctx_allocated;
			ggml_context* ctx_unallocated;
			ggml_cgraph* graph;
		};

		#endregion


		#region ggml-alloc.h

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_tallocr
		{
			public ggml_backend_buffer* buffer;
			public IntPtr @base;
			public size_t alignment;
			public size_t offset;
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_gallocr
		{
			public ggml_backend_buffer_type** bufts; // [n_buffers]
			public ggml_backend_buffer** buffers; // [n_buffers]
			public ggml_dyn_tallocr** buf_tallocs; // [n_buffers]
			public int n_buffers;

			public ggml_hash_set hash_set;
			public hash_node* hash_values; // [hash_set.size]

			public node_alloc* node_allocs; // [n_nodes]
			public int n_nodes;

			public leaf_alloc* leaf_allocs; // [n_leafs]
			public int n_leafs;
		};

		[StructLayout(LayoutKind.Explicit)]
		public struct ggml_dyn_tallocr
		{
			[FieldOffset(0)] public size_t alignment;
			[FieldOffset(8)] public int n_free_blocks;
			[FieldOffset(16)] public free_block[] free_block;  // free_block's count is  [MAX_FREE_BLOCKS] (256)
			[FieldOffset(4112)] public size_t max_size;
		}

		[StructLayout(LayoutKind.Sequential)]
		public struct free_block
		{
			public size_t offset;
			public size_t size;
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct hash_node
		{
			public int n_children;
			public int n_views;
			public int buffer_id;
			public size_t offset; // offset within the buffer
			public bool allocated;
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct leaf_alloc
		{
			public int buffer_id;
			public tensor_alloc leaf;
		};

		[StructLayout(LayoutKind.Explicit)]
		public struct node_alloc
		{
			[FieldOffset(0)] public int buffer_id;
			[FieldOffset(8)] tensor_alloc dst;
			[FieldOffset(24)] tensor_alloc[] src;  //  src' count is  [GGML_MAX_SRC] (10)
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct tensor_alloc
		{
			public size_t offset;
			public size_t size_max; // 0 = pre-allocated, unused, or view
		};

		#endregion


	}
}

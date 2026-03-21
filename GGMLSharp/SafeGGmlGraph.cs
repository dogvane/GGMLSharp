using System;
using static GGMLSharp.InternalStructs;

namespace GGMLSharp
{
	/// <summary>
	/// Managed handle for a ggml computation graph.
	/// A graph stores the nodes, leaves, and optional gradient links required to execute
	/// a forward pass and, when built, a backward pass.
	/// </summary>
	public unsafe class SafeGGmlGraph : SafeGGmlHandleBase
	{
		/// <summary>
		/// Returns true when the graph points to a valid native ggml_cgraph.
		/// </summary>
		public bool IsInitialized => handle != IntPtr.Zero;
		private ggml_cgraph* graph => (ggml_cgraph*)handle;

		/// <summary>
		/// Number of computation nodes currently stored in the graph.
		/// </summary>
		public int NodeCount
		{
			get { return graph->n_nodes; }
			set { graph->n_nodes = value; }
		}

		/// <summary>
		/// Number of leaf tensors currently stored in the graph.
		/// Leafs are external inputs/constants referenced by graph nodes.
		/// </summary>
		public int LeafCount
		{
			get { return graph->n_leafs; }
			set { graph->n_leafs = value; }
		}

		//public long TimeUse => graph->perf_time_us;

		//public long Cycles => graph->perf_cycles;

		//public int Runs => graph->perf_runs;

		/// <summary>
		/// Evaluation order used by the native graph.
		/// </summary>
		public Structs.GGmlGraphEvalOrder EvalOrder => (Structs.GGmlGraphEvalOrder)graph->order;

		private void ThrowIfNotInitialized()
		{
			if (!IsInitialized)
			{
				throw new ObjectDisposedException("Not initialized or disposed");
			}
		}

		/// <summary>
		/// Creates an empty graph handle.
		/// </summary>
		public SafeGGmlGraph()
		{
			this.handle = IntPtr.Zero;
		}

		/// <summary>
		/// Returns a managed snapshot of all graph nodes.
		/// </summary>
		public SafeGGmlTensor[] Nodes
		{
			get
			{
				SafeGGmlTensor[] nodes = new SafeGGmlTensor[NodeCount];
				for (int i = 0; i < NodeCount; i++)
				{
					nodes[i] = new SafeGGmlTensor((IntPtr)graph->nodes[i]);
				}
				return nodes;
			}
		}

		/// <summary>
		/// Returns the node tensor at the given graph index.
		/// </summary>
		public SafeGGmlTensor GetNode(int index)
		{
			ThrowIfNotInitialized();
			return new SafeGGmlTensor((IntPtr)graph->nodes[index]);
		}

		/// <summary>
		/// Returns a managed snapshot of all graph leaf tensors.
		/// </summary>
		public SafeGGmlTensor[] Leafs
		{
			get
			{
				SafeGGmlTensor[] leafs = new SafeGGmlTensor[LeafCount];
				for (int i = 0; i < LeafCount; i++)
				{
					leafs[i] = new SafeGGmlTensor((IntPtr)graph->leafs[i]);
				}
				return leafs;
			}
		}

		/// <summary>
		/// Returns the gradient tensors associated with graph nodes when the graph has backward data.
		/// Returns null when gradients have not been built.
		/// </summary>
		public SafeGGmlTensor[] Grads
		{
			get
			{
				if (graph->grads != null)
				{
					SafeGGmlTensor[] grads = new SafeGGmlTensor[NodeCount];
					for (int i = 0; i < NodeCount; i++)
					{
						grads[i] = new SafeGGmlTensor((IntPtr)graph->grads[i]);
					}
					return grads;
				}
				else
				{
					return null;
				}
			}
		}

		/// <summary>
		/// Computes the graph using the legacy ggml context execution path.
		/// </summary>
		public Structs.GGmlStatus ComputeWithGGmlContext(SafeGGmlContext context, int threads)
		{
			ThrowIfNotInitialized();
			return (Structs.GGmlStatus)Native.ggml_graph_compute_with_ctx(context, this, threads);
		}

		/// <summary>
		/// Computes the graph using an explicit execution plan.
		/// </summary>
		public Structs.GGmlStatus Compute(SafeGGmlGraphPlan plan)
		{
			ThrowIfNotInitialized();
			ggml_cplan* p = (ggml_cplan*)plan.DangerousGetHandle();
			return (Structs.GGmlStatus)Native.ggml_graph_compute(this, p);
		}

		/// <summary>
		/// Expands the forward graph so that it includes all dependencies needed to produce <paramref name="tensor"/>.
		/// </summary>
		public void BuildForwardExpend(SafeGGmlTensor tensor)
		{
			ThrowIfNotInitialized();
			Native.ggml_build_forward_expand(this, tensor);
		}

		/// <summary>
		/// Allocates storage for this graph through a graph allocator.
		/// </summary>
		public bool GraphAllocate(SafeGGmlGraphAllocr allocr)
		{
			ThrowIfNotInitialized();
			return Native.ggml_gallocr_alloc_graph(allocr, this);
		}

		/// <summary>
		/// Computes the graph on a specific backend such as CPU, CUDA, or Vulkan.
		/// </summary>
		public Structs.GGmlStatus BackendCompute(SafeGGmlBackend backend)
		{
			ThrowIfNotInitialized();
			return (Structs.GGmlStatus)Native.ggml_backend_graph_compute(backend, this);
		}

		/// <summary>
		/// Looks up a tensor in the graph by its native ggml name.
		/// </summary>
		public SafeGGmlTensor GetTensor(string name)
		{
			ThrowIfNotInitialized();
			return Native.ggml_graph_get_tensor(this, name);
		}

		/// <summary>
		/// Resets graph execution state in the native runtime.
		/// </summary>
		public void Reset()
		{
			ThrowIfNotInitialized();
			Native.ggml_graph_reset(this);
		}

		/// <summary>
		/// Exports the graph to a file using ggml's native graph export support.
		/// </summary>
		public void Export(string name)
		{
			ThrowIfNotInitialized();
			Native.ggml_graph_export(this, name);
		}

		/// <summary>
		/// Builds an execution plan for the graph.
		/// </summary>
		public SafeGGmlGraphPlan GetPlan(int threads = -1)
		{
			ggml_cplan p = Native.ggml_graph_plan(this, threads);
			SafeGGmlGraphPlan plan = new SafeGGmlGraphPlan(p);
			return plan;
		}

		/// <summary>
		/// Reserves graph-allocator buffers for this graph shape without executing it.
		/// Useful when reusing a graph allocator across multiple runs.
		/// </summary>
		public bool Reserve(SafeGGmlGraphAllocr allocr)
		{
			ThrowIfNotInitialized();
			return Native.ggml_gallocr_reserve(allocr, this);
		}

		/// <summary>
		/// Builds backward graph data using gradient checkpointing.
		/// </summary>
		public void BuildBackwardGradientCheckpointing(SafeGGmlContext ctx, SafeGGmlGraph gb, SafeGGmlGraph temp, SafeGGmlTensor[] checkpoints)
		{
			ThrowIfNotInitialized();
			Native.ggml_build_backward_gradient_checkpointing(ctx, this, gb, temp, checkpoints, checkpoints.Length);
		}

		/// <summary>
		/// Copies graph contents from <paramref name="gf"/> into this graph.
		/// </summary>
		public void Copy(SafeGGmlGraph gf)
		{
			ThrowIfNotInitialized();
			Native.ggml_graph_cpy(this, gf);
		}

		/// <summary>
		/// Expands the backward graph for the current forward graph.
		/// </summary>
		public void BuildBackwardExpend(SafeGGmlContext ctx, SafeGGmlGraph gb, bool keep)
		{
			ThrowIfNotInitialized();
			Native.ggml_build_backward_expand(ctx, this, gb, keep);
		}

		/// <summary>
		/// Prints the graph through ggml's native debug output.
		/// </summary>
		public void Print()
		{
			ThrowIfNotInitialized();
			Native.ggml_graph_print(this);
		}

	}
}


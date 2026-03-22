using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using static GGMLSharp.InternalStructs;
using static GGMLSharp.Structs;

namespace GGMLSharp
{
    /// <summary>
    /// Wrapper for ggml_opt_context from ggml-opt.h
    /// Provides high-level interface for training models using GGML
    /// </summary>
    public unsafe class SafeGGmlOptContext : SafeGGmlHandleBase
    {
        private ggml_opt_context* context => (ggml_opt_context*)handle;

        private bool IsInitialized => handle != IntPtr.Zero;

        /// <summary>
        /// Gets the native handle for this optimizer context (useful for interop with other native functions)
        /// </summary>
        public IntPtr NativeHandle => handle;

        private void ThrowIfNotInitialized()
        {
            if (!IsInitialized)
            {
                throw new ObjectDisposedException("SafeGGmlOptContext", "Not initialized or disposed");
            }
        }

        public SafeGGmlOptContext()
        {
            this.handle = IntPtr.Zero;
        }

        public SafeGGmlOptContext(IntPtr handle) : base(handle, true)
        {
        }

        // Properties from ggml_opt_context
        public long NX => context->nx;
        public float LossBefore => context->loss_before;
        public float LossAfter => context->loss_after;
        public int Iter => context->iter;
        public bool JustInitialized => context->just_initialized;

        /// <summary>
        /// Initialize optimizer context with parameters
        /// </summary>
        public static SafeGGmlOptContext Init(GGmlOptParams opt_params)
        {
            IntPtr ctx = Native.ggml_opt_init(opt_params);
            if (ctx == IntPtr.Zero)
                throw new InvalidOperationException("Failed to initialize optimizer context");
            return new SafeGGmlOptContext(ctx);
        }

        public static GGmlOptParams DefaultParams(SafeGGmlBackendScheduler backendScheduler, GGmlOptLossType lossType)
        {
            return Native.ggml_opt_default_params(backendScheduler?.NativeHandle ?? IntPtr.Zero, lossType);
        }

        public static GGmlOptOptimizerParams GetDefaultOptimizerParams(IntPtr userdata = default)
        {
            return Native.ggml_opt_get_default_optimizer_params(userdata);
        }

        public static GGmlOptOptimizerParams GetConstantOptimizerParams(IntPtr userdata = default)
        {
            return Native.ggml_opt_get_constant_optimizer_params(userdata);
        }

        public static void Fit(
            SafeGGmlBackendScheduler backendScheduler,
            SafeGGmlContext computeContext,
            SafeGGmlTensor inputs,
            SafeGGmlTensor outputs,
            SafeGGmlDataset dataset,
            GGmlOptLossType lossType,
            GGmlOptOptimizerType optimizer,
            GGmlOptGetOptimizerParamsDelegate getOptParams,
            long epochCount,
            long logicalBatchSize,
            float validationSplit,
            bool silent)
        {
            if (computeContext == null || computeContext.IsInvalid)
                throw new ArgumentNullException(nameof(computeContext));
            if (inputs == null || inputs.IsInvalid)
                throw new ArgumentNullException(nameof(inputs));
            if (outputs == null || outputs.IsInvalid)
                throw new ArgumentNullException(nameof(outputs));
            if (dataset == null || dataset.IsInvalid)
                throw new ArgumentNullException(nameof(dataset));

            IntPtr callbackPtr = getOptParams != null
                ? Marshal.GetFunctionPointerForDelegate(getOptParams)
                : IntPtr.Zero;

            Native.ggml_opt_fit(
                backendScheduler?.NativeHandle ?? IntPtr.Zero,
                computeContext.NativeHandle,
                inputs.NativeHandle,
                outputs.NativeHandle,
                dataset.NativeHandle,
                lossType,
                optimizer,
                callbackPtr,
                epochCount,
                logicalBatchSize,
                validationSplit,
                silent);

            GC.KeepAlive(getOptParams);
        }

        /// <summary>
        /// Free the optimizer context
        /// </summary>
        public void Free()
        {
            if (IsInitialized)
            {
                Native.ggml_opt_free(handle);
                handle = IntPtr.Zero;
            }
        }

        /// <summary>
        /// Reset gradients to zero, initialize loss, and optionally reset the optimizer
        /// </summary>
        public void Reset(bool resetOptimizer = false)
        {
            ThrowIfNotInitialized();
            Native.ggml_opt_reset(handle, resetOptimizer);
        }

        /// <summary>
        /// Whether the graphs are allocated statically
        /// </summary>
        public bool HasStaticGraphs
        {
            get
            {
                ThrowIfNotInitialized();
                return Native.ggml_opt_static_graphs(handle);
            }
        }

        /// <summary>
        /// Get forward graph input tensor
        /// </summary>
        public SafeGGmlTensor GetInputs()
        {
            ThrowIfNotInitialized();
            IntPtr tensor = Native.ggml_opt_inputs(handle);
            return tensor != IntPtr.Zero ? new SafeGGmlTensor(tensor) : null;
        }

        /// <summary>
        /// Get forward graph output tensor
        /// </summary>
        public SafeGGmlTensor GetOutputs()
        {
            ThrowIfNotInitialized();
            IntPtr tensor = Native.ggml_opt_outputs(handle);
            return tensor != IntPtr.Zero ? new SafeGGmlTensor(tensor) : null;
        }

        /// <summary>
        /// Get labels tensor
        /// </summary>
        public SafeGGmlTensor GetLabels()
        {
            ThrowIfNotInitialized();
            IntPtr tensor = Native.ggml_opt_labels(handle);
            return tensor != IntPtr.Zero ? new SafeGGmlTensor(tensor) : null;
        }

        /// <summary>
        /// Get loss tensor
        /// </summary>
        public SafeGGmlTensor GetLoss()
        {
            ThrowIfNotInitialized();
            IntPtr tensor = Native.ggml_opt_loss(handle);
            return tensor != IntPtr.Zero ? new SafeGGmlTensor(tensor) : null;
        }

        /// <summary>
        /// Get predictions tensor
        /// </summary>
        public SafeGGmlTensor GetPred()
        {
            ThrowIfNotInitialized();
            IntPtr tensor = Native.ggml_opt_pred(handle);
            return tensor != IntPtr.Zero ? new SafeGGmlTensor(tensor) : null;
        }

        /// <summary>
        /// Get number of correct predictions tensor
        /// </summary>
        public SafeGGmlTensor GetNCorrect()
        {
            ThrowIfNotInitialized();
            IntPtr tensor = Native.ggml_opt_ncorrect(handle);
            return tensor != IntPtr.Zero ? new SafeGGmlTensor(tensor) : null;
        }

        /// <summary>
        /// Get the gradient accumulator for a node from the forward graph
        /// </summary>
        public SafeGGmlTensor GetGradAcc(IntPtr nodeHandle)
        {
            ThrowIfNotInitialized();
            if (nodeHandle == IntPtr.Zero)
                throw new ArgumentNullException(nameof(nodeHandle));

            IntPtr tensor = Native.ggml_opt_grad_acc(handle, nodeHandle);
            return tensor != IntPtr.Zero ? new SafeGGmlTensor(tensor) : null;
        }

        /// <summary>
        /// Get the gradient accumulator for a node from the forward graph (overload for SafeGGmlTensor)
        /// </summary>
        public SafeGGmlTensor GetGradAcc(SafeGGmlTensor node)
        {
            if (node == null || node.IsInvalid)
                throw new ArgumentNullException(nameof(node));

            // Use the NativeHandle property to access the handle
            return GetGradAcc(node.NativeHandle);
        }

        /// <summary>
        /// Get the optimizer type for this context
        /// </summary>
        public GGmlOptOptimizerType GetOptimizerType()
        {
            ThrowIfNotInitialized();
            return Native.ggml_opt_context_optimizer_type(handle);
        }

        /// <summary>
        /// Allocate the next graph for evaluation, either forward or forward + backward
        /// Must be called exactly once prior to calling Eval
        /// </summary>
        public void Alloc(bool backward = true)
        {
            ThrowIfNotInitialized();
            Native.ggml_opt_alloc(handle, backward);
        }

        /// <summary>
        /// Do forward pass, increment result if not NULL, do backward pass if allocated
        /// </summary>
        public void Eval(SafeGGmlOptResult result = null)
        {
            ThrowIfNotInitialized();
            Native.ggml_opt_eval(handle, result?.NativeHandle ?? IntPtr.Zero);
        }

        /// <summary>
        /// Run a full epoch with the given dataset
        /// </summary>
        public void Epoch(SafeGGmlDataset dataset, SafeGGmlOptResult resultTrain,
            SafeGGmlOptResult resultEval, long idataSplit = 0)
        {
            ThrowIfNotInitialized();
            if (dataset == null || dataset.IsInvalid)
                throw new ArgumentNullException(nameof(dataset));

            Native.ggml_opt_epoch(handle, dataset.NativeHandle,
                resultTrain?.NativeHandle ?? IntPtr.Zero,
                resultEval?.NativeHandle ?? IntPtr.Zero,
                idataSplit, IntPtr.Zero, IntPtr.Zero);
        }

        /// <summary>
        /// Get optimizer name as string
        /// </summary>
        public static string GetOptimizerName(GGmlOptOptimizerType optimizer)
        {
            IntPtr namePtr = Native.ggml_opt_optimizer_name(optimizer);
            return Marshal.PtrToStringAnsi(namePtr);
        }
    }

    /// <summary>
    /// Wrapper for ggml_opt_dataset from ggml-opt.h
    /// </summary>
    public unsafe class SafeGGmlDataset : SafeGGmlHandleBase
    {
        private bool IsInitialized => handle != IntPtr.Zero;

        /// <summary>
        /// Gets the native handle for this dataset (useful for interop with other native functions)
        /// </summary>
        public IntPtr NativeHandle => handle;

        private void ThrowIfNotInitialized()
        {
            if (!IsInitialized)
            {
                throw new ObjectDisposedException("SafeGGmlDataset", "Not initialized or disposed");
            }
        }

        public SafeGGmlDataset()
        {
            this.handle = IntPtr.Zero;
        }

        public SafeGGmlDataset(IntPtr handle) : base(handle, true)
        {
        }

        /// <summary>
        /// Initialize a dataset
        /// </summary>
        public static SafeGGmlDataset Init(GGmlType typeData, GGmlType typeLabel,
            long neDatapoint, long neLabel, long ndata, long ndataShard)
        {
            IntPtr dataset = Native.ggml_opt_dataset_init(typeData, typeLabel,
                neDatapoint, neLabel, ndata, ndataShard);
            if (dataset == IntPtr.Zero)
                throw new InvalidOperationException("Failed to initialize dataset");
            return new SafeGGmlDataset(dataset);
        }

        /// <summary>
        /// Free the dataset
        /// </summary>
        public void Free()
        {
            if (IsInitialized)
            {
                Native.ggml_opt_dataset_free(handle);
                handle = IntPtr.Zero;
            }
        }

        /// <summary>
        /// Get the total number of datapoints
        /// </summary>
        public long GetNData()
        {
            ThrowIfNotInitialized();
            return Native.ggml_opt_dataset_ndata(handle);
        }

        /// <summary>
        /// Get the underlying data tensor
        /// </summary>
        public SafeGGmlTensor GetData()
        {
            ThrowIfNotInitialized();
            IntPtr tensor = Native.ggml_opt_dataset_data(handle);
            return tensor != IntPtr.Zero ? new SafeGGmlTensor(tensor) : null;
        }

        /// <summary>
        /// Get the underlying labels tensor
        /// </summary>
        public SafeGGmlTensor GetLabels()
        {
            ThrowIfNotInitialized();
            IntPtr tensor = Native.ggml_opt_dataset_labels(handle);
            return tensor != IntPtr.Zero ? new SafeGGmlTensor(tensor) : null;
        }

        /// <summary>
        /// Shuffle idata first datapoints from dataset with RNG from opt_ctx
        /// Shuffle all datapoints if idata is negative
        /// </summary>
        public void Shuffle(SafeGGmlOptContext optCtx, long idata = -1)
        {
            ThrowIfNotInitialized();
            if (optCtx == null || optCtx.IsInvalid)
                throw new ArgumentNullException(nameof(optCtx));

            Native.ggml_opt_dataset_shuffle(optCtx.NativeHandle, handle, idata);
        }

        /// <summary>
        /// Get batch at position ibatch from dataset
        /// </summary>
        public void GetBatch(SafeGGmlTensor dataBatch, SafeGGmlTensor labelsBatch, long ibatch)
        {
            ThrowIfNotInitialized();
            if (dataBatch == null || dataBatch.IsInvalid)
                throw new ArgumentNullException(nameof(dataBatch));
            if (labelsBatch == null || labelsBatch.IsInvalid)
                throw new ArgumentNullException(nameof(labelsBatch));

            Native.ggml_opt_dataset_get_batch(handle, dataBatch.NativeHandle, labelsBatch.NativeHandle, ibatch);
        }
    }

    /// <summary>
    /// Wrapper for ggml_opt_result from ggml-opt.h
    /// </summary>
    public unsafe class SafeGGmlOptResult : SafeGGmlHandleBase
    {
        private bool IsInitialized => handle != IntPtr.Zero;

        /// <summary>
        /// Gets the native handle for this result (useful for interop with other native functions)
        /// </summary>
        public IntPtr NativeHandle => handle;

        private void ThrowIfNotInitialized()
        {
            if (!IsInitialized)
            {
                throw new ObjectDisposedException("SafeGGmlOptResult", "Not initialized or disposed");
            }
        }

        public SafeGGmlOptResult()
        {
            this.handle = IntPtr.Zero;
        }

        public SafeGGmlOptResult(IntPtr handle) : base(handle, true)
        {
        }

        /// <summary>
        /// Initialize a new result
        /// </summary>
        public static SafeGGmlOptResult Init()
        {
            IntPtr result = Native.ggml_opt_result_init();
            if (result == IntPtr.Zero)
                throw new InvalidOperationException("Failed to initialize result");
            return new SafeGGmlOptResult(result);
        }

        /// <summary>
        /// Free the result
        /// </summary>
        public void Free()
        {
            if (IsInitialized)
            {
                Native.ggml_opt_result_free(handle);
                handle = IntPtr.Zero;
            }
        }

        /// <summary>
        /// Reset the result
        /// </summary>
        public void Reset()
        {
            ThrowIfNotInitialized();
            Native.ggml_opt_result_reset(handle);
        }

        /// <summary>
        /// Get the number of datapoints
        /// </summary>
        public long GetNData()
        {
            ThrowIfNotInitialized();
            long ndata;
            Native.ggml_opt_result_ndata(handle, &ndata);
            return ndata;
        }

        /// <summary>
        /// Get the loss value and uncertainty
        /// </summary>
        public void GetLoss(out double loss, out double? uncertainty)
        {
            ThrowIfNotInitialized();
            double lossVal, uncVal;
            Native.ggml_opt_result_loss(handle, &lossVal, &uncVal);
            loss = lossVal;
            uncertainty = double.IsNaN(uncVal) ? null : (double?)uncVal;
        }

        /// <summary>
        /// Get the loss value only
        /// </summary>
        public double GetLoss()
        {
            ThrowIfNotInitialized();
            double lossVal, uncVal;
            Native.ggml_opt_result_loss(handle, &lossVal, &uncVal);
            return lossVal;
        }

        /// <summary>
        /// Get the accuracy and uncertainty
        /// </summary>
        public void GetAccuracy(out double accuracy, out double? uncertainty)
        {
            ThrowIfNotInitialized();
            double accVal, uncVal;
            Native.ggml_opt_result_accuracy(handle, &accVal, &uncVal);
            accuracy = accVal;
            uncertainty = double.IsNaN(uncVal) ? null : (double?)uncVal;
        }

        /// <summary>
        /// Get the accuracy value only
        /// </summary>
        public double? GetAccuracy()
        {
            ThrowIfNotInitialized();
            double accVal, uncVal;
            Native.ggml_opt_result_accuracy(handle, &accVal, &uncVal);
            return double.IsNaN(accVal) ? null : (double?)accVal;
        }
    }
}

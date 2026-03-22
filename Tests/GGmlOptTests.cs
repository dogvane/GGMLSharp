using GGMLSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using static GGMLSharp.Structs;

namespace GGMLSharp.Tests
{
    /// <summary>
    /// Port of ggml/tests/test-opt.cpp.
    /// Focuses on deterministic single-backend (CPU) checks in this test harness.
    /// </summary>
    public unsafe class GGmlOptTests : TestBase
    {
        private const long NeDatapoint = 2;
        private const long NeLabel = 1;
        private const long NData = 6;

        private const float SgdLr = 1e-4f;
        private const int SgdEpochs = 900;

        private static readonly GGmlOptGetOptimizerParamsDelegate s_testOptPars = GetTestOptPars;
        private static readonly GGmlOptGetOptimizerParamsDelegate s_regressionOptPars = GetRegressionOptPars;

        private sealed class OptFixture : IDisposable
        {
            public SafeGGmlBackend Backend { get; init; } = null!;
            public SafeGGmlBackendScheduler Scheduler { get; init; } = null!;

            public List<SafeGGmlDataset> DatasetsSupervised { get; init; } = null!;
            public SafeGGmlDataset DatasetUnsupervised { get; init; } = null!;
            public List<SafeGGmlTensor> DataBatch { get; init; } = null!;
            public List<SafeGGmlTensor> LabelsBatch { get; init; } = null!;

            public SafeGGmlContext CtxStatic { get; init; } = null!;
            public SafeGGmlContext CtxCompute { get; init; } = null!;

            public GGmlOptParams OptParams;
            public SafeGGmlOptContext? OptContext { get; init; }
            public SafeGGmlTensor Inputs { get; init; } = null!;
            public SafeGGmlTensor Weights { get; init; } = null!;
            public SafeGGmlTensor Outputs { get; init; } = null!;
            public SafeGGmlBackendBuffer Buffer { get; init; } = null!;
            public SafeGGmlOptResult Result { get; init; } = null!;
            public SafeGGmlOptResult Result2 { get; init; } = null!;

            public GGmlOptGetOptimizerParamsDelegate? Callback { get; init; }

            public void Dispose()
            {
                Result?.Free();
                Result2?.Free();
                OptContext?.Free();

                foreach (var dataset in DatasetsSupervised)
                {
                    dataset?.Free();
                }

                DatasetUnsupervised?.Free();
                Buffer?.Free();
                Scheduler?.Free();
                Backend?.Free();
                CtxStatic?.Free();
                CtxCompute?.Free();
            }
        }

        private static GGmlOptOptimizerParams GetTestOptPars(IntPtr userdata)
        {
            var result = SafeGGmlOptContext.GetDefaultOptimizerParams(userdata);
            result.adamw.alpha = 1.0f;
            result.adamw.beta1 = 0.0f;
            result.adamw.beta2 = 0.0f;
            result.adamw.eps = 0.0f;
            result.adamw.wd = 0.0f;
            result.sgd.alpha = 1.0f;
            result.sgd.wd = 0.0f;
            return result;
        }

        private static GGmlOptOptimizerParams GetRegressionOptPars(IntPtr userdata)
        {
            long epoch = 0;
            if (userdata != IntPtr.Zero)
            {
                epoch = *(long*)userdata;
            }

            var result = SafeGGmlOptContext.GetDefaultOptimizerParams(IntPtr.Zero);
            result.adamw.alpha = 0.1f;
            result.sgd.alpha = (float)(SgdLr * Math.Pow(0.99, 1000.0 * epoch / SgdEpochs));
            result.sgd.wd = 1e-10f;
            return result;
        }

        private static void AssertNear(double expected, double actual, double atol, string message)
        {
            if (Math.Abs(expected - actual) > atol)
            {
                throw new Exception($"{message}: expected={expected}, actual={actual}, atol={atol}");
            }
        }

        private static void AssertTrue(bool cond, string message)
        {
            if (!cond)
            {
                throw new Exception(message);
            }
        }

        private static float ReadScalar(SafeGGmlTensor tensor)
        {
            var data = TestAssertions.ReadFloatData(tensor);
            if (data.Length == 0)
            {
                throw new Exception("Tensor has no data.");
            }
            return data[0];
        }

        private OptFixture CreateFixture(
            GGmlOptOptimizerType optim,
            bool initOptContext = true,
            bool optimizerDefaults = true,
            long nbatchLogical = 1,
            long nbatchPhysical = 1,
            GGmlOptLossType lossType = GGmlOptLossType.GGML_OPT_LOSS_TYPE_SUM)
        {
            var datasets = new List<SafeGGmlDataset>((int)NData);
            for (long ndataShard = 1; ndataShard <= NData; ++ndataShard)
            {
                var dataset = SafeGGmlDataset.Init(
                    GGmlType.GGML_TYPE_F32,
                    GGmlType.GGML_TYPE_F32,
                    NeDatapoint,
                    NeLabel,
                    NData,
                    ndataShard);

                var data = new float[NData * NeDatapoint];
                var labels = new float[NData * NeLabel];

                for (long idata = 0; idata < NData; ++idata)
                {
                    for (long id = 0; id < NeDatapoint; ++id)
                    {
                        data[idata * NeDatapoint + id] = 16 * idata + id;
                    }

                    for (long il = 0; il < NeLabel; ++il)
                    {
                        labels[idata * NeLabel + il] = 16 * (16 * idata + il);
                    }
                }

                dataset.GetData()!.SetData(data);
                dataset.GetLabels()!.SetData(labels);
                datasets.Add(dataset);
            }

            var datasetUnsupervised = SafeGGmlDataset.Init(
                GGmlType.GGML_TYPE_F32,
                GGmlType.GGML_TYPE_F32,
                1,
                0,
                NData,
                1);

            {
                var udata = Enumerable.Range(0, (int)NData).Select(i => (float)i).ToArray();
                datasetUnsupervised.GetData()!.SetData(udata);
            }

            var ctxStatic = new SafeGGmlContext(IntPtr.Zero, Size: 64UL * 1024 * 1024, NoAllocateMemory: true);
            var ctxCompute = new SafeGGmlContext(IntPtr.Zero, Size: 64UL * 1024 * 1024, NoAllocateMemory: true);

            var dataBatch = new List<SafeGGmlTensor>((int)NData);
            var labelsBatch = new List<SafeGGmlTensor>((int)NData);
            for (long ndataBatch = 1; ndataBatch <= NData; ++ndataBatch)
            {
                dataBatch.Add(ctxStatic.NewTensor1d(GGmlType.GGML_TYPE_F32, ndataBatch * NeDatapoint));
                labelsBatch.Add(ctxStatic.NewTensor1d(GGmlType.GGML_TYPE_F32, ndataBatch * NeLabel));
            }

            var inputs = ctxStatic.NewTensor1d(GGmlType.GGML_TYPE_F32, nbatchPhysical);
            inputs.Name = "inputs";

            var weights = ctxStatic.NewTensor1d(GGmlType.GGML_TYPE_F32, 1);
            weights.Name = "weights";
            ctxStatic.SetParam(weights);

            var intermediary = ctxCompute.Add(inputs, weights);
            var outputs = ctxCompute.Scale(intermediary, 1.0f);
            outputs.Name = "outputs";

            var backend = SafeGGmlBackend.CpuInit();
            backend.SetCpuThreads(Math.Max(1, Environment.ProcessorCount / 2));
            var scheduler = SafeGGmlBackendScheduler.Create(new[] { backend });

            var buffer = ctxStatic.BackendAllocContextTensors(backend);
            weights.SetData(new[] { NData / 2.0f });

            if (nbatchLogical % nbatchPhysical != 0)
            {
                throw new Exception("nbatchLogical must be divisible by nbatchPhysical.");
            }

            var optPeriod = checked((int)(nbatchLogical / nbatchPhysical));
            var optParams = SafeGGmlOptContext.DefaultParams(scheduler, lossType);
            optParams.ctx_compute = ctxCompute.NativeHandle;
            optParams.inputs = inputs.NativeHandle;
            optParams.outputs = outputs.NativeHandle;
            optParams.opt_period = optPeriod;
            optParams.optimizer = optim;

            GGmlOptGetOptimizerParamsDelegate? callback = null;
            if (!optimizerDefaults)
            {
                callback = s_testOptPars;
                optParams.get_opt_pars = Marshal.GetFunctionPointerForDelegate(callback);
                optParams.get_opt_pars_ud = IntPtr.Zero;
            }

            var optContext = initOptContext ? SafeGGmlOptContext.Init(optParams) : null;
            if (optContext != null)
            {
                AssertTrue(optContext.GetOptimizerType() == optim, "optimizer type mismatch after init");
            }

            return new OptFixture
            {
                Backend = backend,
                Scheduler = scheduler,
                DatasetsSupervised = datasets,
                DatasetUnsupervised = datasetUnsupervised,
                DataBatch = dataBatch,
                LabelsBatch = labelsBatch,
                CtxStatic = ctxStatic,
                CtxCompute = ctxCompute,
                OptParams = optParams,
                OptContext = optContext,
                Inputs = inputs,
                Weights = weights,
                Outputs = outputs,
                Buffer = buffer,
                Result = SafeGGmlOptResult.Init(),
                Result2 = SafeGGmlOptResult.Init(),
                Callback = callback,
            };
        }

        public void Test_Opt_Dataset()
        {
            foreach (var optim in new[]
            {
                GGmlOptOptimizerType.GGML_OPT_OPTIMIZER_TYPE_ADAMW,
                GGmlOptOptimizerType.GGML_OPT_OPTIMIZER_TYPE_SGD
            })
            {
                foreach (var shuffle in new[] { false, true })
                {
                    using var fx = CreateFixture(optim);

                    for (long ndataShard = 1; ndataShard <= NData; ++ndataShard)
                    {
                        var dataset = fx.DatasetsSupervised[(int)ndataShard - 1];
                        if (shuffle)
                        {
                            fx.OptContext!.Reset(true);
                            dataset.Shuffle(fx.OptContext!, -1);
                        }

                        for (long ndataBatch = 1; ndataBatch <= NData; ++ndataBatch)
                        {
                            if (ndataBatch % ndataShard != 0)
                            {
                                continue;
                            }

                            var dataBatch = fx.DataBatch[(int)ndataBatch - 1];
                            var labelsBatch = fx.LabelsBatch[(int)ndataBatch - 1];
                            var idataShuffled = new List<long>();
                            var nbatches = NData / ndataBatch;

                            for (long ibatch = 0; ibatch < nbatches; ++ibatch)
                            {
                                dataset.GetBatch(dataBatch, labelsBatch, ibatch);
                                var data = TestAssertions.ReadFloatData(dataBatch);
                                var labels = TestAssertions.ReadFloatData(labelsBatch);

                                for (long idataBatch = 0; idataBatch < ndataBatch; ++idataBatch)
                                {
                                    var idata = ibatch * ndataBatch + idataBatch;
                                    var idataFound = (long)(data[idataBatch * NeDatapoint] / 16.0f);

                                    if (!shuffle && idataFound != idata)
                                    {
                                        throw new Exception("dataset order mismatch without shuffle");
                                    }

                                    idataShuffled.Add(idataFound);

                                    for (long id = 0; id < NeDatapoint; ++id)
                                    {
                                        var expected = 16 * idataFound + id;
                                        AssertNear(expected, data[idataBatch * NeDatapoint + id], 1e-6, "dataset data mismatch");
                                    }

                                    for (long il = 0; il < NeLabel; ++il)
                                    {
                                        var expected = 16 * (16 * idataFound + il);
                                        AssertNear(expected, labels[idataBatch * NeLabel + il], 1e-6, "dataset label mismatch");
                                    }
                                }
                            }

                            if (!shuffle || NData % ndataBatch == 0)
                            {
                                var ndataMax = (NData / ndataBatch) * ndataBatch;
                                for (long idata = 0; idata < ndataMax; ++idata)
                                {
                                    var cnt = idataShuffled.Count(x => x == idata);
                                    if (cnt != 1)
                                    {
                                        throw new Exception($"shuffled coverage mismatch, idata={idata}, cnt={cnt}");
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        public void Test_Opt_Grad()
        {
            foreach (var optim in new[]
            {
                GGmlOptOptimizerType.GGML_OPT_OPTIMIZER_TYPE_ADAMW,
                GGmlOptOptimizerType.GGML_OPT_OPTIMIZER_TYPE_SGD
            })
            {
                using var fx = CreateFixture(optim, initOptContext: true, optimizerDefaults: false, nbatchLogical: 999999, nbatchPhysical: 1);
                var gradHistory = Enumerable.Repeat(float.NaN, (int)NData).ToArray();

                for (int idata = 0; idata < NData; ++idata)
                {
                    fx.OptContext!.Alloc(backward: true);
                    fx.Inputs.SetData(new[] { (float)idata });
                    fx.OptContext.Eval(fx.Result);
                    gradHistory[idata] = ReadScalar(fx.OptContext.GetGradAcc(fx.Weights)!);
                }

                for (int idata = 0; idata < NData; ++idata)
                {
                    AssertNear(idata + 1, gradHistory[idata], 1e-6, "grad history mismatch");
                }
            }
        }

        public void Test_Opt_ForwardBackward()
        {
            foreach (var optim in new[]
            {
                GGmlOptOptimizerType.GGML_OPT_OPTIMIZER_TYPE_ADAMW,
                GGmlOptOptimizerType.GGML_OPT_OPTIMIZER_TYPE_SGD
            })
            {
                foreach (var highLevel in new[] { false, true })
                {
                    foreach (var shuffle in new[] { false, true })
                    {
                        if (!highLevel && shuffle)
                        {
                            continue;
                        }

                        using var fx = CreateFixture(optim, initOptContext: true, optimizerDefaults: false);
                        var lossTensor = fx.OptContext!.GetLoss()!;
                        var lossHistory = Enumerable.Repeat(float.NaN, (int)NData).ToArray();

                        {
                            var ndata = fx.Result.GetNData();
                            fx.Result.GetLoss(out var loss, out var lossUnc);
                            fx.Result.GetAccuracy(out var acc, out var accUnc);

                            AssertTrue(ndata == 0, "initial result ndata");
                            AssertNear(0.0, loss, 1e-6, "initial loss");
                            AssertTrue(lossUnc is null, "initial loss_unc should be null");
                            AssertTrue(double.IsNaN(acc), "initial accuracy should be NaN");
                            AssertTrue(accUnc is null, "initial accuracy_unc should be null");
                        }

                        if (highLevel)
                        {
                            if (shuffle)
                            {
                                fx.DatasetUnsupervised.Shuffle(fx.OptContext, -1);
                            }

                            fx.OptContext.Epoch(fx.DatasetUnsupervised, resultTrain: null!, resultEval: fx.Result, idataSplit: 0);
                        }
                        else
                        {
                            for (int idata = 0; idata < NData; ++idata)
                            {
                                fx.OptContext.Alloc(backward: false);
                                fx.Inputs.SetData(new[] { (float)idata });
                                fx.OptContext.Eval(fx.Result);
                                lossHistory[idata] = ReadScalar(lossTensor);
                            }
                        }

                        AssertNear(NData / 2.0, ReadScalar(fx.Weights), 1e-6, "weights after forward");

                        {
                            var ndata = fx.Result.GetNData();
                            fx.Result.GetLoss(out var loss, out var lossUnc);
                            fx.Result.GetAccuracy(out var acc, out var accUnc);
                            AssertTrue(ndata == 6, "forward result ndata");
                            AssertNear(33.0, loss, 1e-6, "forward loss");
                            AssertNear(Math.Sqrt(3.5), lossUnc ?? double.NaN, 1e-6, "forward loss_unc");
                            AssertTrue(double.IsNaN(acc), "forward accuracy should be NaN");
                            AssertTrue(accUnc is null, "forward accuracy_unc should be null");
                        }

                        var w0 = ReadScalar(fx.Weights);
                        for (int i = 0; i < 10; ++i)
                        {
                            fx.OptContext.Alloc(backward: true);
                            fx.OptContext.Eval(fx.Result);
                        }
                        fx.Weights.SetData(new[] { w0 });

                        fx.OptContext.Reset(resetOptimizer: false);
                        fx.Result.Reset();

                        if (highLevel)
                        {
                            if (shuffle)
                            {
                                fx.DatasetUnsupervised.Shuffle(fx.OptContext, -1);
                            }

                            fx.OptContext.Epoch(fx.DatasetUnsupervised, resultTrain: fx.Result, resultEval: null!, idataSplit: NData);
                        }
                        else
                        {
                            for (int idata = 0; idata < NData; ++idata)
                            {
                                fx.OptContext.Alloc(backward: true);
                                fx.Inputs.SetData(new[] { (float)idata });
                                fx.OptContext.Eval(fx.Result);
                            }
                        }

                        AssertNear(-NData * 0.5, ReadScalar(fx.Weights), 1e-5, "weights after forward+backward");

                        {
                            var ndata = fx.Result.GetNData();
                            fx.Result.GetLoss(out var loss, out var lossUnc);
                            fx.Result.GetAccuracy(out var acc, out var accUnc);
                            AssertTrue(ndata == 6, "forward/backward result ndata");
                            AssertNear(18.0, loss, 1e-5, "forward/backward loss");
                            if (!shuffle)
                            {
                                AssertNear(0.0, lossUnc ?? double.NaN, 1e-5, "forward/backward loss_unc");
                            }

                            AssertTrue(double.IsNaN(acc), "forward/backward accuracy should be NaN");
                            AssertTrue(accUnc is null, "forward/backward accuracy_unc should be null");
                        }
                    }
                }
            }
        }

        public void Test_Opt_EpochVsFit()
        {
            foreach (var optim in new[]
            {
                GGmlOptOptimizerType.GGML_OPT_OPTIMIZER_TYPE_ADAMW,
                GGmlOptOptimizerType.GGML_OPT_OPTIMIZER_TYPE_SGD
            })
            {
                float weightsEpoch;
                float weightsFit;

                using (var fx = CreateFixture(optim, initOptContext: true, optimizerDefaults: false))
                {
                    fx.DatasetUnsupervised.Shuffle(fx.OptContext!, -1);
                    fx.OptContext!.Epoch(fx.DatasetUnsupervised, fx.Result, null!, NData);
                    weightsEpoch = ReadScalar(fx.Weights);
                }

                using (var fx = CreateFixture(optim, initOptContext: false, optimizerDefaults: false))
                {
                    SafeGGmlOptContext.Fit(
                        fx.Scheduler,
                        fx.CtxCompute,
                        fx.Inputs,
                        fx.Outputs,
                        fx.DatasetUnsupervised,
                        GGmlOptLossType.GGML_OPT_LOSS_TYPE_SUM,
                        optim,
                        s_testOptPars,
                        epochCount: 1,
                        logicalBatchSize: 1,
                        validationSplit: 0.0f,
                        silent: true);

                    weightsFit = ReadScalar(fx.Weights);
                }

                AssertNear(weightsEpoch, weightsFit, 1e-6, $"epoch_vs_fit mismatch ({optim})");
            }
        }

        public void Test_Opt_IdataSplit()
        {
            const int idataSplit = (int)(NData * 2 / 3);

            foreach (var optim in new[]
            {
                GGmlOptOptimizerType.GGML_OPT_OPTIMIZER_TYPE_ADAMW,
                GGmlOptOptimizerType.GGML_OPT_OPTIMIZER_TYPE_SGD
            })
            {
                var adamw = optim == GGmlOptOptimizerType.GGML_OPT_OPTIMIZER_TYPE_ADAMW;

                foreach (var highLevel in new[] { false, true })
                {
                    using var fx = CreateFixture(optim, initOptContext: true, optimizerDefaults: false);
                    var lossTensor = fx.OptContext!.GetLoss()!;

                    for (int epoch = 1; epoch <= 3; ++epoch)
                    {
                        if (highLevel)
                        {
                            fx.DatasetUnsupervised.Shuffle(fx.OptContext, -1);
                            fx.OptContext.Epoch(fx.DatasetUnsupervised, fx.Result, fx.Result2, idataSplit);
                        }
                        else
                        {
                            int idata = 0;
                            for (; idata < idataSplit; ++idata)
                            {
                                fx.OptContext.Alloc(backward: true);
                                fx.Inputs.SetData(new[] { (float)idata });
                                fx.OptContext.Eval(fx.Result);
                                _ = ReadScalar(lossTensor);
                            }

                            for (; idata < NData; ++idata)
                            {
                                fx.OptContext.Alloc(backward: false);
                                fx.Inputs.SetData(new[] { (float)idata });
                                fx.OptContext.Eval(fx.Result2);
                                _ = ReadScalar(lossTensor);
                            }
                        }

                        if (adamw)
                        {
                            var caseInfo = $"idata_split ({optim}, highLevel={highLevel}, epoch={epoch})";

                            AssertNear(NData / 2.0 - epoch * idataSplit, ReadScalar(fx.Weights), 1e-6, $"{caseInfo} weights");

                            var ndataBack = fx.Result.GetNData();
                            fx.Result.GetLoss(out var lossBack, out var lossBackUnc);
                            fx.Result.GetAccuracy(out var accBack, out var accBackUnc);
                            AssertTrue(ndataBack == idataSplit, $"{caseInfo} backward ndata");
                            if (highLevel)
                            {
                                AssertTrue(!double.IsNaN(lossBack) && !double.IsInfinity(lossBack), $"{caseInfo} backward loss finite");
                                AssertTrue(lossBackUnc.HasValue && !double.IsNaN(lossBackUnc.Value) && !double.IsInfinity(lossBackUnc.Value), $"{caseInfo} backward loss_unc finite");
                            }
                            else
                            {
                                AssertNear(28.0 - epoch * 16.0, lossBack, 1e-6, $"{caseInfo} backward loss");
                                AssertNear(0.0, lossBackUnc ?? double.NaN, 1e-6, $"{caseInfo} backward loss_unc");
                            }
                            AssertTrue(double.IsNaN(accBack), $"{caseInfo} backward accuracy");
                            AssertTrue(accBackUnc is null, $"{caseInfo} backward accuracy_unc");

                            var ndataFwd = fx.Result2.GetNData();
                            fx.Result2.GetLoss(out var lossFwd, out var lossFwdUnc);
                            fx.Result2.GetAccuracy(out var accFwd, out var accFwdUnc);
                            AssertTrue(ndataFwd == NData - idataSplit, $"{caseInfo} forward ndata");
                            if (highLevel)
                            {
                                AssertTrue(!double.IsNaN(lossFwd) && !double.IsInfinity(lossFwd), $"{caseInfo} forward loss finite");
                                AssertTrue(lossFwdUnc.HasValue && !double.IsNaN(lossFwdUnc.Value) && !double.IsInfinity(lossFwdUnc.Value), $"{caseInfo} forward loss_unc finite");
                            }
                            else
                            {
                                AssertNear(15.0 - epoch * 8.0, lossFwd, 1e-6, $"{caseInfo} forward loss");
                                AssertNear(Math.Sqrt(0.5), lossFwdUnc ?? double.NaN, 1e-6, $"{caseInfo} forward loss_unc");
                            }
                            AssertTrue(double.IsNaN(accFwd), $"{caseInfo} forward accuracy");
                            AssertTrue(accFwdUnc is null, $"{caseInfo} forward accuracy_unc");
                        }

                        fx.Result.Reset();
                        fx.Result2.Reset();
                    }
                }
            }
        }

        public void Test_Opt_GradientAccumulation()
        {
            var optim = GGmlOptOptimizerType.GGML_OPT_OPTIMIZER_TYPE_ADAMW;

            foreach (var nbatchPhysical in new[] { 2, 1 })
            {
                foreach (var lossType in new[]
                {
                    GGmlOptLossType.GGML_OPT_LOSS_TYPE_SUM,
                    GGmlOptLossType.GGML_OPT_LOSS_TYPE_MEAN
                })
                {
                    using var fx = CreateFixture(
                        optim,
                        initOptContext: true,
                        optimizerDefaults: false,
                        nbatchLogical: 6,
                        nbatchPhysical: nbatchPhysical,
                        lossType: lossType);

                    var gradHistory = Enumerable.Repeat(float.NaN, (int)NData).ToArray();

                    for (int epoch = 1; epoch <= 4; ++epoch)
                    {
                        if (nbatchPhysical == 1)
                        {
                            for (int idata = 0; idata < NData; ++idata)
                            {
                                fx.OptContext!.Alloc(backward: true);
                                fx.Inputs.SetData(new[] { (float)idata });
                                fx.OptContext.Eval(fx.Result);
                                gradHistory[idata] = ReadScalar(fx.OptContext.GetGradAcc(fx.Weights)!);
                            }
                        }
                        else
                        {
                            for (int idata = 0; idata < NData; idata += 2)
                            {
                                fx.OptContext!.Alloc(backward: true);
                                fx.Inputs.SetData(new[] { (float)idata, (float)(idata + 1) });
                                fx.OptContext.Eval(fx.Result);
                                gradHistory[idata] = 0.0f;
                                gradHistory[idata + 1] = ReadScalar(fx.OptContext.GetGradAcc(fx.Weights)!);
                            }
                        }

                        if (lossType == GGmlOptLossType.GGML_OPT_LOSS_TYPE_SUM)
                        {
                            if (nbatchPhysical == 1)
                            {
                                AssertNear(1.0, gradHistory[0], 1e-6, "grad[0]");
                                AssertNear(3.0, gradHistory[2], 1e-6, "grad[2]");
                                AssertNear(5.0, gradHistory[4], 1e-6, "grad[4]");
                            }
                            else
                            {
                                AssertNear(0.0, gradHistory[0], 1e-6, "grad[0]");
                                AssertNear(0.0, gradHistory[2], 1e-6, "grad[2]");
                                AssertNear(0.0, gradHistory[4], 1e-6, "grad[4]");
                            }
                            AssertNear(2.0, gradHistory[1], 1e-6, "grad[1]");
                            AssertNear(4.0, gradHistory[3], 1e-6, "grad[3]");
                            AssertNear(6.0, gradHistory[5], 1e-6, "grad[5]");
                        }
                        else
                        {
                            if (nbatchPhysical == 1)
                            {
                                AssertNear(1.0 / NData, gradHistory[0], 1e-6, "grad[0]");
                                AssertNear(3.0 / NData, gradHistory[2], 1e-6, "grad[2]");
                                AssertNear(5.0 / NData, gradHistory[4], 1e-6, "grad[4]");
                            }
                            else
                            {
                                AssertNear(0.0, gradHistory[0], 1e-6, "grad[0]");
                                AssertNear(0.0, gradHistory[2], 1e-6, "grad[2]");
                                AssertNear(0.0, gradHistory[4], 1e-6, "grad[4]");
                            }
                            AssertNear(2.0 / NData, gradHistory[1], 1e-6, "grad[1]");
                            AssertNear(4.0 / NData, gradHistory[3], 1e-6, "grad[3]");
                            AssertNear(6.0 / NData, gradHistory[5], 1e-6, "grad[5]");
                        }

                        AssertNear((NData / 2.0) - epoch, ReadScalar(fx.Weights), 1e-6, "gradient_accum weights");

                        var ndataResult = fx.Result.GetNData();
                        fx.Result.GetLoss(out var loss, out _);
                        fx.Result.GetAccuracy(out var acc, out var accUnc);
                        AssertTrue(ndataResult == NData / nbatchPhysical, "gradient_accum ndata");

                        if (lossType == GGmlOptLossType.GGML_OPT_LOSS_TYPE_SUM)
                        {
                            AssertNear(39.0 - epoch * 6.0, loss, 1e-6, "gradient_accum loss(sum)");
                        }
                        else
                        {
                            AssertNear((39.0 - epoch * 6.0) / NData, loss, 1e-6, "gradient_accum loss(mean)");
                        }

                        AssertTrue(double.IsNaN(acc), "gradient_accum accuracy");
                        AssertTrue(accUnc is null, "gradient_accum accuracy_unc");
                        fx.Result.Reset();
                    }
                }
            }
        }

        public void Test_Opt_Regression()
        {
            foreach (var optim in new[]
            {
                GGmlOptOptimizerType.GGML_OPT_OPTIMIZER_TYPE_ADAMW,
                GGmlOptOptimizerType.GGML_OPT_OPTIMIZER_TYPE_SGD
            })
            {
                const long ndataRegression = 201;
                const float aTrue = 1.2f;
                const float bTrue = 3.4f;

                var dataset = SafeGGmlDataset.Init(
                    GGmlType.GGML_TYPE_F32,
                    GGmlType.GGML_TYPE_F32,
                    neDatapoint: 1,
                    neLabel: 1,
                    ndata: ndataRegression,
                    ndataShard: ndataRegression);

                var rng = new Random(12345);
                const float xMin = -100.0f;
                const float xMax = 100.0f;
                var data = new float[ndataRegression];
                var labels = new float[ndataRegression];

                for (int i = 0; i < ndataRegression; ++i)
                {
                    var x = xMin + (xMax - xMin) * i / (ndataRegression - 1);

                    var u1 = 1.0 - rng.NextDouble();
                    var u2 = 1.0 - rng.NextDouble();
                    var z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                    var y = aTrue * x + bTrue + 0.1f * (float)z;

                    data[i] = x;
                    labels[i] = y;
                }

                dataset.GetData()!.SetData(data);
                dataset.GetLabels()!.SetData(labels);

                var ctxStatic = new SafeGGmlContext(IntPtr.Zero, 64UL * 1024 * 1024, NoAllocateMemory: true);
                var ctxCompute = new SafeGGmlContext(IntPtr.Zero, 64UL * 1024 * 1024, NoAllocateMemory: true);
                var backend = SafeGGmlBackend.CpuInit();
                var sched = SafeGGmlBackendScheduler.Create(new[] { backend });

                var xTensor = ctxStatic.NewTensor2d(GGmlType.GGML_TYPE_F32, 1, ndataRegression);
                xTensor.Name = "x";

                var a = ctxStatic.NewTensor1d(GGmlType.GGML_TYPE_F32, 1);
                a.Name = "a";
                ctxStatic.SetParam(a);

                var b = ctxStatic.NewTensor1d(GGmlType.GGML_TYPE_F32, 1);
                b.Name = "b";
                ctxStatic.SetParam(b);

                var f = ctxCompute.Add(ctxCompute.Mul(xTensor, a), b);
                f.Name = "f";

                var buf = ctxStatic.BackendAllocContextTensors(backend);
                a.SetData(new[] { 1.0f });
                b.SetData(new[] { 3.0f });

                var adamw = optim == GGmlOptOptimizerType.GGML_OPT_OPTIMIZER_TYPE_ADAMW;
                var nEpoch = adamw ? 100 : SgdEpochs;

                SafeGGmlOptContext.Fit(
                    sched,
                    ctxCompute,
                    xTensor,
                    f,
                    dataset,
                    GGmlOptLossType.GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR,
                    optim,
                    s_regressionOptPars,
                    epochCount: nEpoch,
                    logicalBatchSize: ndataRegression,
                    validationSplit: 0.0f,
                    silent: true);

                var aFit = ReadScalar(a);
                var bFit = ReadScalar(b);

                if (adamw)
                {
                    AssertNear(aTrue, aFit, 1e-2, "regression a");
                    AssertNear(bTrue, bFit, 1e-2, "regression b");
                }
                else
                {
                    AssertTrue(!float.IsNaN(aFit) && !float.IsInfinity(aFit), "sgd regression a invalid");
                    AssertTrue(!float.IsNaN(bFit) && !float.IsInfinity(bFit), "sgd regression b invalid");
                }

                buf.Free();
                sched.Free();
                backend.Free();
                ctxStatic.Free();
                ctxCompute.Free();
                dataset.Free();
            }
        }
    }
}

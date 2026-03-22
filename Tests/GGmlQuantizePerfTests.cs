using GGMLSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace GGMLSharp.Tests
{
    public sealed class GGmlQuantizePerfTests
    {
        private const int Qk = 32;
        private const int L1Size = 32 * 128;

        public void Test_QuantizePerf_DefaultSmoke_AllSupportedTypes()
        {
            var results = RunBenchmarks(new QuantizePerfParams
            {
                Iterations = 1,
            });

            TestAssertions.Equal(results.Count > 0 ? 1 : 0, 1, "benchmark result count");

            foreach (var result in results)
            {
                if (!float.IsFinite(result.SampleValue))
                {
                    throw new Exception($"{result.TypeName}/{result.Operation}: sample value is not finite");
                }

                if (result.MinMicroseconds < 0 || result.AverageMicroseconds < 0)
                {
                    throw new Exception($"{result.TypeName}/{result.Operation}: timing must be non-negative");
                }

                if (result.QuantizedSize <= 0)
                {
                    throw new Exception($"{result.TypeName}/{result.Operation}: quantized size must be positive");
                }
            }
        }

        public void Test_QuantizePerf_Filtering_Honors_Type_And_Operation()
        {
            var parameters = new QuantizePerfParams
            {
                OpQuantizeRowQ = true,
                Iterations = 1,
            };
            parameters.IncludeTypes.Add("q4_0");
            parameters.TestSizes.Add(L1Size);

            var results = RunBenchmarks(parameters);

            TestAssertions.Equal(1, results.Count, "filtered benchmark result count");

            var result = results[0];
            if (!string.Equals(result.TypeName, "q4_0", StringComparison.Ordinal))
            {
                throw new Exception($"expected filtered type q4_0, got {result.TypeName}");
            }

            if (!string.Equals(result.Operation, "quantize_row_q", StringComparison.Ordinal))
            {
                throw new Exception($"expected filtered operation quantize_row_q, got {result.Operation}");
            }
        }

        private static List<BenchmarkResult> RunBenchmarks(QuantizePerfParams parameters)
        {
            GGmlQuantization.InitializeCpu();

            var testSizes = parameters.TestSizes.Count > 0
                ? parameters.TestSizes.OrderBy(size => size).ToArray()
                : new[] { L1Size };

            bool runAllOps = !(parameters.OpQuantizeRowQReference ||
                               parameters.OpQuantizeRowQ ||
                               parameters.OpDequantizeRowQ ||
                               parameters.OpQuantizeRowQDot ||
                               parameters.OpVecDotQ);

            var results = new List<BenchmarkResult>();
            int largest = testSizes[^1];

            float[] testData1 = GenerateData(0.0f, largest);
            float[] testData2 = GenerateData(1.0f, largest);

            try
            {
                for (int i = 0; i < (int)Structs.GGmlType.GGML_TYPE_COUNT; i++)
                {
                    var type = (Structs.GGmlType)i;
                    var traits = GGmlQuantization.GetTypeTraits(type);
                    var cpuTraits = GGmlQuantization.GetCpuTypeTraits(type);

                    if (!cpuTraits.HasFromFloat || !traits.HasToFloat)
                    {
                        continue;
                    }

                    string typeName = traits.TypeName ?? type.ToString();
                    if (parameters.IncludeTypes.Count > 0 &&
                        !parameters.IncludeTypes.Contains(typeName, StringComparer.OrdinalIgnoreCase))
                    {
                        continue;
                    }

                    GGmlQuantization.Initialize(type);

                    if (runAllOps || parameters.OpQuantizeRowQReference)
                    {
                        foreach (int size in testSizes)
                        {
                            int quantizedSize = GGmlQuantization.RowSize(type, size);
                            BenchmarkStats stats = Benchmark(parameters.Iterations, () =>
                            {
                                byte[] output = GGmlQuantization.QuantizeReference(type, Slice(testData1, size));
                                return output[0];
                            });

                            results.Add(new BenchmarkResult(typeName, "quantize_row_q_reference", size, quantizedSize, stats));
                        }
                    }

                    if (runAllOps || parameters.OpQuantizeRowQ)
                    {
                        foreach (int size in testSizes)
                        {
                            int quantizedSize = GGmlQuantization.RowSize(type, size);
                            BenchmarkStats stats = Benchmark(parameters.Iterations, () =>
                            {
                                byte[] output = GGmlQuantization.Quantize(type, Slice(testData1, size));
                                return output[0];
                            });

                            results.Add(new BenchmarkResult(typeName, "quantize_row_q", size, quantizedSize, stats));
                        }
                    }

                    if (runAllOps || parameters.OpDequantizeRowQ)
                    {
                        byte[] quantized = GGmlQuantization.Quantize(type, testData1);

                        foreach (int size in testSizes)
                        {
                            int quantizedSize = GGmlQuantization.RowSize(type, size);
                            byte[] slice = SliceBytes(quantized, quantizedSize);
                            BenchmarkStats stats = Benchmark(parameters.Iterations, () =>
                            {
                                float[] output = GGmlQuantization.Dequantize(type, slice, size);
                                return output[0];
                            });

                            results.Add(new BenchmarkResult(typeName, "dequantize_row_q", size, quantizedSize, stats));
                        }
                    }

                    if (runAllOps || parameters.OpQuantizeRowQDot)
                    {
                        foreach (int size in testSizes)
                        {
                            int quantizedSize = GGmlQuantization.RowSize(type, size);
                            BenchmarkStats stats = Benchmark(parameters.Iterations, () =>
                            {
                                byte[] output = GGmlQuantization.QuantizeForVecDot(type, Slice(testData1, size));
                                return output[0];
                            });

                            results.Add(new BenchmarkResult(typeName, "quantize_row_q_dot", size, quantizedSize, stats));
                        }
                    }

                    if ((runAllOps || parameters.OpVecDotQ) && cpuTraits.HasVecDot)
                    {
                        byte[] lhs = GGmlQuantization.Quantize(type, testData1);
                        byte[] rhs = GGmlQuantization.QuantizeForVecDot(type, testData2);

                        foreach (int size in testSizes)
                        {
                            int quantizedSize = GGmlQuantization.RowSize(type, size);
                            byte[] lhsSlice = SliceBytes(lhs, quantizedSize);
                            byte[] rhsSlice = SliceBytes(rhs, GGmlQuantization.RowSize(cpuTraits.VecDotType, size));
                            BenchmarkStats stats = Benchmark(parameters.Iterations, () =>
                            {
                                return GGmlQuantization.VecDot(type, size, lhsSlice, rhsSlice);
                            });

                            results.Add(new BenchmarkResult(typeName, "vec_dot_q", size, quantizedSize, stats));
                        }
                    }
                }
            }
            finally
            {
                GGmlQuantization.Free();
            }

            return results;
        }

        private static BenchmarkStats Benchmark(int iterations, Func<float> action)
        {
            double totalMicroseconds = 0.0;
            double minMicroseconds = double.MaxValue;
            float sample = 0.0f;

            for (int i = 0; i < iterations; i++)
            {
                long start = Stopwatch.GetTimestamp();
                sample = action();
                long end = Stopwatch.GetTimestamp();
                double elapsed = (end - start) * 1_000_000.0 / Stopwatch.Frequency;

                totalMicroseconds += elapsed;
                minMicroseconds = Math.Min(minMicroseconds, elapsed);
            }

            return new BenchmarkStats((float)minMicroseconds, (float)(totalMicroseconds / iterations), sample);
        }

        private static float[] GenerateData(float offset, int count)
        {
            var data = new float[count];
            for (int i = 0; i < count; i++)
            {
                data[i] = 0.1f + 2.0f * MathF.Cos(i + offset);
            }

            return data;
        }

        private static float[] Slice(float[] source, int size)
        {
            if (source.Length == size)
            {
                return source;
            }

            var slice = new float[size];
            Array.Copy(source, slice, size);
            return slice;
        }

        private static byte[] SliceBytes(byte[] source, int size)
        {
            if (source.Length == size)
            {
                return source;
            }

            var slice = new byte[size];
            Array.Copy(source, slice, size);
            return slice;
        }

        private sealed class QuantizePerfParams
        {
            public List<string> IncludeTypes { get; } = new();
            public List<int> TestSizes { get; } = new();
            public bool OpQuantizeRowQReference { get; init; }
            public bool OpQuantizeRowQ { get; init; }
            public bool OpDequantizeRowQ { get; init; }
            public bool OpQuantizeRowQDot { get; init; }
            public bool OpVecDotQ { get; init; }
            public int Iterations { get; init; } = 1;
        }

        private readonly record struct BenchmarkStats(float MinMicroseconds, float AverageMicroseconds, float SampleValue);

        private readonly record struct BenchmarkResult(
            string TypeName,
            string Operation,
            int Size,
            int QuantizedSize,
            BenchmarkStats Stats)
        {
            public float MinMicroseconds => Stats.MinMicroseconds;
            public float AverageMicroseconds => Stats.AverageMicroseconds;
            public float SampleValue => Stats.SampleValue;
            public float MinCyclesPer32Values => Qk * MinMicroseconds / Size;
        }
    }
}

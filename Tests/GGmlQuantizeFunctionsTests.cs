using GGMLSharp;
using System;
using System.Collections.Generic;

namespace GGMLSharp.Tests
{
    public sealed class GGmlQuantizeFunctionsTests
    {
        private const float MaxQuantizationReferenceError = 0.0001f;
        private const float MaxQuantizationTotalError = 0.002f;
        private const float MaxQuantizationTotalErrorTernary = 0.01f;
        private const float MaxQuantizationTotalError2Bits = 0.0075f;
        private const float MaxQuantizationTotalError3Bits = 0.0040f;
        private const float MaxQuantizationTotalError3BitsXxs = 0.0050f;
        private const float MaxDotProductError = 0.02f;
        private const float MaxDotProductErrorLowBit = 0.04f;
        private const float MaxDotProductErrorTernary = 0.15f;
        private const int TestSize = 32 * 128;

        public void Test_QuantizeFns_Match_UpstreamThresholds()
        {
            GGmlQuantization.InitializeCpu();

            var testData1 = GenerateData(0.0f, TestSize);
            var testData2 = GenerateData(1.0f, TestSize);
            var failures = new List<string>();

            try
            {
                for (int i = 0; i < (int)Structs.GGmlType.GGML_TYPE_COUNT; i++)
                {
                    var type = (Structs.GGmlType)i;
                    var traits = GGmlQuantization.GetTypeTraits(type);
                    var cpuTraits = GGmlQuantization.GetCpuTypeTraits(type);

                    if (traits.BlockSize == 0)
                    {
                        continue;
                    }
                    
                    Console.WriteLine($"Testing {traits.TypeName ?? type.ToString()}...");

                    GGmlQuantization.Initialize(type);
                    string typeName = traits.TypeName ?? type.ToString();

                    if (cpuTraits.HasFromFloat && traits.HasToFloat)
                    {
                        float totalError = TotalQuantizationError(type, testData1);
                        float maxQuantizationError = GetMaxQuantizationError(type);
                        if (!(totalError < maxQuantizationError))
                        {
                            failures.Add($"{typeName}: total quantization error {totalError} >= {maxQuantizationError}");
                        }

                        if (traits.HasFromFloatReference)
                        {
                            float referenceError = ReferenceQuantizationError(type, testData1);
                            if (!(referenceError < MaxQuantizationReferenceError))
                            {
                                failures.Add($"{typeName}: reference quantization error {referenceError} >= {MaxQuantizationReferenceError}");
                            }
                        }

                        if (cpuTraits.HasVecDot)
                        {
                            float dotError = DotProductError(type, testData1, testData2);
                            float maxDotError = GetMaxDotProductError(type);
                            if (!(dotError < maxDotError))
                            {
                                failures.Add($"{typeName}: dot product error {dotError} >= {maxDotError}");
                            }
                        }
                    }
                }
            }
            finally
            {
                GGmlQuantization.Free();
            }

            if (failures.Count > 0)
            {
                throw new Exception(string.Join(Environment.NewLine, failures));
            }
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

        private static float ArrayRmse(float[] lhs, float[] rhs)
        {
            double sum = 0.0;
            for (int i = 0; i < lhs.Length; i++)
            {
                double diff = lhs[i] - rhs[i];
                sum += diff * diff;
            }

            return (float)(Math.Sqrt(sum) / lhs.Length);
        }

        private static float TotalQuantizationError(Structs.GGmlType type, float[] testData)
        {
            byte[] quantized = GGmlQuantization.Quantize(type, testData);
            float[] dequantized = GGmlQuantization.Dequantize(type, quantized, testData.Length);
            return ArrayRmse(testData, dequantized);
        }

        private static float ReferenceQuantizationError(Structs.GGmlType type, float[] testData)
        {
            byte[] quantized = GGmlQuantization.Quantize(type, testData);
            float[] dequantized = GGmlQuantization.Dequantize(type, quantized, testData.Length);

            byte[] quantizedReference = GGmlQuantization.QuantizeReference(type, testData);
            float[] dequantizedReference = GGmlQuantization.Dequantize(type, quantizedReference, testData.Length);

            return ArrayRmse(dequantized, dequantizedReference);
        }

        private static float DotProductError(Structs.GGmlType type, float[] lhs, float[] rhs)
        {
            byte[] lhsQuantized = GGmlQuantization.Quantize(type, lhs);
            byte[] rhsQuantized = GGmlQuantization.QuantizeForVecDot(type, rhs);

            float result = GGmlQuantization.VecDot(type, lhs.Length, lhsQuantized, rhsQuantized);
            float reference = DotProduct(lhs, rhs);

            return MathF.Abs(result - reference) / lhs.Length;
        }

        private static float DotProduct(float[] lhs, float[] rhs)
        {
            double sum = 0.0;
            for (int i = 0; i < lhs.Length; i++)
            {
                sum += lhs[i] * rhs[i];
            }

            return (float)sum;
        }

        private static float GetMaxQuantizationError(Structs.GGmlType type)
        {
            return type switch
            {
                Structs.GGmlType.GGML_TYPE_TQ1_0 => MaxQuantizationTotalErrorTernary,
                Structs.GGmlType.GGML_TYPE_TQ2_0 => MaxQuantizationTotalErrorTernary,
                Structs.GGmlType.GGML_TYPE_Q2_K => MaxQuantizationTotalError2Bits,
                Structs.GGmlType.GGML_TYPE_IQ2_S => MaxQuantizationTotalError2Bits,
                Structs.GGmlType.GGML_TYPE_Q3_K => MaxQuantizationTotalError3Bits,
                Structs.GGmlType.GGML_TYPE_IQ3_S => MaxQuantizationTotalError3Bits,
                Structs.GGmlType.GGML_TYPE_IQ3_XXS => MaxQuantizationTotalError3BitsXxs,
                _ => MaxQuantizationTotalError,
            };
        }

        private static float GetMaxDotProductError(Structs.GGmlType type)
        {
            return type switch
            {
                Structs.GGmlType.GGML_TYPE_Q2_K => MaxDotProductErrorLowBit,
                Structs.GGmlType.GGML_TYPE_IQ2_XS => MaxDotProductErrorLowBit,
                Structs.GGmlType.GGML_TYPE_IQ2_XXS => MaxDotProductErrorLowBit,
                Structs.GGmlType.GGML_TYPE_IQ3_XXS => MaxDotProductErrorLowBit,
                Structs.GGmlType.GGML_TYPE_IQ3_S => MaxDotProductErrorLowBit,
                Structs.GGmlType.GGML_TYPE_IQ2_S => MaxDotProductErrorLowBit,
                Structs.GGmlType.GGML_TYPE_TQ1_0 => MaxDotProductErrorTernary,
                Structs.GGmlType.GGML_TYPE_TQ2_0 => MaxDotProductErrorTernary,
                _ => MaxDotProductError,
            };
        }
    }
}

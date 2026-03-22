using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using GGMLSharp;
using GGMLSharp.Tests;

CrtNativeMethods.DisableAssertPopup();
CrtNativeMethods.DisableAbortDialog();
CrtNativeMethods.DisableWindowsErrorDialog();
Environment.SetEnvironmentVariable("_NO_DEBUG_HEAP", "1");

var testFramework = new TestFramework();
testFramework.RunAllTests();
return testFramework.FailureCount == 0 ? 0 : 1;

public sealed class TestFramework
{
    private readonly List<TestSuite> _testSuites = new();
    private readonly BackendType _availableBackend;
    private readonly string? _testFilter;

    private int _successCount;
    private int _failureCount;
    private int _skippedCount;

    public int FailureCount => _failureCount;

    public TestFramework()
    {
        _availableBackend = DetectAvailableBackend();
        _testFilter = Environment.GetEnvironmentVariable("GGML_TEST_FILTER")?.Trim();
        RegisterTestSuites();
    }

    private BackendType DetectAvailableBackend()
    {
        Console.WriteLine("=== Backend Detection ===");

        try
        {
            using var cudaBackend = SafeGGmlBackend.CudaInit();
            if (cudaBackend != null && !cudaBackend.IsInvalid)
            {
                Console.WriteLine("CUDA backend available");
                return BackendType.CUDA;
            }
        }
        catch
        {
            // Fall back to CPU below.
        }

        Console.WriteLine("CPU backend available");
        return BackendType.CPU;
    }

    private void RegisterTestSuites()
    {
        _testSuites.Add(new TestSuite("GGML Arange Tests", new GGmlArangeTests()));
        _testSuites.Add(new TestSuite("GGML Backend Ops Tests", new GGmlBackendOpsTests()));
        _testSuites.Add(new TestSuite("GGML Cont Tests", new GGmlContTests()));
        _testSuites.Add(new TestSuite("GGML Conv Transpose 1D Tests", new GGmlConvTranspose1DTests()));
        _testSuites.Add(new TestSuite("GGML Conv Transpose Tests", new GGmlConvTransposeTests()));
        _testSuites.Add(new TestSuite("GGML Conv1D DW C1 Tests", new GGmlConv1dDepthwiseC1Tests()));
        _testSuites.Add(new TestSuite("GGML Conv1D DW C2 Tests", new GGmlConv1dDepthwiseC2Tests()));
        _testSuites.Add(new TestSuite("GGML Conv1D Tests", new GGmlConv1dTests()));
        _testSuites.Add(new TestSuite("GGML Conv2D Tests", new GGmlConv2dTests()));
        _testSuites.Add(new TestSuite("GGML Conv2D DW Tests", new GGmlConv2dDepthwiseTests()));
        _testSuites.Add(new TestSuite("GGML Custom Op Tests", new GGmlCustomOpTests()));
        _testSuites.Add(new TestSuite("GGML Dup Tests", new GGmlDupTests()));
        _testSuites.Add(new TestSuite("GGML Interpolate Tests", new GGmlInterpolateTests()));
        _testSuites.Add(new TestSuite("GGML Pad Reflect 1D Tests", new GGmlPadReflect1dTests()));
        _testSuites.Add(new TestSuite("GGML Pool Tests", new GGmlPoolTests()));
        _testSuites.Add(new TestSuite("GGML Quantize Functions Tests", new GGmlQuantizeFunctionsTests()));
        _testSuites.Add(new TestSuite("GGML Quantize Perf Tests", new GGmlQuantizePerfTests()));
        _testSuites.Add(new TestSuite("GGML Rel Pos Tests", new GGmlRelPosTests()));
        _testSuites.Add(new TestSuite("GGML Roll Tests", new GGmlRollTests()));
        _testSuites.Add(new TestSuite("GGML Timestep Embedding Tests", new GGmlTimestepEmbeddingTests()));
        _testSuites.Add(new TestSuite("GGML Optimizer Tests", new GGmlOptTests()));
    }

    public void RunAllTests()
    {
        Console.WriteLine($"=== GGMLSharp Test Run (backend: {_availableBackend}) ===");
        if (!string.IsNullOrWhiteSpace(_testFilter))
        {
            Console.WriteLine($"=== Filter: {_testFilter} ===");
        }
        Console.WriteLine();

        foreach (var suite in _testSuites)
        {
            RunTestSuite(suite);
        }

        PrintSummary();
        Cleanup();
    }

    private void RunTestSuite(TestSuite suite)
    {
        var tests = suite.GetTests();
        if (!string.IsNullOrWhiteSpace(_testFilter))
        {
            var suiteMatches = suite.Name.Contains(_testFilter, StringComparison.OrdinalIgnoreCase);
            tests = suiteMatches
                ? tests
                : tests
                    .Where(test => test.Name.Contains(_testFilter, StringComparison.OrdinalIgnoreCase))
                    .ToList();
        }

        tests = FilterTestsByBackend(tests);
        if (tests.Count == 0)
        {
            return;
        }

        Console.WriteLine($"--- {suite.Name} ---");
        Console.WriteLine();

        foreach (var test in tests)
        {
            RunSingleTest(test);
        }

        Console.WriteLine();
    }

    private List<TestCase> FilterTestsByBackend(List<TestCase> tests)
    {
        var filtered = new List<TestCase>();

        foreach (var test in tests)
        {
            var backendAttr = test.Method?.GetCustomAttribute<BackendRequirementAttribute>();
            if (backendAttr == null ||
                backendAttr.RequiredBackend == BackendType.Both ||
                backendAttr.RequiredBackend == _availableBackend)
            {
                filtered.Add(test);
                continue;
            }

            Console.WriteLine($"SKIP {test.Name} (requires {backendAttr.RequiredBackend})");
            _skippedCount++;
        }

        return filtered;
    }

    private void RunSingleTest(TestCase test)
    {
        Console.Write($"RUN  {test.Name} ... ");

        try
        {
            test.Action();
            _successCount++;
            Console.WriteLine("PASS");
        }
        catch (Exception ex)
        {
            _failureCount++;
            Console.WriteLine("FAIL");
            PrintTestError(test.Name, ex);
        }
    }

    private static void PrintTestError(string testName, Exception ex)
    {
        Console.WriteLine($"  {testName}: {ex.Message}");

        if (ex.InnerException != null)
        {
            Console.WriteLine($"  inner: {ex.InnerException.Message}");
        }

        var firstStackLine = ex.StackTrace?
            .Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries)
            .FirstOrDefault();

        if (!string.IsNullOrWhiteSpace(firstStackLine))
        {
            Console.WriteLine($"  at: {firstStackLine.Trim()}");
        }
    }

    private void PrintSummary()
    {
        Console.WriteLine("=== Summary ===");
        Console.WriteLine($"Passed : {_successCount}");
        Console.WriteLine($"Failed : {_failureCount}");
        Console.WriteLine($"Skipped: {_skippedCount}");
        Console.WriteLine($"Total  : {_successCount + _failureCount + _skippedCount}");
    }

    private void Cleanup()
    {
        foreach (var suite in _testSuites)
        {
            suite.Dispose();
        }
    }
}

public sealed class TestSuite : IDisposable
{
    public string Name { get; }

    private readonly Type _testType;

    public TestSuite(string name, object testInstance)
    {
        Name = name;
        _testType = testInstance.GetType();
        if (testInstance is IDisposable disposable)
        {
            disposable.Dispose();
        }
    }

    public List<TestCase> GetTests()
    {
        return _testType
            .GetMethods(BindingFlags.Instance | BindingFlags.Public)
            .Where(method => method.Name.StartsWith("Test_", StringComparison.Ordinal) && !method.IsStatic)
            .Select(method => new TestCase(
                method.Name,
                () =>
                {
                    object? testInstance = null;
                    try
                    {
                        testInstance = Activator.CreateInstance(_testType)
                            ?? throw new InvalidOperationException($"Unable to create test instance for {_testType.FullName}");
                        method.Invoke(testInstance, null);
                    }
                    catch (TargetInvocationException tie) when (tie.InnerException != null)
                    {
                        throw tie.InnerException;
                    }
                    finally
                    {
                        if (testInstance is IDisposable disposable)
                        {
                            disposable.Dispose();
                        }
                    }
                },
                method))
            .OrderBy(test => test.Name, StringComparer.Ordinal)
            .ToList();
    }

    public void Dispose()
    {
        // Each test case gets its own instance and disposes it immediately after execution.
    }
}

public sealed class TestCase
{
    public string Name { get; }
    public Action Action { get; }
    public MethodInfo? Method { get; }

    public TestCase(string name, Action action, MethodInfo? method)
    {
        Name = name;
        Action = action;
        Method = method;
    }
}

public static class CrtNativeMethods
{
    [DllImport("ucrtbased.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi, ExactSpelling = true)]
    public static extern int _CrtSetReportMode(int reportType, int reportMode);

    [DllImport("ucrtbased.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi, ExactSpelling = true)]
    public static extern int _CrtSetReportFile(int reportType, IntPtr reportFile);

    [DllImport("msvcr140d.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern int _CrtSetReportMode140(int reportType, int reportMode);

    [DllImport("msvcr140d.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern int _CrtSetReportFile140(int reportType, IntPtr reportFile);

    [DllImport("msvcrtd.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void _set_abort_behavior(int flags, int mask);

    [DllImport("kernel32.dll")]
    public static extern uint SetErrorMode(uint mode);

    public const int _CRT_ASSERT = 2;
    public const int _CRT_ERROR = 1;
    public const int _CRTDBG_MODE_FILE = 0x1;
    public const int _CRTDBG_MODE_DEBUG = 0x2;
    public static readonly IntPtr _CRTDBG_FILE_STDERR = new(-3);

    public const int _WRITE_ABORT_MSG = 0x1;
    public const int _CALL_REPORTFAULT = 0x2;

    public const uint SEM_FAILCRITICALERRORS = 0x0001;
    public const uint SEM_NOGPFAULTERRORBOX = 0x0002;
    public const uint SEM_NOOPENFILEERRORBOX = 0x8000;

    public static void DisableAssertPopup()
    {
        try
        {
            _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
            _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
            _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
            _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
        }
        catch
        {
            try
            {
                _CrtSetReportMode140(_CRT_ASSERT, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
                _CrtSetReportFile140(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
            }
            catch
            {
                // Ignore if the debug CRT is unavailable.
            }
        }
    }

    public static void DisableAbortDialog()
    {
        try
        {
            _set_abort_behavior(0, _WRITE_ABORT_MSG);
            _set_abort_behavior(0, _CALL_REPORTFAULT);
        }
        catch
        {
            // Ignore if the runtime is unavailable.
        }
    }

    public static void DisableWindowsErrorDialog()
    {
        try
        {
            SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
        }
        catch
        {
            // Ignore if the API is unavailable.
        }
    }
}

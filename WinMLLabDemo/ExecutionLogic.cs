using Microsoft.ML.OnnxRuntime;
using Microsoft.Windows.AI.MachineLearning;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using IOPath = System.IO.Path;

namespace WinMLLabDemo
{
    internal static class ExecutionLogic
    {
        private static OrtEnv _ortEnv;
        private const string ModelName = "SqueezeNet";
        private const string ModelExtension = ".onnx";

        static ExecutionLogic()
        {
            // Create a new instance of EnvironmentCreationOptions
            EnvironmentCreationOptions envOptions = new()
            {
                logId = "WinMLLabDemo",
                logLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING
            };

            // Pass the options by reference to CreateInstanceWithOptions
            _ortEnv = OrtEnv.CreateInstanceWithOptions(ref envOptions);
        }

        public static IReadOnlyList<OrtEpDevice> LoadExecutionProviders()
        {
            // Get all the EPs available in the environment
            return _ortEnv.GetEpDevices();
        }

        public static async Task InitializeWinMLEPsAsync()
        {
            // TODO: Get the WinML EP catalog and download all EPs
        }

        public static string CompileModelForExecutionProvider(OrtEpDevice executionProvider)
        {
            string baseModelPath = IOPath.Combine(AppDomain.CurrentDomain.BaseDirectory, $"{ModelName}{ModelExtension}");
            string compiledModelPath = ModelHelpers.GetCompiledModelPath(executionProvider);

            try
            {
                var sessionOptions = GetSessionOptions(executionProvider);

                // TODO: Create compilation options, set the input and output, and compile
            }
            catch
            {
                throw new Exception($"Failed to create session with execution provider: {executionProvider.EpName}");
            }

            return compiledModelPath;
        }

        public static InferenceSession LoadModel(string compiledModelPath, OrtEpDevice executionProvider)
        {
            var sessionOptions = GetSessionOptions(executionProvider);

            // TODO: Return an inference session
            throw new NotImplementedException();
        }

        public static async Task<string> RunModelAsync(InferenceSession session, string imagePath, string compiledModelPath, OrtEpDevice executionProvider)
        {
            // Prepare inputs
            var inputs = await ModelHelpers.BindInputs(imagePath, session);

            // TODO: Run the inference, format and return the results
            throw new NotImplementedException();
        }

        private static SessionOptions GetSessionOptions(OrtEpDevice executionProvider)
        {
            // Create a session
            var sessionOptions = new SessionOptions();

            Dictionary<string, string> epOptions = new(StringComparer.OrdinalIgnoreCase);

            switch (executionProvider.EpName)
            {
                case "VitisAIExecutionProvider":
                    sessionOptions.AppendExecutionProvider(_ortEnv, [executionProvider], epOptions);
                    break;

                case "OpenVINOExecutionProvider":
                    // Configure threading for OpenVINO EP
                    epOptions["num_of_threads"] = "4";
                    sessionOptions.AppendExecutionProvider(_ortEnv, [executionProvider], epOptions);
                    break;

                case "QNNExecutionProvider":
                    // Configure performance mode for QNN EP
                    epOptions["htp_performance_mode"] = "high_performance";
                    sessionOptions.AppendExecutionProvider(_ortEnv, [executionProvider], epOptions);
                    break;

                case "NvTensorRTRTXExecutionProvider":
                    // Configure performance mode for TensorRT RTX EP
                    sessionOptions.AppendExecutionProvider(_ortEnv, [executionProvider], epOptions);
                    break;

                default:
                    break;
            }

            return sessionOptions;
        }
    }
}

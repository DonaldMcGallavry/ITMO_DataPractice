using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Emit;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.FastTree;

namespace ITMO_DataPractice
{
    class Program
    {
        private static string _dataPath = "bike_sharing.csv";
        public class BikeRentalData
        {
            [LoadColumn(0)]
            public float Season { get; set; }
            
            [LoadColumn(1)]
            public float Month { get; set; }
            [LoadColumn(2)]
            public float Hour { get; set; }
            [LoadColumn(3)]
            public float Holiday { get; set; }
            [LoadColumn(4)]
            public float Weekday { get; set; }
            [LoadColumn(5)]
            public float WorkingDay { get; set; }
            [LoadColumn(6)]
            public float WeatherCondition { get; set; }
            [LoadColumn(7)]
            public float Temperature { get; set; }
            [LoadColumn(8)]
            public float Humidity { get; set; }
            [LoadColumn(9)]
            public float Windspeed { get; set; }
            [LoadColumn(10)]
            public bool RentalType { get; set; }
        }
        public class RentalTypePrediction
        {
            [ColumnName("PredictedLabel")]
            public bool PredictedRentalType { get; set; }
            public float Probability { get; set; }
            public float Score { get; set; }
        }
        static void Main(string[] args)
        {
            Console.WriteLine("Предсказание типа аренды велосипеда");

            var mlContext = new MLContext(seed: 0);
            // 2. Загрузка данных
            var loader = mlContext.Data.CreateTextLoader(new TextLoader.Options
            {
                Separators = new [] { ',' },
                HasHeader = true,
                Columns = new[]
                {
                    new TextLoader.Column("Season", DataKind.Single, 0),
                    new TextLoader.Column("Month", DataKind.Single,1),
                    new TextLoader.Column("Hour", DataKind.Single, 2),
                    new TextLoader.Column("Holiday", DataKind.Single,3),
                    new TextLoader.Column("Weekday", DataKind.Single,4),
                    new TextLoader.Column("WorkingDay", DataKind.Single,5),
                    new TextLoader.Column("WeatherCondition", DataKind.Single,6),
                    new TextLoader.Column("Temperature", DataKind.Single,7),
                    new TextLoader.Column("Humidity", DataKind.Single,8),
                    new TextLoader.Column("Windspeed", DataKind.Single,9),
                    new TextLoader.Column("RentalType", DataKind.Boolean,10)
                }
            });
            var data = loader.Load(_dataPath);

            // 3. Разделение данных на обучающую и тестовую выборки
            var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

            // 4. Создание пайплайна обработки данных
            var pipeline = mlContext.Transforms.CopyColumns("Label", "RentalType")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("SeasonEncoded", "Season"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("WeatherEncoded", "WeatherCondition"))
                .Append(mlContext.Transforms.Concatenate("NumFeatures", "Month", "Hour", "WorkingDay", "Weekday", "Holiday", "Temperature", "Humidity", "Windspeed"))
                .Append(mlContext.Transforms.NormalizeMinMax("NumFeatures"))
                .Append(mlContext.Transforms.Concatenate("Features", "SeasonEncoded", "WeatherEncoded", "NumFeatures"));

            // 5. Обучение моделей и выбор лучшей
            var trainers = new (string name, IEstimator<ITransformer> trainer)[]
            {
                ("FastTree", mlContext.BinaryClassification.Trainers.FastTree(
                    labelColumnName: "Label", featureColumnName: "Features")),
                ("LightGBM", mlContext.BinaryClassification.Trainers.LightGbm(
                    labelColumnName: "Label", featureColumnName: "Features")),
                ("LogisticRegression", mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(
                    labelColumnName: "Label", featureColumnName: "Features"))
            };
            BinaryClassificationMetrics bestMet = null;
            ITransformer bestModel = null;
            string bestName = string.Empty;

            // 6. Оценка качества модели
            foreach (var (name, trainer) in trainers)
            {
                Console.WriteLine($"Training {name} _");
                var model = pipeline.Append(trainer).Fit(split.TrainSet);
                var prediction = model.Transform(split.TestSet);
                var metric = mlContext.BinaryClassification.Evaluate(prediction, "Label");

                Console.WriteLine($"{name}: \t AUC = {metric.AreaUnderRocCurve:P2}\t F1 = {metric.F1Score:P2}");

                if (bestMet == null || metric.F1Score > bestMet.F1Score)
                {
                    bestMet = metric;
                    bestModel = model;
                    bestName = name;
                }
            }
            // 7. Выполнение предсказаний
            Console.WriteLine($"Best model: {bestName} (AUC = {bestMet.AreaUnderRocCurve:P2}, F1 = {bestMet.F1Score:P2}\n");

            var predictor = mlContext.Model.CreatePredictionEngine<BikeRentalData, RentalTypePrediction>(bestModel);
            var sample = new BikeRentalData
            {
                Season = 2,
                Month = 11,
                Hour = 22,
                Holiday = 1,
                Weekday = 5,
                WorkingDay = 1,
                WeatherCondition = 1,
                Temperature = 14,
                Humidity = 60,
                Windspeed = 30
            };
            var result = predictor.Predict(sample);

            Console.WriteLine($"Sample prediction -> {(result.PredictedRentalType ? "Long-Term" : "Short-term")} (probability{result.Probability:P1})\n");
            mlContext.Model.Save(bestModel, data.Schema, "BikeModel.zip");
            Console.WriteLine("Данные сохранены");

            Console.WriteLine("Нажмите любую клавишу для завершения...");
            Console.ReadKey();
        }
    }
}

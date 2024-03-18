using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Accord.MachineLearning;
using Accord.MachineLearning.VectorMachines;
using Accord.Math.Distances;

class Program
{
	static List<string> allWords = new List<string>();
	static List<List<string>> strings = new List<List<string>>();
	static List<string> fileNames = new List<string>();

	static TFIDF vectorizer = new TFIDF()
	{
		Tf = TermFrequency.Log,
		Idf = InverseDocumentFrequency.Default
	};
	static void Main()
	{
		List<List<double>> vectors = new List<List<double>>();

		string inputDirectory = "../../../books/gutenberg_txt/gutenberg";
		ProcessFilesInDirectory(inputDirectory);

		for(int i = 0; i < strings.Count; i++)
		{
			vectors.Add(TextToVectorTFIDF(strings[i].ToArray(), vectorizer).ToList());
		}

		List<int> leadersId = new List<int>();
		Dictionary<int, List<int>> clusters = new Dictionary<int, List<int>>();

		for(int i = 0; i < Math.Sqrt(vectors.Count); i++)
		{
			Random random = new Random();
			int N = vectors.Count;
			int randomId = random.Next(0, N);
			while (leadersId.Contains(randomId))
			{
				randomId = random.Next(0, N);
			}

			leadersId.Add(randomId);
		}

		for(int i = 0; i < vectors.Count; i++)
		{
			if(leadersId.Contains(i))
			{
				continue;
			}

			double maxDictance = -1;
			int leaderId = -1;

			for(int j = 0; j < leadersId.Count; j++)
			{
				double distance = CalculateCosineSimilarity(vectors[i].ToArray(), vectors[leadersId[j]].ToArray());
				if(distance > maxDictance)
				{
					maxDictance = distance;
					leaderId = leadersId[j];
				}
			}

			if(clusters.ContainsKey(leaderId))
			{
				clusters[leaderId].Add(i);
			}
			else
			{
				clusters.Add(leaderId, new List<int>() { i});
			}

			Console.WriteLine($"Document {i} is in cluster {leaderId}");
        }

		string outputFile = "../../../clusters.txt";

		using (StreamWriter writer = new StreamWriter(outputFile))
		{
			foreach (var cluster in clusters)
			{
				writer.WriteLine($"Leader: {cluster.Key} - {fileNames[cluster.Key]} Count: {cluster.Value.Count}");
				foreach (var doc in cluster.Value)
				{
					writer.Write($"{doc} - {fileNames[doc]}; ");
				}
				writer.WriteLine();
			}
		}
	}

	static List<string> ReadDocument(string filePath)
	{
		var text = File.ReadAllText(filePath).Tokenize();
		var words = text.Where(x => ContainsEnglishCharacters(x)).Distinct().ToArray();
		vectorizer.Learn(words.Tokenize());
        Console.WriteLine($"Read {filePath}");
		strings.Add(words.ToList());
        return words.ToList();
	}
	static bool ContainsEnglishCharacters(string input)
	{
		return Regex.IsMatch(input, @"^[a-zA-Z]+$");
	}

	static double[] TextToVectorTFIDF(string[] text, TFIDF vectorizer)
	{
		double[] vector = vectorizer.Transform(text);

		return vector;
	}

	static double CalculateCosineSimilarity(double[] vectorA, double[] vectorB)
	{
		var cosine = new Cosine();

		double similarity = cosine.Similarity(vectorA, vectorB);

		return similarity;
	}
	static void ProcessFilesInDirectory(string inputDirectory)
	{
		foreach (string filePath in Directory.EnumerateFiles(inputDirectory, "*.txt"))
		{
			string fileName = Path.GetFileNameWithoutExtension(filePath);

			fileNames.Add(fileName);

			allWords.AddRange(ReadDocument(filePath));
		}

		foreach (string subdirectory in Directory.EnumerateDirectories(inputDirectory))
		{
			string subdirectoryName = Path.GetFileName(subdirectory);

			ProcessFilesInDirectory(subdirectory);
		}
	}
}

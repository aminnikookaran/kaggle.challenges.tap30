package tap30;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.functions.SimpleLinearRegression;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.M5Rules;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.M5P;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ArffLoader.ArffReader;

public class Tap30Weka {
	private static final int NUM_CELLS = 8;
	private static final int NUM_TIMES = 732;
	private static final int NUM_ITERATIONS = 10;
	public static void main(String[] args) throws Exception {
		List<List<List<Boolean>>> isTrain = new ArrayList<>();
		List<List<List<Double>>> requests = new ArrayList<>();
		for (int i = 0; i < NUM_CELLS; i ++) {
			List<List<Boolean>> trainOrTestPerRow = new ArrayList<>();
			List<List<Double>> requestsPerRow = new ArrayList<>();
			for (int j = 0; j < NUM_CELLS; j ++) {
				trainOrTestPerRow.add(new ArrayList<>());
				requestsPerRow.add(new ArrayList<>());
			}
			isTrain.add(trainOrTestPerRow);
			requests.add(requestsPerRow);
		}

		List<Double> requestSums = new ArrayList<>();
		double requestCount = 0;
		String line;
		BufferedReader bufferedReader = new BufferedReader(new FileReader("data.txt"));
		bufferedReader.readLine();
		bufferedReader.readLine();
		int row = 0;
		while ((line = bufferedReader.readLine()) != null) {
			String[] values = line.split(" ");
			for (int column = 0; column < NUM_CELLS; column++) {
				double value = Double.parseDouble(values[column]);
				if (value < 0)
					isTrain.get(row).get(column).add(false);
				else
					isTrain.get(row).get(column).add(true);
				value = Math.max(value, 0);
				requests.get(row).get(column).add(value);
				requestCount += value;
			}
			row++;
			if (row == NUM_CELLS) {
				row = 0;
				requestSums.add(requestCount);
				requestCount = 0;
			}
		}
		bufferedReader.close();

		for (int k = 1; k < NUM_TIMES; k++) {
			for (int i = 0; i < NUM_CELLS; i++) {
				for (int j = 0; j < NUM_CELLS; j++) {
					if (isTrain.get(i).get(j).get(k))
						continue;
					double value = requests.get(i).get(j).get(k - 1);
					requests.get(i).get(j).set(k, value);
					requestSums.set(k, requestSums.get(k) + value);
				}
			}
		}

		for (int k = (NUM_TIMES-2); k >= 0; k--) {
			for (int i = 0; i < NUM_CELLS; i++) {
				for (int j = 0; j < NUM_CELLS; j++) {
					if (isTrain.get(i).get(j).get(k))
						continue;
					double preValue = requests.get(i).get(j).get(k);
					double value = requests.get(i).get(j).get(k + 1);
					double newValue = (preValue + value) / 2;
					requests.get(i).get(j).set(k, newValue);
					requestSums.set(k, (requestSums.get(k) - preValue) + newValue);
				}
			}
		}

		for (int iteration = 1; iteration <= NUM_ITERATIONS; iteration++) {
			BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter("data.arff"));
			bufferedWriter.write("@relation RequestFeatures\n\n");
			for (int i = 0; i < 8; i++)
				bufferedWriter.write("@attribute " + i + " numeric\n");
			bufferedWriter.write("\n@data\n");
			for (int k = 0; k < NUM_TIMES; k++) {
				for (int i = 0; i < NUM_CELLS; i++) {
					for (int j = 0; j < NUM_CELLS; j++) {
						if (!isTrain.get(i).get(j).get(k))
							continue;
						bufferedWriter.write(String.valueOf(i));
						bufferedWriter.write(',');
						bufferedWriter.write(String.valueOf(j));
						bufferedWriter.write(',');
						if (k == 0)
							bufferedWriter.write('0');
						else
							bufferedWriter.write(String.valueOf(requests.get(i).get(j).get(k-1)));
						bufferedWriter.write(',');
						if (k == (NUM_TIMES-1))
							bufferedWriter.write('0');
						else
							bufferedWriter.write(String.valueOf(requests.get(i).get(j).get(k+1)));
						bufferedWriter.write(',');
						if (k == 0)
							bufferedWriter.write('0');
						else
							bufferedWriter.write(String.valueOf(requestSums.get(k-1)));
						bufferedWriter.write(',');
						bufferedWriter.write(String.valueOf(requestSums.get(k)));
						bufferedWriter.write(',');
						if (k == (NUM_TIMES-1))
							bufferedWriter.write('0');
						else
							bufferedWriter.write(String.valueOf(requestSums.get(k+1)));
						bufferedWriter.write(',');
						bufferedWriter.write(String.valueOf(requests.get(i).get(j).get(k)));
						bufferedWriter.write('\n');
					}
				}
			}
			bufferedWriter.close();

			bufferedReader = new BufferedReader(new FileReader("data.arff"));
			ArffReader arffReader = new ArffReader(bufferedReader, 1000);
			Instances instances = arffReader.getStructure();
			instances.setClassIndex(instances.numAttributes() - 1);
			Instance instance;
			while ((instance = arffReader.readInstance(instances)) != null)
				instances.add(instance);

			Classifier classifier = new RandomForest();
			classifier.buildClassifier(instances);

			Evaluation evaluation = new Evaluation(instances);
			evaluation.evaluateModel(classifier, instances);
			System.out.println("Iteration " + iteration + ":\t" + evaluation.rootMeanSquaredError());

			for (int k = 0; k < NUM_TIMES; k++) {
				for (int i = 0; i < NUM_CELLS; i++) {
					for (int j = 0; j < NUM_CELLS; j++) {
						if (isTrain.get(i).get(j).get(k))
							continue;
						double[] attributes = new double[8];
						attributes[0] = i;
						attributes[1] = j;
						if (k == 0)
							attributes[2] = 0;
						else
							attributes[2] = requests.get(i).get(j).get(k-1);
						if (k == (NUM_TIMES-1))
							attributes[3] = 0;
						else
							attributes[3] = requests.get(i).get(j).get(k+1);
						if (k == 0)
							attributes[4] = 0;
						else
							attributes[4] = requestSums.get(k-1);
						attributes[5] = requestSums.get(k);
						if (k == (NUM_TIMES-1))
							attributes[6] = 0;
						else
							attributes[6] = requestSums.get(k+1);
						attributes[7] = requests.get(i).get(j).get(k);

						instance = new DenseInstance(1.0, attributes);
						instance.setDataset(instances);
						double value = classifier.classifyInstance(instance);
						value = Math.max(value, 0);

						requests.get(i).get(j).set(k, value);
						requestSums.set(k, (requestSums.get(k) - attributes[7]) + value);
					}
				}
			}
		}

		BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter("data.csv"));
		bufferedWriter.write("id,demand\n");
		for (int k = 0; k < NUM_TIMES; k++) {
			for (int i = 0; i < NUM_CELLS; i++) {
				for (int j = 0; j < NUM_CELLS; j++) {
					if (isTrain.get(i).get(j).get(k))
						continue;
					bufferedWriter.write(String.valueOf(k));
					bufferedWriter.write(':');
					bufferedWriter.write(String.valueOf(i));
					bufferedWriter.write(':');
					bufferedWriter.write(String.valueOf(j));
					bufferedWriter.write(',');
					bufferedWriter.write(String.valueOf(requests.get(i).get(j).get(k)));
					bufferedWriter.write('\n');
				}
			}
		}
		bufferedWriter.close();

		bufferedWriter = new BufferedWriter(new FileWriter("data2.txt"));
		bufferedWriter.write("732\n8 8\n");
		for (int k = 0; k < NUM_TIMES; k++) {
			for (int i = 0; i < NUM_CELLS; i++) {
				for (int j = 0; j < NUM_CELLS; j++) {
					bufferedWriter.write(String.valueOf(Math.round(requests.get(i).get(j).get(k))));
					if (j < (NUM_CELLS - 1))
						bufferedWriter.write(' ');
				}
				bufferedWriter.write('\n');
			}
		}
		bufferedWriter.close();
	}
}
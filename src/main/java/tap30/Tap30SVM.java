package tap30;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.trees.REPTree;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ArffLoader.ArffReader;

public class Tap30SVM {
	// LinearRegression 92 93 59
	// MultilayerPerceptron 79 93 70
	// SimpleLinearRegression 119 119 75
	// DecisionTable 94 137 106
	// M5Rules 69 75 52
	// DecisionStump 141 91 125
	// M5P 67 76 50
	// RandomForest 27 71 22
	// RandomTree 3 103 3
	// REPTree 78 91 56

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

		for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
			BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter("data.dat"));
			for (int k = 0; k < NUM_TIMES; k++) {
				for (int i = 0; i < NUM_CELLS; i++) {
					for (int j = 0; j < NUM_CELLS; j++) {
						if (!isTrain.get(i).get(j).get(k))
							continue;
						bufferedWriter.write(String.valueOf(requests.get(i).get(j).get(k)));
						bufferedWriter.write(" 1:");
						bufferedWriter.write(String.valueOf(i));
						bufferedWriter.write(" 2:");
						bufferedWriter.write(String.valueOf(j));
						bufferedWriter.write(" 3:");
						if (k == 0)
							bufferedWriter.write('0');
						else
							bufferedWriter.write(String.valueOf(requests.get(i).get(j).get(k-1)));
						bufferedWriter.write(" 4:");
						if (k == (NUM_TIMES-1))
							bufferedWriter.write('0');
						else
							bufferedWriter.write(String.valueOf(requests.get(i).get(j).get(k+1)));
						bufferedWriter.write(" 5:");
						if (k == 0)
							bufferedWriter.write('0');
						else
							bufferedWriter.write(String.valueOf(requestSums.get(k-1)));
						bufferedWriter.write(" 6:");
						bufferedWriter.write(String.valueOf(requestSums.get(k)));
						bufferedWriter.write(" 7:");
						if (k == (NUM_TIMES-1))
							bufferedWriter.write('0');
						else
							bufferedWriter.write(String.valueOf(requestSums.get(k+1)));
						bufferedWriter.write('\n');
					}
				}
			}
			bufferedWriter.close();

			ProcessBuilder processBuilder = new ProcessBuilder("svm_learn","-z","r","data.dat","data.mdl");
			processBuilder.redirectErrorStream(true);
			Process process = processBuilder.start();
			bufferedReader =	new BufferedReader(new InputStreamReader(process.getInputStream()));
			while ((line = bufferedReader.readLine())!= null) {}
			bufferedReader.close();
			process.waitFor();

			processBuilder = new ProcessBuilder("svm_classify","data.dat","data.mdl","data.pdt");
			processBuilder.redirectErrorStream(true);
			process = processBuilder.start();
			bufferedReader = new BufferedReader(new InputStreamReader(process.getInputStream()));
			while ((line = bufferedReader.readLine())!= null) {}
			bufferedReader.close();
			process.waitFor();

			double RMS = 0;
			int RMScount = 0;
			BufferedReader requestsBufferedReader = new BufferedReader(new FileReader("data.dat"));
			BufferedReader predictionsBufferedReader = new BufferedReader(new FileReader("data.pdt"));
			while ((line = requestsBufferedReader.readLine()) != null) {
				String prediction = predictionsBufferedReader.readLine();
				double predictionValue = Double.parseDouble(prediction);
				String[] split = line.split(" ");
				double request = Double.parseDouble(split[0]);
				RMS += Math.pow(request - predictionValue, 2);
				RMScount++;
			}
			predictionsBufferedReader.close();
			requestsBufferedReader.close();

			RMS /= RMScount;
			RMS = Math.sqrt(RMS);

			System.out.println("Iteration " + iteration + ":\t" + RMS);

			bufferedWriter = new BufferedWriter(new FileWriter("data.dat"));
			for (int k = 0; k < NUM_TIMES; k++) {
				for (int i = 0; i < NUM_CELLS; i++) {
					for (int j = 0; j < NUM_CELLS; j++) {
						if (isTrain.get(i).get(j).get(k))
							continue;
						bufferedWriter.write(String.valueOf(requests.get(i).get(j).get(k)));
						bufferedWriter.write(" 1:");
						bufferedWriter.write(String.valueOf(i));
						bufferedWriter.write(" 2:");
						bufferedWriter.write(String.valueOf(j));
						bufferedWriter.write(" 3:");
						if (k == 0)
							bufferedWriter.write('0');
						else
							bufferedWriter.write(String.valueOf(requests.get(i).get(j).get(k-1)));
						bufferedWriter.write(" 4:");
						if (k == (NUM_TIMES-1))
							bufferedWriter.write('0');
						else
							bufferedWriter.write(String.valueOf(requests.get(i).get(j).get(k+1)));
						bufferedWriter.write(" 5:");
						if (k == 0)
							bufferedWriter.write('0');
						else
							bufferedWriter.write(String.valueOf(requestSums.get(k-1)));
						bufferedWriter.write(" 6:");
						bufferedWriter.write(String.valueOf(requestSums.get(k)));
						bufferedWriter.write(" 7:");
						if (k == (NUM_TIMES-1))
							bufferedWriter.write('0');
						else
							bufferedWriter.write(String.valueOf(requestSums.get(k+1)));
						bufferedWriter.write('\n');
					}
				}
			}
			bufferedWriter.close();

			processBuilder = new ProcessBuilder("svm_classify","data.dat","data.mdl","data.pdt");
			processBuilder.redirectErrorStream(true);
			process = processBuilder.start();
			bufferedReader = new BufferedReader(new InputStreamReader(process.getInputStream()));
			while ((line = bufferedReader.readLine())!= null) {}
			bufferedReader.close();
			process.waitFor();

			bufferedReader = new BufferedReader(new FileReader("data.pdt"));
			for (int k = 0; k < NUM_TIMES; k++) {
				for (int i = 0; i < NUM_CELLS; i++) {
					for (int j = 0; j < NUM_CELLS; j++) {
						if (isTrain.get(i).get(j).get(k))
							continue;
						double value = Math.max(Double.parseDouble(bufferedReader.readLine()), 0);
						double preValue = requests.get(i).get(j).get(k);
						requests.get(i).get(j).set(k, value);
						requestSums.set(k, (requestSums.get(k) - preValue) + value);
					}
				}
			}
			bufferedReader.close();
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
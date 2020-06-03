package nndemo;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class FirstNN {
	private static final String MNIST_DATASET_ROOT_FOLDER = System.getProperty("user.home") + File.separator
			+ "Development" + File.separator + "ml" + File.separator + "mnist_png" + File.separator;
	private static final int HEIGHT = 28;
	private static final int WIDTH = 28;
	private static final int N_OUTCOMES = 10;

	public static ListDataSetIterator<DataSet> getDatasetIterator(String datafile)
			throws IOException, ClassNotFoundException {
		ObjectInputStream ois = new ObjectInputStream(new FileInputStream(MNIST_DATASET_ROOT_FOLDER + datafile));
		DataSet dataSet = (DataSet) ois.readObject();
		ois.close();

		List<DataSet> listDataSet = dataSet.asList();
		Collections.shuffle(listDataSet, new Random(System.currentTimeMillis()));
		int batchSize = 10;
		return new ListDataSetIterator<DataSet>(listDataSet, batchSize);
	}
	
	public static void main(String[] args) throws Exception {
		long t0 = System.currentTimeMillis();

		DataSetIterator dsi = getDatasetIterator("mnist_train.obj");
		int nEpochs = 20;

		System.out.println("Build model....");
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123)
				.updater(new Sgd(3))
				// .l2(1e-4)
				.list()
				.layer(new DenseLayer.Builder()
						.nIn(HEIGHT * WIDTH).nOut(30).activation(Activation.SIGMOID).weightInit(WeightInit.NORMAL)
						.build())
				.layer(new OutputLayer.Builder(LossFunction.SQUARED_LOSS).nIn(30).nOut(N_OUTCOMES)
						.activation(Activation.SIGMOID).weightInit(WeightInit.NORMAL).build())
				.build();
		
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		model.setListeners(new ScoreIterationListener(1));
		System.out.println("Train model....");
		model.fit(dsi, nEpochs);

		DataSetIterator testDsi = getDatasetIterator("mnist_test.obj");

		System.out.println("Evaluate model....");
		Evaluation eval = model.evaluate(testDsi);
		System.out.println(eval.stats());

		long t1 = System.currentTimeMillis();
		double t = (double)(t1 - t0) / 1000.0;
		System.out.println("\n\nTotal time: "+t+" seconds");
	}
}

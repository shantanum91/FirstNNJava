package FirstNN;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;

import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

public class SaveDSI {

	private static final String MNIST_DATASET_ROOT_FOLDER = System.getProperty("user.home") + File.separator
			+ "Development" + File.separator + "ml" + File.separator + "mnist_png" + File.separator;
	private static final int HEIGHT = 28;
	private static final int WIDTH = 28;
	private static final int N_SAMPLES_TRAINING = 60000;
	private static final int N_SAMPLES_TESTING = 10000;
	private static final int N_OUTCOMES = 10;

	private static DataSet getDataSetIterator(String folderPath, int nSamples) throws IOException {

		File folder = new File(folderPath);
		File[] digitFolders = folder.listFiles();

		NativeImageLoader nil = new NativeImageLoader(HEIGHT, WIDTH);
		ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);

		INDArray input = Nd4j.create(new int[] { nSamples, HEIGHT * WIDTH });
		INDArray output = Nd4j.create(new int[] { nSamples, N_OUTCOMES });

		int n = 0;

		for (File digitFolder : digitFolders) {
			int labelDigit = Integer.parseInt(digitFolder.getName());

			File[] imageFiles = digitFolder.listFiles();
			for (File imageFile : imageFiles) {
				// read the image as a one dimensional array of 0..255 values
				INDArray img = nil.asRowVector(imageFile);
				// scale the 0..255 integer values into a 0..1 floating range
				scaler.transform(img);
				// copy the img array into the input matrix, in the next row
				input.putRow(n, img);
				// in the same row of the output matrix, fire (set to 1 value) the column
				// correspondent to the label
				output.put(n, labelDigit, 1.0);
				n++;
			}
		}

		return new DataSet(input, output);
	}

	public static void main(String[] args) throws IOException {
		DataSet ds = getDataSetIterator(MNIST_DATASET_ROOT_FOLDER + "training", N_SAMPLES_TRAINING);

		ObjectOutputStream os = new ObjectOutputStream(
				new FileOutputStream(MNIST_DATASET_ROOT_FOLDER + "mnist_train.obj"));
		os.writeObject(ds);
		os.flush();
		os.close();

		DataSet ds2 = getDataSetIterator(MNIST_DATASET_ROOT_FOLDER + "testing", N_SAMPLES_TESTING);

		ObjectOutputStream os2 = new ObjectOutputStream(new FileOutputStream(MNIST_DATASET_ROOT_FOLDER + "mnist_test.obj"));
		os2.writeObject(ds2);
		os2.flush();
		os2.close();

		System.out.println("Done");
	}

}

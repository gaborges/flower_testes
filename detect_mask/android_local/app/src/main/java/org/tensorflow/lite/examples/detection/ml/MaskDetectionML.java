package org.tensorflow.lite.examples.detection.ml;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;


public class MaskDetectionML {

    private final Context context;
    private Interpreter tfliteInterpreter;

    private static final String classes[] = {"Unknown","With Mak","Without Mask"};


    public static final int INPUT_IMAGE_SIZE = 224;
    public static final int NUM_CLASSES = 2;

    public MaskDetectionML(Context context) throws IOException {
        this.context = context;
        try {
            MappedByteBuffer tfliteModel = loadModelFile(context.getAssets(), "model.tflite");
            Interpreter.Options options = new Interpreter.Options();
            tfliteInterpreter = new Interpreter(tfliteModel, options);

        } catch (IOException e) {
            // TODO Handle the exception
            Log.e("ERRORR",e.toString());
            throw e;
        }
    }

    // Preprocess the input image (resize and normalize)
    private float[][][][] preprocessImage(Bitmap bitmap) {
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, true);
        float[][][][] inputTensor = new float[1][INPUT_IMAGE_SIZE][INPUT_IMAGE_SIZE][3];
        for (int y = 0; y < INPUT_IMAGE_SIZE; y++) {
            for (int x = 0; x < INPUT_IMAGE_SIZE; x++) {
                int pixel = resizedBitmap.getPixel(x, y);
                inputTensor[0][y][x][0] = ((pixel >> 16) & 0xFF) / 255.0f;
                inputTensor[0][y][x][1] = ((pixel >> 8) & 0xFF) / 255.0f;
                inputTensor[0][y][x][2] = (pixel & 0xFF) / 255.0f;
            }
        }
        return inputTensor;
    }

    public String inferMaskState(Bitmap image){
        // Preprocess the input image
        float[][][][] inputTensor = preprocessImage(image);

        // Prepare the output tensor for the model
        float[][] outputTensor = new float[1][NUM_CLASSES];

        // Run the inference
        tfliteInterpreter.run(inputTensor, outputTensor);

        // Find the predicted class with the highest probability
        int predictedClass = -1;
        float maxProb = -1.0f;
        for (int i = 0; i < NUM_CLASSES; i++) {
            if (outputTensor[0][i] > maxProb) {
                predictedClass = i;
                maxProb = outputTensor[0][i];
            }
        }

        return classes[predictedClass+1];

    }

    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
}

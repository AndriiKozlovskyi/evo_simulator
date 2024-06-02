package org.example;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class NetworkManager {
    private MultiLayerNetwork model;
    private int inputSize;
    private int outputSize;

    public NetworkManager(int inputSize, int outputSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        init();
    }

    public MultiLayerNetwork init() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.001))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(inputSize).nOut(32)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(32).nOut(16)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(16).nOut(outputSize)
                        .build())
                .build();

        this.model = new MultiLayerNetwork(conf);
        this.model.init();
        this.model.setListeners(new ScoreIterationListener(10));
        return this.model;
    }

    public MultiLayerNetwork getModel() {
        return model;
    }
}

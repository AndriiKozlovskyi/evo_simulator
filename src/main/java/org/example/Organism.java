package org.example;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.Random;

public class Organism {
    MultiLayerNetwork model;
    private double x;
    private double y;
    private double speed;
    private double energy;
    private double movementCostInEnergyPoints = 0.1;
    private int offspringCount;

    private int inputSize;
    private int outputSize;

    public Organism(int inputSize, int outputSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.model = new NetworkManager(inputSize, outputSize).getModel();

    }

    public double[] decide(double[] state) {
        INDArray input = Nd4j.create(state, new int[]{1, state.length});
        INDArray output = model.output(input);
        return output.toDoubleVector();
    }

    private void move(double[] action) {
        decreaseEnergyLevel(movementCostInEnergyPoints);
        this.x = this.x + action[0] * speed;
        this.y = this.y + action[1] * speed;
    }

    private void decreaseEnergyLevel(double energyPoints) {
        this.energy -= energyPoints;
    }


    public void update(double[] state, double[] action, List<Organism> population) {
        move(action);
    }

    private void decideToReproduce(double[] actions) {
        if(actions[2] > 0.5 && this.energy > 20) {
            this.energy -= 20;
            this.offspringCount ++;
        }
    }

    public Organism reproduce() {
        Organism offspring = new Organism(this.inputSize, this.outputSize);
        offspring.model.setParams(this.model.params().dup());

        Random rand = new Random();
        INDArray params = offspring.model.params();
        for (int i = 0; i < params.length(); i++) {
            if (rand.nextDouble() < 0.05) {
                params.putScalar(i, params.getDouble(i) + rand.nextGaussian() * 0.1);
            }
        }
        offspring.model.setParams(params);
        offspring.energy = this.energy; // Начальная энергия потомка
        offspring.x = this.x;
        offspring.y = this.y;
        return offspring;
    }

}

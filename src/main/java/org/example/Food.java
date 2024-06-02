package org.example;

import java.util.Random;
public class Food {
    private int x, y;

    public Food(int min, int max) {
        this.x = new Random().nextInt(min, max);
        this.y = new Random().nextInt(min, max);
    }

    public int getX() {
        return this.x;
    }

    public int getY() {
        return this.y;
    }

    public void setX(int x) {
        this.x = x;
    }

    public void setY(int y) {
        this.y = y;
    }
}

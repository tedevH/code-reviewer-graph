package com.shop;

public class Order {
    private final double[] items;

    public Order(double[] items) {
        this.items = items;
    }

    public String total() {
        return Util.fmt(sum());
    }

    double sum() {
        double t = 0;
        for (double x : items) t += x;
        return t;
    }
}

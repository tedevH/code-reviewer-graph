package com.shop.api;

import com.shop.Order;

public class Api {
    public String handle(double[] items) {
        return new Order(items).total();
    }
}

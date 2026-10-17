import { fmt } from "./util.js";
import React from "react";

export class Order {
  constructor(private items: number[]) {}

  total(): string {
    return fmt(this.sum());
  }

  sum(): number {
    return this.items.reduce((a, b) => a + b, 0);
  }
}

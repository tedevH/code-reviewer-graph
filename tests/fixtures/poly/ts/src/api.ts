import { Order } from "./order";
import * as util from "./util";

export const handler = (items: number[]) => {
  const order = new Order(items);
  return util.fmt(order.sum());
};

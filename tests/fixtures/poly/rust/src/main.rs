mod order;
mod util;

use crate::order::Order;

fn main() {
    let o = Order::new(vec![1.0]);
    println!("{}", o.total());
}

use crate::util::fmt;

pub struct Order {
    items: Vec<f64>,
}

impl Order {
    pub fn new(items: Vec<f64>) -> Self {
        Order { items }
    }

    pub fn total(&self) -> String {
        fmt(self.sum())
    }

    fn sum(&self) -> f64 {
        self.items.iter().sum()
    }
}

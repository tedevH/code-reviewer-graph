#include <stdio.h>
#include "util.h"

static double sum(double *xs, int n) {
    double t = 0;
    for (int i = 0; i < n; i++) t += xs[i];
    return t;
}

int order_total(double *xs, int n) {
    return fmt_cents(sum(xs, n));
}

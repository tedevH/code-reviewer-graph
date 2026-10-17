<?php
namespace App;

use App\Util;

class Order {
    public function total(): string {
        return Util::fmt($this->sum());
    }

    private function sum(): float {
        return 1.0;
    }
}
